import itertools
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from neuralop.models import SFNO
import torch
from copy import deepcopy
import torch.nn as nn
from utils import CODANODataset
from tqdm import tqdm
from pytorch_msssim import MS_SSIM
import torch.optim as optim
from physics import PhysicsLoss
import numpy as np


MSSSIM_MODULE = MS_SSIM(
    data_range=1.0,
    size_average=True,
    channel=1,  # after unsqueeze(1)
    spatial_dims=3,  # 3D input
    win_size=7,
    win_sigma=1.0,
)


def train_cv(
    data_path,
    cr_dirs,
    hyperparams: dict,
    n_splits: int,
    n_epochs: int,
    batch_size: int,
    loss_fn,
    device: str,
):
    """
    Cross-validation training over hyperparameter grid.

    Parameters
    ----------
    data_path : str
        Root path where CR directories are stored
    hyperparams : dict
        Hyperparameter dictionary: {param_name: [list of values]}
        Must contain keys: "n_modes", "hidden_channels", "projection_channel_ratio", "factorization"
    n_splits : int
        Number of KFold splits (CR-level)
    n_epochs : int
        Number of epochs
    batch_size : int
        Batch size
    loss_fn : torch.nn.Module
        Loss function
    device : str
        "cuda" or "cpu"

    Returns
    -------
    results : list of dict
        Each dict has: hyperparameters, avg val loss, avg best epoch
    """

    # 2. Build hyperparameter grid
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    param_combinations = list(itertools.product(*values))

    # 3. Setup KFold on cr_dirs
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for param_tuple in param_combinations:
        param_dict = dict(zip(keys, param_tuple))
        print(f"\n=== Training with hyperparameters: {param_dict} ===")

        fold_val_losses = []
        fold_val_mse = []
        fold_val_acc = []
        fold_val_psnr = []
        fold_val_mse_masked = []
        fold_val_msssim = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(cr_dirs)):
            print(f"\nFold {fold+1}/{n_splits}")

            train_crs = [cr_dirs[i] for i in train_idx]
            val_crs = [cr_dirs[i] for i in val_idx]

            train_dataset = CODANODataset(data_path, train_crs)
            val_dataset = CODANODataset(
                data_path,
                val_crs,
                br_min=train_dataset.br_min,
                br_max=train_dataset.br_max,
                bt_min=train_dataset.bt_min,
                bt_max=train_dataset.bt_max,
                bp_max=train_dataset.bp_max,
                bp_min=train_dataset.bp_min,
                jr_min=train_dataset.jr_min,
                jr_max=train_dataset.jr_max,
                jt_min=train_dataset.jt_min,
                jt_max=train_dataset.jt_max,
                jp_max=train_dataset.jp_max,
                jp_min=train_dataset.jp_min,
            )
            print(len(val_dataset))

            # Instantiate model using full param_dict
            model = SFNO(
                n_modes=(110, 128),
                in_channels=1,
                out_channels=139,
                hidden_channels=param_dict["hidden_channels"],
                projection_channel_ratio=2,
                factorization="dense",
                n_layers=param_dict["n_layers"],
            )

            # Train one model
            (
                train_losses,
                val_losses,
                train_mse,
                val_mse,
                train_mse_masked,
                val_mse_masked,
                train_msssim,
                val_msssim,
                train_acc,
                val_acc,
                train_psnr,
                val_psnr,
                best_epoch,
                best_state_dict,
            ) = train(
                model,
                train_dataset,
                val_dataset,
                n_epochs=n_epochs,
                batch_size=batch_size,
                loss_fn=loss_fn,
                device=device,
                lr=8e-4,
                weight_decay=0.0,
                verbose=False,
            )

            fold_val_losses.append(val_losses[best_epoch])
            fold_val_mse.append(val_mse[best_epoch])
            fold_val_acc.append(val_acc[best_epoch])
            fold_val_psnr.append(val_psnr[best_epoch])
            fold_val_mse_masked.append(val_mse_masked[best_epoch])
            fold_val_msssim.append(val_msssim[best_epoch])

        results.append(
            {
                "hyperparameters": param_dict,
                "val_loss": fold_val_losses,
                "val_mse": fold_val_mse,
                "val_acc": fold_val_acc,
                "val_psnr": fold_val_psnr,
                "val_mse_masked": fold_val_mse_masked,
                "val_msssim": fold_val_msssim,
            }
        )

    return results


def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    n_epochs: int,
    batch_size: int,
    loss_fn: nn.Module,
    device: str,
    lr: float = 8e-4,
    weight_decay: float = 0.0,
    verbose=True,
    physics_informed=False,
):
    """
    Train a model on train_dataset, validate on val_dataset.

    Returns
    -------
    train_losses : list of float
    val_losses : list of float
    best_epoch : int
    best_state_dict : model weights at best epoch
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    autocast_device_type = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler(device=autocast_device_type, enabled=False)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    train_losses = []
    val_losses = []

    train_physics_loss = []
    val_physics_loss = []

    if physics_informed:
        physics_loss = PhysicsLoss(
            r=train_dataset.r,
            theta=train_dataset.theta,
            phi=train_dataset.phi,
            br_min=train_dataset.br_min,
            bt_min=train_dataset.bt_min,
            bp_min=train_dataset.bp_min,
            br_max=train_dataset.br_max,
            bt_max=train_dataset.bt_max,
            bp_max=train_dataset.bp_max,
            jr_min=train_dataset.jr_min,
            jt_min=train_dataset.jt_min,
            jp_min=train_dataset.jp_min,
            jr_max=train_dataset.jr_max,
            jt_max=train_dataset.jt_max,
            jp_max=train_dataset.jp_max,
            device=device,
        ).to(device)
    # train_mse = []
    # val_mse = []
    # train_mse_masked = []
    # val_mse_masked = []
    # train_msssim = []
    # val_msssim = []
    # train_acc = []
    # val_acc = []
    # train_psnr = []
    # val_psnr = []

    # climatology = train_dataset.climatology.to(device)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_physics_loss = 0.0
        # running_mse = 0.0
        # running_mse_masked = 0.0
        # running_msssim = 0.0
        # running_acc = 0.0
        # running_psnr = 0.0
        for x, y in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False
        ):
            optimizer.zero_grad()
            pred = model(x.to(device))
            
            pred = pred.view(*y.shape)

            data_loss = loss_fn(pred, y.to(device))

            loss = data_loss

            br_pred, bt_pred, bp_pred, jr_pred, jt_pred, jp_pred = (
                pred[:, 0, :, :],
                pred[:, 1, :, :],
                pred[:, 2, :, :],
                pred[:, 3, :, :],
                pred[:, 4, :, :],
                pred[:, 5, :, :],
            )

            if physics_informed:
                physics_loss_value = physics_loss(
                    br_pred, bt_pred, bp_pred, jr_pred, jt_pred, jp_pred
                )
                running_physics_loss += physics_loss_value.item() * x.size(0)
                loss += physics_loss_value

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            # real_y = y * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            # real_pred = pred * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
            # running_mse += mse_score(real_y, real_pred)
            # running_mse_masked += mse_score_masked(real_y, real_pred, sobel_edge_map(y))
            # running_msssim += mssim_score(MSSSIM_MODULE, y, pred)
            # running_acc += acc_score(y, pred, climatology)
            # running_psnr += psnr_score(y, pred)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        if physics_informed:
            epoch_train_physics_loss = running_physics_loss / len(train_loader.dataset)
            train_physics_loss.append(epoch_train_physics_loss)
        # epoch_train_mse = running_mse / len(train_loader)
        # epoch_train_mse_masked = running_mse_masked / len(train_loader)
        # epoch_train_msssim = running_msssim / len(train_loader)
        # epoch_train_acc = running_acc / len(train_loader)
        # epoch_train_psnr = running_psnr / len(train_loader)
        train_losses.append(epoch_train_loss)
        # train_mse.append(epoch_train_mse)
        # train_mse_masked.append(epoch_train_mse_masked)
        # train_msssim.append(epoch_train_msssim)
        # train_acc.append(epoch_train_acc)
        # train_psnr.append(epoch_train_psnr)

        scheduler.step(epoch_train_loss)

        # Validation
        model.eval()
        running_loss = 0.0
        running_physics_loss = 0.0
        # running_mse = 0.0
        # running_mse_masked = 0.0
        # running_msssim = 0.0
        # running_acc = 0.0
        # running_psnr = 0.0
        with torch.no_grad():
            for x, y in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False
            ):
                pred = model(x.to(device))
                
                pred = pred.view(*y.shape)

                data_loss = loss_fn(pred, y.to(device))

                loss = data_loss

                br_pred, bt_pred, bp_pred, jr_pred, jt_pred, jp_pred = (
                    pred[:, 0, :, :],
                    pred[:, 1, :, :],
                    pred[:, 2, :, :],
                    pred[:, 3, :, :],
                    pred[:, 4, :, :],
                    pred[:, 5, :, :],
                )

                if physics_informed:
                    physics_loss_value = physics_loss(
                        br_pred, bt_pred, bp_pred, jr_pred, jt_pred, jp_pred
                    )
                    running_physics_loss += physics_loss_value.item() * x.size(0)
                    loss += physics_loss_value

                running_loss += loss.item() * x.size(0)
                # real_y = y * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                # real_pred = pred * (train_dataset.v_max - train_dataset.v_min) + train_dataset.v_min
                # running_mse += mse_score(real_y, real_pred)
                # running_mse_masked += mse_score_masked(real_y, real_pred, sobel_edge_map(y))
                # running_msssim += mssim_score(MSSSIM_MODULE, y, pred)
                # running_acc += acc_score(y, pred, climatology)
                # running_psnr += psnr_score(y, pred)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        if physics_informed:
            epoch_val_physics_loss = running_physics_loss / len(val_loader.dataset)
            val_physics_loss.append(epoch_val_physics_loss)
        # epoch_val_mse = running_mse / len(val_loader)
        # epoch_val_mse_masked = running_mse_masked / len(val_loader)
        # epoch_val_msssim = running_msssim / len(val_loader)
        # epoch_val_acc = running_acc / len(val_loader)
        # epoch_val_psnr = running_psnr / len(val_loader)
        val_losses.append(epoch_val_loss)
        # val_mse.append(epoch_val_mse)
        # val_mse_masked.append(epoch_val_mse_masked)
        # val_msssim.append(epoch_val_msssim)
        # val_acc.append(epoch_val_acc)
        # val_psnr.append(epoch_val_psnr)

        if verbose:
            print(
                f"Epoch {epoch+1}:\n",
                f"Train Loss = {epoch_train_loss:.6f} | Val Loss = {epoch_val_loss:.6f}",
                # f"Train MSE = {epoch_train_mse:.6f} | Val MSE = {epoch_val_mse:.6f}\n",
                # f"Train MSE MASKED = {epoch_train_mse_masked:.6f} | Val MSE MASKED = {epoch_val_mse_masked:.6f}\n",
                # f"Train MS-SSIM = {epoch_train_msssim:.6f} | Val MS-SSIM = {epoch_val_msssim:.6f}\n",
                # f"Train ACC = {epoch_train_acc:.6f} | Val ACC = {epoch_val_acc:.6f}\n",
                # f"Train PSNR = {epoch_train_psnr:.6f} | Val PSNR = {epoch_val_psnr:.6f}\n",
            )
            if physics_informed:
                print(
                    f"Log Train Physics Loss = {np.log10(epoch_train_physics_loss + 1e-12):.6f} | Log Val Physics Loss = {np.log10(epoch_val_physics_loss + 1e-12):.6f}"
                )
            print(
                "================================================================================================",
            )

        # Save best model
        if epoch_val_loss < best_val_loss:
            del best_state_dict
            best_state_dict = deepcopy(model.state_dict())
            best_val_loss = epoch_val_loss
            best_epoch = epoch
    if verbose:
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    return (
        train_losses,
        val_losses,
        train_physics_loss,
        val_physics_loss,
        # train_mse,
        # val_mse,
        # train_mse_masked,
        # val_mse_masked,
        # train_msssim,
        # val_msssim,
        # train_acc,
        # val_acc,
        # train_psnr,
        # val_psnr,
        best_epoch,
        best_state_dict,
    )
