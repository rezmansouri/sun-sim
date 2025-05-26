import itertools
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from neuralop.models import SFNO
import torch
from copy import deepcopy
import torch.nn as nn
from utils import SphericalNODataset
from tqdm import tqdm
from pytorch_msssim import MS_SSIM
import torch.optim as optim
from metrics import nnse_score, mssim_score, rmse_score, acc_score, psnr_score


MSSSIM_MODULE = MS_SSIM(
    data_range=1.0,
    size_average=True,
    channel=1,  # after unsqueeze(1)
    spatial_dims=3,  # 3D input
    win_size=7,
    win_sigma=1.0,
)


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
    scaler = torch.amp.GradScaler(device=autocast_device_type)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    train_losses = []
    val_losses = []
    train_rmse = []
    val_rmse = []
    train_nnse = []
    val_nnse = []
    train_msssim = []
    val_msssim = []
    train_acc = []
    val_acc = []
    train_psnr = []
    val_psnr = []

    climatology = train_dataset.climatology.to(device)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        running_nnse = 0.0
        running_msssim = 0.0
        running_acc = 0.0
        running_psnr = 0.0
        for cube in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False
        ):


            pred = []
            for i in range(cube.shape[1] - 1):
                x = cube[:, i, :, :].unsqueeze(1).to(device)
                y = cube[:, i + 1, :, :].unsqueeze(1).to(device)
                pred_slice = model(x)
                pred.append(pred_slice)
                
                
                optimizer.zero_grad()
                loss = loss_fn(pred_slice, y)
                running_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            y = cube[:, 1:, :, :].to(device)
            pred = torch.stack(pred, dim=1).squeeze(2)

            running_rmse += rmse_score(y, pred)
            running_nnse += nnse_score(y, pred, climatology)
            running_msssim += mssim_score(MSSSIM_MODULE, y, pred)
            running_acc += acc_score(y, pred, climatology)
            running_psnr += psnr_score(y, pred)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_rmse = running_rmse / len(train_loader)
        epoch_train_nnse = running_nnse / len(train_loader)
        epoch_train_msssim = running_msssim / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader)
        epoch_train_psnr = running_psnr / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_rmse.append(epoch_train_rmse)
        train_nnse.append(epoch_train_nnse)
        train_msssim.append(epoch_train_msssim)
        train_acc.append(epoch_train_acc)
        train_psnr.append(epoch_train_psnr)

        scheduler.step(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        running_rmse = 0.0
        running_nnse = 0.0
        running_msssim = 0.0
        running_acc = 0.0
        running_psnr = 0.0
        with torch.no_grad():
            for cube in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False
            ):

                pred = []
                for i in range(cube.shape[1] - 1):
                    x = cube[:, i, :, :].unsqueeze(1).to(device)
                    y = cube[:, i + 1, :, :].unsqueeze(1).to(device)
                    pred_slice = model(x)
                    pred.append(pred_slice)
                    
                    loss = loss_fn(pred_slice, y)
                    val_loss += loss.item()
                    
                y = cube[:, 1:, :, :].to(device)
                pred = torch.stack(pred, dim=1).squeeze(2)
                running_rmse += rmse_score(y, pred)
                running_nnse += nnse_score(y, pred, climatology)
                running_msssim += mssim_score(MSSSIM_MODULE, y, pred)
                running_acc += acc_score(y, pred, climatology)
                running_psnr += psnr_score(y, pred)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_rmse = running_rmse / len(val_loader)
        epoch_val_nnse = running_nnse / len(val_loader)
        epoch_val_msssim = running_msssim / len(val_loader)
        epoch_val_acc = running_acc / len(val_loader)
        epoch_val_psnr = running_psnr / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_rmse.append(epoch_val_rmse)
        val_nnse.append(epoch_val_nnse)
        val_msssim.append(epoch_val_msssim)
        val_acc.append(epoch_val_acc)
        val_psnr.append(epoch_val_psnr)

        if verbose:
            print(
                f"Epoch {epoch+1}:\n",
                f"Train Loss = {epoch_train_loss:.6f} | Val Loss = {epoch_val_loss:.6f}\n",
                f"Train RMSE = {epoch_train_rmse:.6f} | Val RMSE = {epoch_val_rmse:.6f}\n",
                f"Train NNSE = {epoch_train_nnse:.6f} | Val NNSE = {epoch_val_nnse:.6f}\n",
                f"Train MS-SSIM = {epoch_train_msssim:.6f} | Val MS-SSIM = {epoch_val_msssim:.6f}\n",
                f"Train ACC = {epoch_train_acc:.6f} | Val ACC = {epoch_val_acc:.6f}\n",
                f"Train PSNR = {epoch_train_psnr:.6f} | Val PSNR = {epoch_val_psnr:.6f}\n",
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
        train_rmse,
        val_rmse,
        train_nnse,
        val_nnse,
        train_msssim,
        val_msssim,
        train_acc,
        val_acc,
        train_psnr,
        val_psnr,
        best_epoch,
        best_state_dict,
    )
