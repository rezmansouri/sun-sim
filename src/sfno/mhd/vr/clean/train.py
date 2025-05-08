import os
import sys
import numpy as np
import torch
import json
from neuralop import LpLoss
from neuralop.models import SFNO
from trainer import train
from utils import SphericalNODataset, get_cr_dirs
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    (
        data_path,
        batch_size,
        n_epochs,
        hidden_channels,
        n_modes,
        projection_channel_ratio,
        factorization,
    ) = sys.argv[1:]
    batch_size, n_epochs, hidden_channels, n_modes, projection_channel_ratio = (
        int(batch_size),
        int(n_epochs),
        int(hidden_channels),
        int(n_modes),
        int(projection_channel_ratio),
    )

    cr_dirs = get_cr_dirs(data_path)
    cr_train, cr_val = train_test_split(cr_dirs, test_size=0.2, random_state=42)

    loss_fn = LpLoss(d=2, p=2, reduction="sum")

    train_dataset = SphericalNODataset(data_path, cr_train)
    val_dataset = SphericalNODataset(
        data_path, cr_val, v_min=train_dataset.v_min, v_max=train_dataset.v_max
    )

    out_path = f"hidden_channels-{hidden_channels}_n_modes-{n_modes}_projection-{projection_channel_ratio}_factorization-{factorization}"
    os.makedirs(
        out_path,
        exist_ok=True,
    )

    cfg = {
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 8e-4,
        "train_files": cr_train,
        "val_files": cr_val,
        "v_min": float(train_dataset.v_min),
        "v_max": float(train_dataset.v_max),
        "hidden_channels": hidden_channels,
        "n_modes": n_modes,
        "projection_channel_ratio": projection_channel_ratio,
        "factorization": factorization,
    }
    with open(os.path.join(out_path, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    model = SFNO(
        n_modes=(n_modes, n_modes),
        in_channels=1,
        out_channels=139,
        hidden_channels=hidden_channels,
        projection_channel_ratio=projection_channel_ratio,
        factorization=factorization,
    )

    (
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
    )

    torch.save(best_state_dict, os.path.join(out_path, "model.pt"))
    with open(
        os.path.join(out_path, f"best_epoch-{best_epoch}.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"best_epoch: {best_epoch}")
    np.save(os.path.join(out_path, "train_losses.npy"), train_losses)
    np.save(os.path.join(out_path, "val_losses.npy"), val_losses)
    np.save(os.path.join(out_path, "train_rmse.npy"), train_rmse)
    np.save(os.path.join(out_path, "val_rmse.npy"), val_rmse)
    np.save(os.path.join(out_path, "train_nnse.npy"), train_nnse)
    np.save(os.path.join(out_path, "val_nnse.npy"), val_nnse)
    np.save(os.path.join(out_path, "train_msssim.npy"), train_msssim)
    np.save(os.path.join(out_path, "val_msssim.npy"), val_msssim)
    np.save(os.path.join(out_path, "train_acc.npy"), train_acc)
    np.save(os.path.join(out_path, "val_acc.npy"), val_acc)
    np.save(os.path.join(out_path, "train_psnr.npy"), train_psnr)
    np.save(os.path.join(out_path, "val_psnr.npy"), val_psnr)
    print("Training completed.")


if __name__ == "__main__":
    main()
