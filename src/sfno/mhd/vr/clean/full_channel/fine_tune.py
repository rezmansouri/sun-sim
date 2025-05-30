import os
import sys
import numpy as np
import torch
import json
from trainer import train
from neuralop.models import SFNO
from neuralop.losses import LpLoss
from utils import SphericalNODataset, get_cr_dirs, MaskedLpLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    (
        data_path,
        batch_size,
        n_epochs,
        hidden_channels,
        n_layers,
        scale_up,
        slice_start,
        slice_end,
        state_path,
        loss_d,
        loss_p,
    ) = sys.argv[1:]
    (
        batch_size,
        n_epochs,
        hidden_channels,
        n_layers,
        scale_up,
        slice_start,
        slice_end,
        loss_d,
        loss_p,
    ) = (
        int(batch_size),
        int(n_epochs),
        int(hidden_channels),
        int(n_layers),
        int(scale_up),
        int(slice_start),
        int(slice_end),
        int(loss_d),
        int(loss_p),
    )

    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:split_ix], cr_dirs[split_ix:]

    loss_fn = MaskedLpLoss(i=slice_start, j=slice_end, d=loss_d, p=loss_p)

    train_dataset = SphericalNODataset(data_path, cr_train, scale_up=scale_up)
    val_dataset = SphericalNODataset(
        data_path,
        cr_val,
        scale_up=scale_up,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
    )

    out_path = f"n_layers-{n_layers}_hidden_channels-{hidden_channels}_finetuned_{slice_start}-{slice_end}"
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
        "n_layers": n_layers,
        "scale_up": scale_up,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "loss_d": loss_d,
        "loss_p": loss_p,
    }
    with open(os.path.join(out_path, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    model = SFNO(
        n_modes=(110 * scale_up, 128 * scale_up),
        in_channels=1,
        out_channels=139,
        hidden_channels=hidden_channels,
        factorization="dense",
        projection_channel_ratio=2,
        n_layers=n_layers,
    )
    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state)

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
