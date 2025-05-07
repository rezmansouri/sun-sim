import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from neuralop import LpLoss
from neuralop import Trainer
from neuralop.models import SFNO
from neuralop.training import AdamW
from utils import SphericalNODataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    data_path, batch_size, n_epochs, hidden_channels, n_modes = sys.argv[1:]
    batch_size, n_epochs, hidden_channels, n_modes = (
        int(batch_size),
        int(n_epochs),
        int(hidden_channels),
        int(n_modes),
    )
    instruments = [
        "kpo_mas_mas_std_0101",
        "mdi_mas_mas_std_0101",
        "hmi_mast_mas_std_0101",
        "hmi_mast_mas_std_0201",
        "hmi_masp_mas_std_0201",
        "mdi_mas_mas_std_0201",
    ]
    cr_dirs = sorted(
        [
            d
            for d in os.listdir(data_path)
            if d.startswith("cr") and os.path.isdir(os.path.join(data_path, d))
        ]
    )

    cr_train, cr_val = train_test_split(cr_dirs, test_size=0.25, random_state=42)

    train_sim_paths = []
    for cr in cr_train:
        cr_path = os.path.join(data_path, cr)
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                train_sim_paths.append(instrument_path)

    val_sim_paths = []
    for cr in cr_val:
        cr_path = os.path.join(data_path, cr)
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                val_sim_paths.append(instrument_path)

    train_dataset = SphericalNODataset(train_sim_paths)
    min_max_dict = train_dataset.get_min_max()
    val_dataset = SphericalNODataset(
        val_sim_paths,
        v_min=min_max_dict["v_min"],
        v_max=min_max_dict["v_max"],
    )
    cfg = {
        "instruments": instruments,
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 8e-4,
        "train_files": train_sim_paths,
        "val_files": val_sim_paths,
        "v_min": float(min_max_dict["v_min"]),
        "v_max": float(min_max_dict["v_max"]),
        "hidden_channels": hidden_channels,
        "n_modes": n_modes,
    }
    with open("cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = SFNO(
        n_modes=(n_modes, n_modes),
        in_channels=1,
        out_channels=139,
        hidden_channels=hidden_channels,
        projection_channel_ratio=2,
        factorization="dense",
    )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=8e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    l2loss = LpLoss(d=2, p=2, reduction="sum")

    train_loss = l2loss
    eval_losses = {"l2": l2loss}  #'h1': h1loss,

    trainer = Trainer(
        model=model,
        n_epochs=n_epochs,
        device=device,
        wandb_log=False,
        eval_interval=3,
        use_distributed=False,
        verbose=True,
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders={(111, 128): val_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_best="(111, 128)_l2",
    )


if __name__ == "__main__":
    main()
