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
    subdir_paths = sorted(os.listdir(data_path))
    cr_paths = [os.path.join(data_path, p) for p in subdir_paths if p.startswith("cr")]
    sim_paths = []
    for cr_path in cr_paths:
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                sim_paths.append(instrument_path)
    split_ix = int(len(sim_paths) * 0.75)
    train_dataset = SphericalNODataset(sim_paths[:split_ix])
    min_max_dict = train_dataset.get_min_max()
    val_dataset = SphericalNODataset(
        sim_paths[split_ix:],
        v_min=min_max_dict["v_min"],
        v_max=min_max_dict["v_max"],
    )
    cfg = {
        "instruments": instruments,
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 8e-4,
        "train_files": sim_paths[:split_ix],
        "val_files": sim_paths[split_ix:],
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
        out_channels=140,
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
