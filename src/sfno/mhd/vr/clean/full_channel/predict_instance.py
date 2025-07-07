import time
import psutil
import gc
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from neuralop.models import SFNO
from utils import get_cr_dirs, SphericalNODataset
from torch.utils.data import DataLoader

device = "cpu"


def main():
    data_path, result_path = sys.argv[1:]
    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:split_ix], cr_dirs[split_ix:]

    train_dataset = SphericalNODataset(data_path, cr_train, scale_up=1)
    val_dataset = SphericalNODataset(
        data_path,
        cr_val,
        v_min=train_dataset.v_min,
        v_max=train_dataset.v_max,
        scale_up=1,
    )

    cfg_path = os.path.join(result_path, "cfg.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2
    model = SFNO(
        n_modes=(110, 128),
        in_channels=1,
        out_channels=139,
        hidden_channels=cfg["hidden_channels"],
        projection_channel_ratio=2,
        factorization="dense",
        n_layers=cfg["n_layers"],
    ).to(device)
    state = torch.load(
        os.path.join(result_path, "model.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(state)

    model.eval()

    with torch.no_grad():
        for instance in tqdm(val_loader, leave=False):

            x, cube = instance["x"].to(device), instance["y"].to(device)
            start = time.perf_counter()
            yhats = model(x)  # shape (B, T-1, H, W)
            end = time.perf_counter()
            mem_after = process.memory_info().rss / 1024**2  # in MB
            break
        
    print(f"Time elapsed: {end - start:.4f} seconds")
    print(f"Memory used: {mem_after - mem_before:.2f} MB")


if __name__ == "__main__":
    main()
