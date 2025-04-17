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
    cfg_path, state_path = sys.argv[1:]
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    instruments = [
        "kpo_mas_mas_std_0101",
        "mdi_mas_mas_std_0101",
        "hmi_mast_mas_std_0101",
        "hmi_mast_mas_std_0201",
        "hmi_masp_mas_std_0201",
        "mdi_mas_mas_std_0201",
    ]
    sim_paths = cfg["val_files"]
    val_dataset = SphericalNODataset(
        sim_paths,
        rho_min=cfg["rho_min"],
        rho_max=cfg["rho_max"],
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = SFNO(
        n_modes=(cfg["n_modes"], cfg["n_modes"]),
        in_channels=1,
        out_channels=140,
        hidden_channels="hidden_channels",
        projection_channel_ratio=2,
        factorization="dense",
    )
    model = model.to(device)
    state = torch.load(state_path, weights_only=False, map_location=device)
    model.load_state_dict(state)
    l2loss = LpLoss(d=2, p=2, reduction="sum")

    losses = []

    model.eval()
    for batch in tqdm(val_loader):
        sample = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            out = model(**sample)
        loss = l2loss(out, sample["y"])
        losses.append(loss.item())

    losses = np.mean(losses)

    with open("loss.txt", "w") as f:
        f.write(str(losses))

    print(losses)


if __name__ == "__main__":
    main()
