import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from neuralop import LpLoss
from neuralop import Trainer
from neuralop.models import SFNO
from neuralop.training import AdamW
from utils import PhysicalLaw
from utils import SphericalNODataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg_path, state_path = sys.argv[1:]
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    sim_paths = cfg["val_files"]
    val_dataset = SphericalNODataset(
        sim_paths,
        v_min=cfg["v_min"],
        v_max=cfg["v_max"],
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = SFNO(
        n_modes=(cfg["n_modes"], cfg["n_modes"]),
        in_channels=2,
        out_channels=278,
        hidden_channels=cfg["hidden_channels"],
        projection_channel_ratio=2,
        factorization="dense",
    )
    model = model.to(device)
    state = torch.load(state_path, weights_only=False, map_location=device)
    model.load_state_dict(state)
    l2loss = LpLoss(d=2, p=2, reduction="sum")

    law = PhysicalLaw(
        val_dataset.r,
        val_dataset.v_min,
        val_dataset.v_max,
        val_dataset.rho_min,
        val_dataset.rho_max,
    )

    v_losses, rho_losses, law_losses = [], [], []

    model.eval()
    for batch in tqdm(val_loader):
        sample = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            out = model(**sample)
            law_out = law.forward(out, sample["p"])
            law_loss = F.mse_loss(law_out, torch.zeros_like(out))
        law_losses.append(law_loss.item())
        v_pred = out[:, 0::2, :, :]
        rho_pred = out[:, 1::2, :, :]
        v = sample["y"][:, 0::2, :, :]
        rho = sample["y"][:, 1::2, :, :]
        v_loss = l2loss(v_pred, v)
        rho_loss = l2loss(rho_pred, rho)
        v_losses.append(v_loss.item())
        rho_losses.append(rho_loss.item())

    v_losses, rho_losses, law_losses = np.mean(v_losses), np.mean(rho_losses), np.mean(law_losses)

    with open("loss.txt", "w") as f:
        f.write(f"v: {v_losses}, rho: {rho_losses}, physical: {law_losses}")

    print("v", v_losses, "rho", rho_losses, 'physical', law_losses)


if __name__ == "__main__":
    main()
