import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm, trange
from utils import SphericalDataset
from architectures import UNetSpherical
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def update_loss_weights(w):
    for i in range(1, len(w)):
        len_w = len(w)
        w[len_w - i] += w[len_w - i - 1] * 0.4
        w[len_w - i - 1] *= 0.8
        w = np.array(w) / sum(w)
    return w


def main():
    data_path, batch_size, n_epochs, n_neighbors, seq_len = sys.argv[1:]
    batch_size, n_epochs, n_neighbors, seq_len = (
        int(batch_size),
        int(n_epochs),
        int(n_neighbors),
        int(seq_len),
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
    train_dataset = SphericalDataset(
        sim_paths[:split_ix],
    )
    min_max_dict = train_dataset.get_min_max()
    val_dataset = SphericalDataset(
        sim_paths[split_ix:],
        b_min=min_max_dict["b_min"],
        b_max=min_max_dict["b_max"],
    )
    weights = np.arange(1, seq_len + 1, dtype=np.float32)
    weights /= sum(weights)
    cfg = {
        "instruments": instruments,
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "train_files": sim_paths[:split_ix],
        "val_files": sim_paths[split_ix:],
        "train_min": float(min_max_dict["b_min"]),
        "train_max": float(min_max_dict["b_max"]),
        "n_neighbors": n_neighbors,
        "seq_len": seq_len,
        "loss_weights": weights.tolist(),
    }
    with open("cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = UNetSpherical(32, in_channels=2, out_channels=1, knn=n_neighbors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler_counter = 0

    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []

    result_path = os.path.join(".")

    for epoch in range(1, int(n_epochs) + 1):
        t_loss = 0
        model.train()
        for cube in tqdm(train_loader):
            x0 = cube[:, 0, :, :]
            for i in trange(seq_len, leave=False):
                xi = cube[:, i, :, :]
                y = cube[:, i + 1, :, :]
                x = torch.cat([x0, xi], dim=-1)
                yhat = model(x.to(device))
                loss = loss_fn(yhat, y.to(device))
                t_loss += loss * weights[i]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = t_loss.item() / (len(train_loader) * seq_len * batch_size)
        train_loss.append(t_loss)
        v_loss = 0
        scheduler_counter += 1
        model.eval()
        for cube in tqdm(val_loader):
            x0 = cube[:, 0, :, :].to(device)
            xi = cube[:, 0, :, :].to(device)
            for i in trange(seq_len, leave=False):
                x = torch.cat([x0, xi], dim=-1)
                y = cube[:, i + 1, :, :]
                with torch.no_grad():
                    yhat = model(x.to(device))
                    xi = yhat
                loss = loss_fn(yhat, y.to(device))
                v_loss += loss.item() * weights[i]
        v_loss = v_loss / (len(val_loader) * seq_len * batch_size)
        if v_loss < best_val_loss:
            scheduler_counter = 0
            best_epoch = epoch
            best_state = model.state_dict()
            best_val_loss = v_loss
        val_loss.append(v_loss)
        print("epoch:", f"{epoch}/{n_epochs}", end="\t")
        print("training loss:", t_loss, end="\t")
        print("validation loss:", v_loss)
        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"\nlowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0
            weights = update_loss_weights(weights)
            print(f"\nupdated loss weights: {weights}")

    np.save(os.path.join(result_path, "train_loss.npy"), np.array(train_loss))
    np.save(os.path.join(result_path, "val_loss.npy"), np.array(val_loss))
    torch.save(best_state, os.path.join(result_path, f"{best_epoch}.pth"))


if __name__ == "__main__":
    main()
