import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import GraphUNet
from utils import GraphDataset
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    data_path, hidden_dim, depth, target_slice, batch_size, n_epochs = sys.argv[1:]
    hidden_dim, depth, target_slice, batch_size, n_epochs = (
        int(hidden_dim),
        int(depth),
        int(target_slice),
        int(batch_size),
        int(n_epochs),
    )
    instruments = [
        "kpo_mas_mas_std_0101",
        # "mdi_mas_mas_std_0101",
        # "hmi_mast_mas_std_0101",
        # "hmi_mast_mas_std_0201",
        # "hmi_masp_mas_std_0201",
        # "mdi_mas_mas_std_0201",
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
    train_dataset = GraphDataset(sim_paths[:split_ix], target_slice=1)
    min_max_dict = train_dataset.get_min_max()
    val_dataset = GraphDataset(
        sim_paths[split_ix:],
        target_slice=1,
        b_min=min_max_dict["b_min"],
        b_max=min_max_dict["b_max"],
    )
    cfg = {
        "instruments": instruments,
        "target_slice": target_slice,
        "depth": depth,
        "hidden_dim": hidden_dim,
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "train_files": sim_paths[:split_ix],
        "val_files": sim_paths[split_ix:],
        "train_min": float(min_max_dict["b_min"]),
        "train_max": float(min_max_dict["b_max"]),
    }
    with open("cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = GraphUNet(
        in_channels=1,
        hidden_channels=hidden_dim,
        out_channels=1,
        depth=depth,
        pool_ratios=0.5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []

    result_path = os.path.join(".")

    edge_index = train_dataset.edge_index.to(device)
    batch = torch.arange(batch_size).repeat_interleave(train_dataset[0][0].shape[0])

    for epoch in range(1, int(n_epochs) + 1):
        t_loss = []
        model.train()
        for x, y in tqdm(train_loader):
            x = x.view(-1, 1)
            yhat = model(x.to(device), edge_index=edge_index, batch=batch)
            yhat = yhat.view(batch_size, -1, 1)
            loss = loss_fn(yhat, y.to(device))
            t_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        v_loss = []
        model.eval()
        for x, y in tqdm(val_loader):
            with torch.no_grad():
                x = x.view(-1, 1)
                yhat = model(x.to(device), edge_index=edge_index, batch=batch)
                yhat = yhat.view(batch_size, -1, 1)
                loss = loss_fn(yhat, y.to(device))
                v_loss.append(loss.item())
        v_loss = np.mean(v_loss)
        if v_loss < best_val_loss:
            best_epoch = epoch
            best_state = model.state_dict()
            best_val_loss = v_loss
        val_loss.append(v_loss)
        print("epoch:", f"{epoch}/{n_epochs}", end="\t")
        print("training loss:", t_loss, end="\t")
        print("validation loss:", v_loss)

    np.save(os.path.join(result_path, "train_loss.npy"), np.array(train_loss))
    np.save(os.path.join(result_path, "val_loss.npy"), np.array(val_loss))
    torch.save(best_state, os.path.join(result_path, f"{best_epoch}.pth"))


if __name__ == "__main__":
    main()
