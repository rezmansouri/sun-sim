import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import PointDataset, get_xyz

# from static_model import PointNetModel
from dynamic_model import PointNetModel
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    data_path, batch_size, n_epochs, depth, starting_dim = sys.argv[1:]
    batch_size, n_epochs, depth, starting_dim = (
        int(batch_size),
        int(n_epochs),
        int(depth),
        int(starting_dim),
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
    train_dataset = PointDataset(sim_paths[:split_ix])
    min_max_dict = train_dataset.get_min_max()
    val_dataset = PointDataset(
        sim_paths[split_ix:],
        b_min=min_max_dict["b_min"],
        b_max=min_max_dict["b_max"],
    )
    dims = [starting_dim * 2**i for i in range(depth + 1)]
    cfg = {
        "instruments": instruments,
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "train_files": sim_paths[:split_ix],
        "val_files": sim_paths[split_ix:],
        "train_min": float(min_max_dict["b_min"]),
        "train_max": float(min_max_dict["b_max"]),
        "dims": dims,
    }
    with open("cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = PointNetModel(dims=dims).to(device)
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
        t_loss = []
        model.train()
        for cube in tqdm(train_loader):
            K = 9  # cube.shape[1]
            yhats = []
            for k in range(K - 1):
                x = cube[:, k, :, :]
                yhat = model(x.to(device))
                yhats.append(yhat)
            yhats = torch.stack(yhats, dim=1)
            loss = loss_fn(yhats.squeeze(), cube[:, 1:K, 3, :].to(device))
            t_loss.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        v_loss = []
        scheduler_counter += 1
        model.eval()
        for cube in tqdm(val_loader):
            with torch.no_grad():
                K = 9  # cube.shape[1]
                yhats = []
                x = cube[:, 0, :, :]
                for k in range(K - 1):
                    yhat = model(x.to(device))
                    yhats.append(yhat)
                    xx, yy, zz = get_xyz(
                        val_dataset.ii, val_dataset.jj, val_dataset.kk[k]
                    )
                    xx, yy, zz = (
                        np.tile(xx, (cube.shape[0], 1)),
                        np.tile(yy, (cube.shape[0], 1)),
                        np.tile(zz, (cube.shape[0], 1)),
                    )
                    print(xx.shape, yy.shape, zz.shape, yhat.shape)
                    xx, yy, zz = (
                        torch.tensor(xx, dtype=torch.float32).to(device),
                        torch.tensor(yy, dtype=torch.float32).to(device),
                        torch.tensor(zz, dtype=torch.float32).to(device),
                    )
                    x = torch.stack((xx, yy, zz, yhat.squeeze()), dim=1)
                yhats = torch.stack(yhats, dim=1)
                loss = loss_fn(yhats.squeeze(), cube[:, 1:K, 3, :].to(device))
                v_loss.append(loss)
        v_loss = np.mean(v_loss)
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

    np.save(os.path.join(result_path, "train_loss.npy"), np.array(train_loss))
    np.save(os.path.join(result_path, "val_loss.npy"), np.array(val_loss))
    torch.save(best_state, os.path.join(result_path, f"{best_epoch}.pth"))


if __name__ == "__main__":
    main()
