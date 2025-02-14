import os
import sys
import torch
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import EncoderDecoder
import matplotlib.pyplot as plt
from torch.optim.adam import Adam
from utils import Dataset
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

instruments = [
    "kpo_mas_mas_std_0101",
    # "mdi_mas_mas_std_0101",
    # "hmi_mast_mas_std_0101",
    # "hmi_mast_mas_std_0201",
    # "hmi_masp_mas_std_0201",
    # "mdi_mas_mas_std_0201",
]


def main():
    # data_path = "/Users/reza/Career/DMLab/SURROGATE/Data/psi_web_sample"
    # batch_size = 1
    # base_channels = 2
    # latent_dim = 256
    # n_epochs = 10
    data_path, batch_size, base_channels, latent_dim, n_epochs = sys.argv[1:]
    batch_size, base_channels, latent_dim, n_epochs = [
        int(i) for i in [batch_size, base_channels, latent_dim, n_epochs]
    ]
    subdir_paths = sorted(os.listdir(data_path))
    cr_paths = []
    for subdir_path in subdir_paths:
        p = os.path.join(data_path, subdir_path)
        if not p.startswith("cr"):
            continue
        all_present = True
        for instrument in instruments:
            if not os.path.exists(os.path.join(p, instrument)):
                all_present = False
                break
        if all_present:
            cr_paths.append(p)
    split_ix = int(len(cr_paths) * 0.75)
    train_dataset = Dataset(cr_paths=cr_paths[:split_ix], instruments=instruments)
    min_max_dict = train_dataset.get_min_max()
    cfg = {
        "data_path": data_path,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "train_crs": cr_paths[:split_ix],
        "test_crs": cr_paths[split_ix:],
        "instruments": train_dataset.instruments,
        **min_max_dict,
        "base_channels": base_channels,
        "latent_dim": latent_dim,
    }
    with open("cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    test_dataset = Dataset(
        instruments=instruments,
        cr_paths=cr_paths[split_ix:],
        v_min=min_max_dict["v_min"],
        v_max=min_max_dict["v_max"],
        # rho_min=min_max_dict["rho_min"],
        # rho_max=min_max_dict["rho_max"],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print("train dataset", "cr" + cr_paths[0], "to", cr_paths[split_ix])
    # print("test dataset", "cr" + cr_paths[split_ix], "to", cr_paths[-1])
    model = EncoderDecoder(
        in_channels=1, base_channels=base_channels, latent_dim=latent_dim
    ).to(device)
    optimizer = Adam(model.parameters())
    loss_fn = nn.MSELoss()

    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []

    seq_len = train_dataset[0].shape[1] - 1

    result_path = os.path.join(".")

    for epoch in range(1, n_epochs + 1):
        t_loss = []
        model.train()
        for cubes in tqdm(train_loader):
            y = cubes[:, :, 1:, :, :]
            yhat = model(cubes.to(device))
            loss = loss_fn(yhat, y.to(device))
            t_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        v_loss = []
        model.eval()
        for cubes in tqdm(test_loader):
            with torch.no_grad():
                x = cubes[:, :, 0, :, :]
                y = x[:, :, 1:, :, :]
                yhat = model.predict(cubes.to(device), n_slices=seq_len)
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
