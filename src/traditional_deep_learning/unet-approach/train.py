import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import UNet
import matplotlib.pyplot as plt
from torch.optim.adam import Adam
from utils import Dataset
from pprint import pprint
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # data_path = "/Users/reza/Career/DMLab/SURROGATE/Data/psi_web_sample"
    # batch_size = 2
    # num_layers = 1
    # hidden_dim = 4
    # n_epochs = 10
    data_path, batch_size, n_epochs = sys.argv[1:]
    batch_size, n_epochs = [int(i) for i in [batch_size, n_epochs]]
    subdir_paths = sorted(os.listdir(data_path))
    cr_paths = [os.path.join(data_path, p) for p in subdir_paths if p.startswith("cr")][:100]
    split_ix = int(len(cr_paths) * 0.75)
    train_dataset = Dataset(cr_paths=cr_paths[:split_ix])
    min_max_dict = train_dataset.get_min_max()
    channels = [32, 64, 128]
    cfg = {
        'data_path': data_path,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'train_crs': [cr_paths[0], cr_paths[split_ix]],
        'test_crs': [cr_paths[split_ix], cr_paths[-1]],
        'instruments': train_dataset.instruments,
        'channels': channels,
        **min_max_dict
    }
    with open('cfg.json', 'w') as f:
        json.dump(cfg, f)
    test_dataset = Dataset(
        cr_paths=cr_paths[split_ix:],
        v_min=min_max_dict["v_min"],
        v_max=min_max_dict["v_max"],
        rho_min=min_max_dict["rho_min"],
        rho_max=min_max_dict["rho_max"],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(
        "train dataset",
        "cr" + cr_paths[0],
        "to",
        cr_paths[split_ix],
        "size:",
        len(train_dataset),
    )
    print(
        "test dataset",
        "cr" + cr_paths[split_ix],
        "to",
        cr_paths[-1],
        "size",
        len(test_dataset),
    )
    model = UNet(n_channels=3, output_ch=2, channels=channels).to(device)
    optimizer = Adam(model.parameters())
    loss_fn = nn.MSELoss()

    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []

    seq_len = train_dataset[0][0].shape[0]

    result_path = "."

    for epoch in range(1, n_epochs + 1):
        print("epoch:", f"{epoch}/{n_epochs}", end="\t")
        t_loss = []
        model.train()
        for x, y in train_loader:
            yhat = []
            for k in range(0, seq_len):
                x_k = x[:, k, :, :, :]
                yhat_k = model(x_k.to(device))
                yhat.append(yhat_k)
            yhat = torch.stack(yhat, dim=1)
            loss = loss_fn(yhat.to(device), y.to(device))
            t_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        print("training loss:", t_loss, end="\t")
        v_loss = []
        model.eval()
        for x, y in test_loader:
            yhat = []
            for k in range(0, seq_len):
                x_k = x[:, k, :, :, :]
                yhat_k = model(x_k.to(device))
                yhat.append(yhat_k)
            yhat = torch.stack(yhat, dim=1)
            loss = loss_fn(yhat.to(device), y.to(device))
            v_loss.append(loss.item())
        v_loss = np.mean(v_loss)
        if v_loss < best_val_loss:
            best_epoch = epoch
            best_state = model.state_dict()
            best_val_loss = v_loss
        val_loss.append(v_loss)
        print("validation loss:", v_loss)

    np.save(os.path.join(result_path, "train_loss.npy"), np.array(train_loss))
    np.save(os.path.join(result_path, "val_loss.npy"), np.array(val_loss))
    torch.save(best_state, os.path.join(result_path, f"{best_epoch}.pth"))


if __name__ == "__main__":
    main()
