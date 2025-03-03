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
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    val_dataset = GraphDataset(
        cfg["val_files"],
        target_slice=cfg["target_slice"],
        b_min=cfg["train_min"],
        b_max=cfg["train_max"],
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model = GraphUNet(
        in_channels=1,
        hidden_channels=cfg["hidden_dim"],
        out_channels=1,
        depth=cfg["depth"],
        pool_ratios=0.5,
    ).to(device)
    
    state = torch.load('')
    
    model.load_state_dict(state)

    edge_index = val_dataset.edge_index.to(device)
    edge_index_batch = edge_index.clone()

    batch_size = cfg["batch_size"]

    n_nodes = val_dataset[0][0].shape[0]

    batch = torch.arange(1).repeat_interleave(n_nodes).to(device)

    model.eval()
    xs = []
    yhats = []
    ys = []
    for x, y in tqdm(val_loader):
        with torch.no_grad():
            x = x.view(-1, 1)
            yhat = model(x.to(device), edge_index=edge_index_batch, batch=batch)
            yhat = yhat.view(batch_size, -1, 1)
            yhats.append(yhat.cpu().numpy())
            xs.append(x.numpy())
            ys.append(y.numpy())
    xs = np.stack(xs, axis=0)
    yhats = np.stack(yhats, axis=0)
    ys = np.stack(ys, axis=0)
    result_path = os.path.join(".")
    np.save(os.path.join(result_path, "xs.npy"), xs)
    np.save(os.path.join(result_path, "yhats.npy"), yhats)
    np.save(os.path.join(result_path, "ys.npy"), ys)


if __name__ == "__main__":
    main()
