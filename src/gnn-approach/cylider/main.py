import os
import sys
import torch
import json
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import grid
from torch_geometric.data import Data
import torch_geometric.nn as gnn
from pyhdf.SD import SD, SDC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spherical_grid_edges(height, width):
    # Create grid edges first (without periodic boundary conditions)
    e, _ = grid(height=height, width=width)

    # Periodic boundary for horizontal edges (right-left wrapping)
    horizontal_edges = []
    for i in range(height):
        left = i * width
        right = (i + 1) * width - 1
        horizontal_edges.append((right, left))  # Connect right edge to left edge

    # Convert horizontal_edges to a tensor
    horizontal_edges = torch.tensor(horizontal_edges).t().long()

    # Concatenate the original edge index with the new periodic horizontal edges
    ee = torch.cat([e, horizontal_edges], dim=1)

    return ee


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


class CustomDataset(Dataset):
    def __init__(self, file_paths, num_time_steps, min_=None, max_=None):
        """
        Args:
            file_paths (list): List of paths to Br files.
        """
        self.edge_index = spherical_grid_edges(101, 101)
        self.num_time_steps = num_time_steps
        cubes = []
        for file_path in file_paths:
            arr, i, j, k = read_hdf(
                file_path, ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"]
            )
            arr = np.transpose(arr, (2, 1, 0))
            k = np.expand_dims(np.expand_dims(k, axis=-1), axis=-1)
            arr = arr * (k**2)
            cubes.append(arr)
        cubes = np.stack(cubes, axis=0)
        if max_ is not None and min_ is not None:
            self.min_, self.max_ = min_, max_
        else:
            self.min_, self.max_ = np.min(cubes), np.max(cubes)
        cubes = (cubes - self.min_) / (self.max_ - self.min_)
        self.cubes = cubes

    def __len__(self):
        return self.cubes.shape[0]

    def get_graph_data(self, data):
        if data.dtype == torch.float32:
            x = data.flatten().unsqueeze(-1)
        else:
            x = torch.tensor(data.flatten(), dtype=torch.float32).unsqueeze(-1)
        return Data(x=x, edge_index=self.edge_index)

    def __getitem__(self, idx):
        slices = []
        cube = self.cubes[idx]
        cube = torch.tensor(cube, dtype=torch.float)
        for k in range(self.num_time_steps):

            data = self.get_graph_data(cube[k])
            slices.append(data)
        return slices, cube[1:]


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = gnn.GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.conv4 = gnn.GCNConv(hidden_dim * 4, hidden_dim * 2)
        self.conv5 = gnn.GCNConv(hidden_dim * 2, hidden_dim)
        self.conv6 = gnn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.sigmoid(self.conv6(x, edge_index))
        return x


def main():
    # data_path, hidden_dim, n_epochs = (
    #     "/Users/reza/Career/DMLab/SURROGATE/Data/pfss",
    #     2,
    #     5,
    # )
    data_path, hidden_dim, n_epochs = sys.argv[1:]
    file_paths = []
    for date in os.listdir(data_path):
        file_path = os.path.join(data_path, date, "br.hdf")
        if not os.path.exists(file_path):
            continue
        file_paths.append(file_path)
    split_ix = int(len(file_paths) * 0.75)
    train_dataset = CustomDataset(file_paths[:split_ix], num_time_steps=100)
    val_dataset = CustomDataset(
        file_paths[split_ix:],
        num_time_steps=100,
        min_=train_dataset.min_,
        max_=train_dataset.max_,
    )
    cfg = {
        "hidden_dim": hidden_dim,
        "num_epochs": int(n_epochs),
        "batch_size": 1,
        "learning_rate": 1e-3,
        "train_files": file_paths[:split_ix],
        "val_files": file_paths[split_ix:],
        "train_min": float(train_dataset.min_),
        "train_max": float(train_dataset.max_),
    }
    with open("cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    model = GNNModel(1, int(hidden_dim), 1).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    best_val_loss = torch.inf
    best_state = None
    best_epoch = -1
    train_loss, val_loss = [], []

    result_path = os.path.join(".")

    for epoch in range(1, int(n_epochs) + 1):
        t_loss = []
        model.train()
        for slices, sim_cube in tqdm(train_dataset):
            yhat = []
            for k in range(len(slices) - 1):
                x = slices[k]
                yhat_k = model(x.to(device))
                yhat.append(yhat_k.reshape(101, 101))
            yhat = torch.stack(yhat, dim=0)
            loss = loss_fn(yhat, sim_cube.to(device))
            t_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t_loss = np.mean(t_loss)
        train_loss.append(t_loss)
        v_loss = []
        model.eval()
        for slices, sim_cube in tqdm(val_dataset):
            with torch.no_grad():
                yhat = []
                x = slices[0]
                for _ in range(len(slices) - 1):
                    yhat_k = model(x.to(device))
                    yhat.append(yhat_k.reshape(101, 101))
                    x = val_dataset.get_graph_data(yhat_k)
                yhat = torch.stack(yhat, dim=0)
                loss = loss_fn(yhat, sim_cube.to(device))
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
