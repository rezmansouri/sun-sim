import os
import torch
import cv2 as cv
import numpy as np
from pyhdf.SD import SD, SDC
from os.path import join as path_join
from torch_geometric.data import Dataset

FILE_NAMES = ["br_r0.hdf", "br002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


# def pad_3d_array_to_square(three_d_array):
#     # assuming h > w
#     h, w = three_d_array.shape[:2]
#     assert h >= w, f"h is not geq than w, h: {h}, w:{w}"
#     pad_left = (h - w) // 2
#     pad_right = (h - w) - pad_left
#     if pad_left > 0 or pad_right > 0:
#         three_d_array = np.pad(
#             three_d_array, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge"
#         )
#     return three_d_array


def pad_2d_array(two_d_array, target_shape):
    # assuming h > w
    h, w = two_d_array.shape
    target_h, target_w = target_shape
    pad_left = (target_w - w) // 2
    pad_right = (target_w - w) - pad_left
    if pad_left > 0 or pad_right > 0:
        two_d_array = np.pad(two_d_array, ((0, 0), (pad_left, pad_right)), mode="edge")
    return two_d_array


# def resize(two_d_array, size):
#     if two_d_array.shape == (size, size):
#         return two_d_array
#     return cv.resize(two_d_array, (size, size), interpolation=cv.INTER_LANCZOS4)


def get_sim(sim_path):
    b_bc_path, b_path = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    b_bc, _, _ = read_hdf(b_bc_path, ["Data-Set-2", "fakeDim0", "fakeDim1"])
    b, _, _, b_k = read_hdf(b_path, ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"])

    b_bc = pad_2d_array(b_bc, (b.shape[0], b.shape[1]))

    b_k = np.expand_dims(np.expand_dims(b_k, axis=0), axis=0)
    b *= b_k**2

    # b_bc = pad_2d_array_to_square(b_bc)
    # b_bc = resize(b_bc, b.shape[0])
    # b = pad_3d_array_to_square(b)

    b_bc = np.transpose(b_bc, (1, 0))
    b = np.transpose(b, (2, 1, 0))

    b_bc = np.expand_dims(b_bc, axis=0)
    cube = np.concatenate([b_bc, b], axis=0)

    cube = np.expand_dims(cube, axis=1)

    return cube


def get_sims(sim_paths):
    sims = []
    for sim_path in sim_paths:
        sims.append(get_sim(sim_paths))
    sims = np.stack(sims, axis=0)
    return sims


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = array - min_ / (max_ - min_)
    return array, min_, max_


def spherical_grid_edges(rows, cols):
    edges = set()

    # Compute all indexes beforehand
    indexes = np.arange(rows * cols, dtype=np.long).reshape(rows, cols)

    for i in range(rows):
        for j in range(cols):
            current = indexes[i, j]

            # Right neighbor (wrap around horizontally)
            right = indexes[i, (j + 1) % cols]
            edges.add((min(current, right), max(current, right)))
            edges.add((max(current, right), min(current, right)))

            # Down neighbor (no vertical wrap around)
            if i < rows - 1:
                down = indexes[i + 1, j]
                edges.add((min(current, down), max(current, down)))
                edges.add((max(current, down), min(current, down)))

    # Create a clique for the first row
    first_row_indices = indexes[0, :]
    for i in range(len(first_row_indices)):
        for j in range(i + 1, len(first_row_indices)):
            edges.add((first_row_indices[i], first_row_indices[j]))
            edges.add((first_row_indices[j], first_row_indices[i]))

    # Create a clique for the last row
    last_row_indices = indexes[-1, :]
    for i in range(len(last_row_indices)):
        for j in range(i + 1, len(last_row_indices)):
            edges.add((last_row_indices[i], last_row_indices[j]))
            edges.add((last_row_indices[j], last_row_indices[i]))

    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()


class GraphDataset(Dataset):
    def __init__(
        self,
        cr_paths,
        target_slice=70,
        instruments=[
            "kpo_mas_mas_std_0101",
            # "mdi_mas_mas_std_0101",
            # "hmi_mast_mas_std_0101",
            # "hmi_mast_mas_std_0201",
            # "hmi_masp_mas_std_0201",
            # "mdi_mas_mas_std_0201",
        ],
        b_min=None,
        b_max=None,
    ):
        super().__init__()
        self.instruments = instruments
        self.target_slice = target_slice
        sims, self.b_min, self.b_max = min_max_normalize(sims, b_min, b_max)
        _, self.k, _, self.i, self.j = sims.shape
        self.edge_index = spherical_grid_edges(self.i, self.j)
        self.sims = sims

    def __getitem__(self, index):
        cube = self.sims[index]
        br_r0 = np.reshape(cube[0], (self.i * self.j, 1))
        br002 = np.reshape(cube[self.target_slice], (self.i * self.j, 1))
        # br_r0_graph = Data(
        #     x=torch.tensor(br_r0, dtype=torch.float),
        #     edge_index=self.edge_index,
        # )
        # br002_graph = Data(
        #     x=torch.tensor(br002, dtype=torch.float),
        #     edge_index=self.edge_index,
        # )
        return torch.tensor(br_r0), torch.tensor(br002)  # br_r0_graph, br002_graph

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"b_min": float(self.b_min), "b_max": float(self.b_max)}
