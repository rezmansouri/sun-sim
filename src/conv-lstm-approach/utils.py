import os
import torch
import h5py as h5
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import join as path_join


def get_cube(path):
    with h5.File(path) as h5_file:
        cube = h5_file["Data"][:]  # (I, J, K)
    cube = torch.tensor(
        np.transpose(cube, (2, 0, 1)), dtype=torch.float32
    )  # Reshape to (K, I, J)
    cube = torch.unsqueeze(cube, 1)  # Add channel dim (K, C, I, J)
    cube = F.interpolate(
        cube, (cube.shape[2], cube.shape[2]), mode="bicubic", align_corners=True
    )
    return cube


def get_cubes(cubes_path):
    cube_names = os.listdir(cubes_path)
    cubes = []
    for cube_name in cube_names:
        cube = get_cube(path_join(cubes_path, cube_name))
        cubes.append(cube)
    return torch.stack(cubes)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, cubes_path):
        super().__init__()
        cubes = get_cubes(cubes_path)
        self.min_k_wise = torch.amin(cubes, dim=(0, 2, 3, 4), keepdim=True)
        self.max_k_wise = torch.amax(cubes, dim=(0, 2, 3, 4), keepdim=True)
        self.cubes = cubes - self.min_k_wise / (self.max_k_wise - self.min_k_wise)

    def __getitem__(self, index):
        cube = self.cubes[index]
        # with teacher forcing
        return cube[:-1, :, :, :], cube[1:, :, :, :]

    def __len__(self):
        return len(self.cubes)

    def get_min_max(self):
        return self.min_k_wise, self.max_k_wise


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, cubes_path, min_k_wise, max_k_wise, starting_slice=1):
        super().__init__()
        cubes = get_cubes(cubes_path)
        self.cubes = cubes - min_k_wise / (max_k_wise - min_k_wise)
        self.starting_slice = starting_slice

    def __getitem__(self, index):
        cube = self.cubes[index]
        # without teacher forcing
        return (
            cube[self.starting_slice : self.starting_slice + 1, :, :, :],
            cube[self.starting_slice + 1 :, :, :, :],
        )

    def __len__(self):
        return len(self.cubes)
