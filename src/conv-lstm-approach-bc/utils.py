import os
import torch
import cv2 as cv
import numpy as np
from pyhdf.SD import SD, SDC
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import join as path_join


def get_cube(bc_path, cube_path):
    f = SD(bc_path, SDC.READ)
    bc = f.select("Data-Set-2").get()
    h, w = bc.shape
    pad_left = (h - w) // 2
    pad_right = (h - w) - pad_left
    bc = np.pad(bc, ((0, 0), (pad_left, pad_right)), mode="edge")
    f = SD(cube_path, SDC.READ)
    cube = f.select("Data-Set-2").get()
    h, w = cube.shape[:2]
    if bc.shape != (h, h):
        # print(bc.shape)
        bc = cv.resize(bc, (h, h), interpolation=cv.INTER_LANCZOS4)
    bc = np.expand_dims(bc, axis=0)
    pad_left = (h - w) // 2
    pad_right = (h - w) - pad_left
    cube = np.pad(cube, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge")
    cube = np.transpose(cube, (2, 0, 1))  # Reshape to (K, I, J)
    # print(bc.shape, cube.shape)
    cube = np.concatenate([bc, cube], axis=0)
    cube = np.expand_dims(cube, axis=1)
    return torch.tensor(cube, dtype=torch.float32)


def get_cubes(cr_paths, instruments, variables):
    cubes = []
    for cr_path in cr_paths:
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if not os.path.exists(instrument_path):
                # print(instrument_path)
                continue
            for variable in variables:
                bc_path = os.path.join(instrument_path, f"{variable}r_r0.hdf")
                cube_path = os.path.join(instrument_path, f"{variable}r002.hdf")
                cubes.append(get_cube(bc_path, cube_path))
    return torch.stack(cubes)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cr_paths,
        instruments=[
            "kpo_mas_mas_std_0101",
            "mdi_mas_mas_std_0101",
            "hmi_mast_mas_std_0101",
            "hmi_mast_mas_std_0201",
            "hmi_masp_mas_std_0201",
            "mdi_mas_mas_std_0201",
        ],
        variables=["v"],
    ):
        super().__init__()
        cubes = get_cubes(cr_paths, instruments, variables)
        self.min = torch.min(cubes)
        self.max = torch.max(cubes)
        self.cubes = cubes - self.min / (self.max - self.min)

    def __getitem__(self, index):
        cube = self.cubes[index]
        # with teacher forcing
        return cube[:-1, :, :, :], cube[1:, :, :, :]

    def __len__(self):
        return len(self.cubes)

    def get_min_max(self):
        return self.min, self.max


class TestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cr_paths,
        train_min,
        train_max,
        starting_slice=0,
        instruments=[
            "kpo_mas_mas_std_0101",
            "mdi_mas_mas_std_0101",
            "hmi_mast_mas_std_0101",
            "hmi_mast_mas_std_0201",
            "hmi_masp_mas_std_0201",
            "mdi_mas_mas_std_0201",
        ],
        variables=["v"],
    ):
        super().__init__()
        cubes = get_cubes(cr_paths, instruments, variables)
        self.cubes = cubes - train_min / (train_max - train_min)
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
