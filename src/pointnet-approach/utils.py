# TODO normalize x, y, z?

import torch
import numpy as np
from pyhdf.SD import SD, SDC
from os.path import join as path_join
from torch_geometric.data import Dataset

FILE_NAMES = ["br002.hdf"]


def get_xyz(ii, jj, r):
    theta, phi = np.meshgrid(ii, jj, indexing="ij")
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    return x_flat, y_flat, z_flat


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_ijk(sim_path):
    b_path = path_join(sim_path, FILE_NAMES[0])
    i, j, k = read_hdf(b_path, ["fakeDim0", "fakeDim1", "fakeDim2"])
    return i, j, k


def get_sim(sim_path):
    (b_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    b, k = read_hdf(b_path, ["Data-Set-2", "fakeDim2"])

    k = np.expand_dims(np.expand_dims(k, axis=0), axis=0)
    b *= k**2

    return b


def get_sims(sim_paths):
    sims = []
    for sim_path in sim_paths:
        sims.append(get_sim(sim_path))
    sims = np.stack(sims, axis=0)
    return sims


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = array - min_ / (max_ - min_)
    return array, min_, max_


class PointDataset(Dataset):
    def __init__(
        self,
        sim_paths,
        input_slice_ix=0,
        target_slice_ix=70,
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
        self.input_slice_ix = input_slice_ix
        self.target_slice_ix = target_slice_ix
        sims = get_sims(sim_paths)
        sims, self.b_min, self.b_max = min_max_normalize(sims, b_min, b_max)
        self.ii, self.jj, self.kk = get_ijk(sim_paths[0])
        self.r = self.kk[input_slice_ix]
        self.xx, self.yy, self.zz = get_xyz(self.ii, self.jj, self.r)
        self.sims = sims

    def __getitem__(self, index):
        cube = self.sims[index]
        input_slice = cube[:, :, self.input_slice_ix].ravel()
        target_slice = cube[:, :, self.target_slice_ix].ravel()
        print(self.xx.shape, self.yy.shape, self.zz.shape, input_slice.shape)
        input_points = np.row_stack((self.xx, self.yy, self.zz, input_slice))
        target_points = np.expand_dims(target_slice.ravel(), axis=-1)
        return torch.tensor(input_points, dtype=torch.float32), torch.tensor(
            target_points, dtype=torch.float32
        )

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"b_min": float(self.b_min), "b_max": float(self.b_max)}
