# TODO normalize x, y, z?

import torch
import numpy as np
from pyhdf.SD import SD, SDC
from os.path import join as path_join
from torch_geometric.data import Dataset

FILE_NAMES = ["br002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_ijk(sim_path):
    b_path = path_join(sim_path, FILE_NAMES[0])
    i, j, k = read_hdf(b_path, ["fakeDim0", "fakeDim1", "fakeDim2"])
    min_i, max_i = np.min(i), np.max(i)
    min_j, max_j = np.min(j), np.max(j)
    min_k, max_k = np.min(k), np.max(k)
    i = (i - min_i) / (max_i - min_i)
    j = (j - min_j) / (max_j - min_j)
    k = (k - min_k) / (max_k - min_k)
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
        b_min=None,
        b_max=None,
    ):
        super().__init__()
        sims = get_sims(sim_paths)
        sims, self.b_min, self.b_max = min_max_normalize(sims, b_min, b_max)
        self.i, self.j, self.k = get_ijk(sim_paths[0])
        self.sim_paths = sim_paths
        self.sims = sims

    def __getitem__(self, index):
        cube = self.sims[index]
        x, y = [], []
        x_intensity = cube[:, :, 0].ravel()
        for slice_ix in range(1, 141):
            radius = self.k[slice_ix]
            ii, jj = np.meshgrid(self.i, self.j, indexing="ij")
            ii = ii.ravel()
            jj = jj.ravel()
            radius = np.tile(radius, (128 * 110))
            intensity = cube[:, :, slice_ix].ravel()
            slc = np.row_stack((ii, jj, radius, x_intensity))
            y.append(np.expand_dims(intensity, axis=-1))
            x.append(slc)
        y = np.stack(y, axis=0)
        x = np.stack(x, axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"b_min": float(self.b_min), "b_max": float(self.b_max)}
