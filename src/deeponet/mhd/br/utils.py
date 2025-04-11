import torch
import cv2 as cv
import numpy as np
from pyhdf.SD import SD, SDC
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from os.path import join as path_join


FILE_NAMES = ["br002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_sim(sim_path):
    (b_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    b, phi, theta, rho = read_hdf(
        b_path, ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"]
    )

    rho = np.expand_dims(np.expand_dims(rho, axis=0), axis=0)
    b *= rho**2

    b = b.transpose(2, 1, 0)

    return b, torch.as_tensor(phi, dtype=torch.float32), torch.as_tensor(theta, dtype=torch.float32)


def get_sims(sim_paths):
    sims = []
    for sim_path in tqdm(sim_paths):
        sim, _, _ = get_sim(sim_path)
        sims.append(sim)
    sims = np.stack(sims, axis=0)
    return sims


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = (array - min_) / (max_ - min_)
    return array, min_, max_


class DeepONetDataset(Dataset):
    def __init__(
        self,
        sim_paths,
        b_min=None,
        b_max=None,
    ):
        super().__init__()
        _, phi, theta = get_sim(sim_paths[0])
        Theta, Phi = torch.meshgrid(theta, phi, indexing="ij")
        self.trunk_input = torch.stack([Theta.flatten(), Phi.flatten()], dim=-1)
        sims = get_sims(sim_paths)
        sims, self.b_min, self.b_max = min_max_normalize(sims, b_min, b_max)
        self.sims = sims

    def __getitem__(self, index):
        cube = self.sims[index]
        x = torch.tensor(cube[0, :, :], dtype=torch.float32).reshape(-1)
        y = torch.tensor(cube[1:, :, :], dtype=torch.float32).reshape(14080, -1)
        return x, y

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"b_min": float(self.b_min), "b_max": float(self.b_max)}
