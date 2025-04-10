import torch
import cv2 as cv
import numpy as np
from pyhdf.SD import SD, SDC
from tqdm import tqdm
from torch.utils.data import Dataset
from os.path import join as path_join


FILE_NAMES = ["vr002.hdf", "rho002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def resize_3d(array, new_height, new_width):
    resized_array = np.zeros((array.shape[0], new_height, new_width), dtype=array.dtype)

    for i in range(array.shape[0]):
        resized_array[i] = cv.resize(
            array[i], (new_width, new_height), interpolation=cv.INTER_LINEAR
        )

    return resized_array


def get_sim(sim_path, height, width):
    (v_path, rho_path) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    [v] = read_hdf(v_path, ["Data-Set-2"])
    [rho] = read_hdf(rho_path, ["Data-Set-2"])

    v = v.transpose(2, 1, 0)
    rho = rho.transpose(2, 1, 0)

    if height != v.shape[1] or width != v.shape[2]:
        v = resize_3d(v, height, width)
        rho = resize_3d(rho, height, width)

    return v, rho


def get_sims(sim_paths, height, width):
    vs, rhos = [], []
    for sim_path in tqdm(sim_paths):
        v, rho = get_sim(sim_path, height, width)
        vs.append(v)
        rhos.append(rho)
    return vs, rhos


def min_max_normalize(cubes, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_, max_ = np.min(cubes), np.max(cubes)
    cubes = (np.array(cubes, dtype=np.float32) - min_) / (max_ - min_)
    return cubes, min_, max_


class SphericalNODataset(Dataset):
    def __init__(
        self,
        sim_paths,
        height=111,
        width=128,
        v_min=None,
        v_max=None,
        rho_min=None,
        rho_max=None,
    ):
        super().__init__()
        vs, rhos = get_sims(sim_paths, height, width)
        vs, self.v_min, self.v_max = min_max_normalize(vs, v_min, v_max)
        rhos, self.rho_min, self.rho_max = min_max_normalize(rhos, rho_min, rho_max)
        self.vs, self.rhos = vs, rhos

    def __getitem__(self, index):
        v = self.vs[index]
        rho = self.rhos[index]
        x = np.concatenate((v[0], rho[0]), axis=0)
        y_stacked = np.stack((v[1:], rho[1:]), axis=1)
        y = y_stacked.reshape(-1, *v.shape[1:])
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.vs)

    def get_min_max(self):
        return {
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "rho_min": float(self.rho_min),
            "rho_max": float(self.rho_max),
        }
