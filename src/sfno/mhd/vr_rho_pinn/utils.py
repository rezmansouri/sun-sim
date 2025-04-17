import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
from pyhdf.SD import SD, SDC
from tqdm import tqdm
from torch.utils.data import Dataset
from os.path import join as path_join


FILE_NAMES = ["vr002.hdf", "rho002.hdf", "p002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_radii(sim_path):
    (v_path, _, _) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    radii = read_hdf(v_path, ["fakeDim2"])[0]
    return radii


def resize_3d(array, new_height, new_width):
    resized_array = np.zeros((array.shape[0], new_height, new_width), dtype=array.dtype)

    for i in range(array.shape[0]):
        resized_array[i] = cv.resize(
            array[i], (new_width, new_height), interpolation=cv.INTER_LINEAR
        )

    return resized_array


def get_sim(sim_path, height, width):
    (v_path, rho_path, p_path) = [
        path_join(sim_path, file_name) for file_name in FILE_NAMES
    ]
    v = read_hdf(v_path, ["Data-Set-2"])[0]
    rho = read_hdf(rho_path, ["Data-Set-2"])[0]
    p = read_hdf(p_path, ["Data-Set-2"])[0]

    v = v.transpose(2, 1, 0)
    rho = rho.transpose(2, 1, 0)
    p = p.transpose(2, 1, 0)

    if height != v.shape[1] or width != v.shape[2]:
        v = resize_3d(v, height, width)
        rho = resize_3d(rho, height, width)

    return v, rho[:-1, :, :], p


def get_sims(sim_paths, height, width):
    vs, rhos, ps = [], [], []
    for sim_path in tqdm(sim_paths):
        v, rho, p = get_sim(sim_path, height, width)
        vs.append(v)
        rhos.append(rho)
        ps.append(p)
    return vs, rhos, ps


def min_max_normalize(cubes, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_, max_ = np.min(cubes), np.max(cubes)
    cubes = (np.array(cubes, dtype=np.float32) - min_) / (max_ - min_)
    return cubes, min_, max_


def min_max_inverse(cubes, min_, max_):
    cubes = cubes * (max_ - min_) + min_
    return cubes


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
        self.r = get_radii(sim_paths[0])
        vs, rhos, ps = get_sims(sim_paths, height, width)
        vs, self.v_min, self.v_max = min_max_normalize(vs, v_min, v_max)
        rhos, self.rho_min, self.rho_max = min_max_normalize(rhos, rho_min, rho_max)
        self.vs, self.rhos, self.ps = vs, rhos, ps

    def __getitem__(self, index):
        v = self.vs[index]
        rho = self.rhos[index]
        p = self.ps[index]
        x = np.stack((v[0], rho[0]), axis=0)
        y_stacked = np.stack((v[1:], rho[1:]), axis=1)
        y = y_stacked.reshape(-1, *v.shape[1:])
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "p": torch.tensor(p, dtype=torch.float32),
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


class PhysicalLaw:

    def __init__(
        self,
        r,
        v_min,
        v_max,
        rho_min,
        rho_max,
        g=6.6743,
        v_constant=481.3711,
        rho_constant=1.6726e-16,
    ):
        self.dr = float(r[1] - r[0])
        self.v_min = v_min
        self.v_max = v_max
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.g = g
        self.v_constant = v_constant
        self.rho_constant = rho_constant

    def forward(self, pred, p):
        """
        pred: (batch, 278, 111, 128). pred[:, 0, 111, 128] is v, pred[:, 1, 111, 128] is rho
        """
        v_pred = pred[:, 0::2, :, :]  # shape: (batch, 139, 111, 128)
        rho_pred = pred[:, 1::2, :, :]  # shape: (batch, 139, 111, 128)

        v_pred = min_max_inverse(v_pred, self.v_min, self.v_max)

        rho_pred = min_max_inverse(rho_pred, self.rho_min, self.rho_max)

        v_pred = v_pred * self.v_constant

        rho_pred = rho_pred * self.rho_constant

        dv_r_dr = torch.gradient(v_pred, dim=1)[0] / self.dr
        # First derivative of v_r (radial velocity) w.r.t. r (along axis 0)
        d_p_dr = torch.gradient(p, dim=1)[0] / self.dr
        # First derivative of pressure w.r.t. r (1D array)

        term1 = rho_pred * v_pred * dv_r_dr  # Convective term: rho * v_r * (dv_r / dr)
        term2 = -d_p_dr  # Pressure gradient term: - dp / dr
        term3 = rho_pred * self.g  # Gravitational term: rho * g

        value = term2 + term3 - term1

        return value
