import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
from pyhdf.SD import SD, SDC
from tqdm import tqdm
from torch.utils.data import Dataset
from os.path import join as path_join
import torch.nn.functional as F


FILE_NAMES = ["vr002.hdf", "rho002.hdf", "p002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_radii(sim_path):
    (_, rho_path, _) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    radii = read_hdf(rho_path, ["fakeDim2"])[0]
    return radii


def resize_3d(array, new_height, new_width):
    resized_array = np.zeros((array.shape[0], new_height, new_width), dtype=array.dtype)

    for i in range(array.shape[0]):
        resized_array[i] = cv.resize(
            array[i], (new_width, new_height), interpolation=cv.INTER_LINEAR
        )

    return resized_array


def interpolate_1d(arr, new_len):
    """
    Linearly resample a 1-D array to `new_len` points.

    Parameters
    ----------
    arr : (N,) array_like
        Original data (here N = 140).
    new_len : int
        Desired output length.

    Returns
    -------
    out : (new_len,) ndarray
        Interpolated array.
    """
    N = len(arr)
    old_positions = np.linspace(0, N - 1, N)  # [0, 1, …, N-1]
    new_positions = np.linspace(0, N - 1, new_len)  # evenly spaced over same span
    return np.interp(new_positions, old_positions, arr)


def interp_channels(
    x: torch.Tensor, new_C: int, mode: str = "trilinear", align_corners: bool = True
):
    """
    Interpolate only the channel axis of a (C, H, W) tensor.

    Parameters
    ----------
    x            : (C, H, W)   real or float32/64 tensor
    new_C        : int         desired #channels after resampling
    mode         : str         'trilinear' | 'nearest' | 'area' (3-D modes)
    align_corners: bool        like usual; keep True for metric grids

    Returns
    -------
    (new_C, H, W) tensor
    """
    _, H, W = x.shape  # original sizes

    # (1, 1, C, H, W)  →  treat (C, H, W) as a 3-D volume (D, H, W)
    v = torch.tensor(x, dtype=torch.float32)
    v = v.unsqueeze(0).unsqueeze(0)

    # Only the depth (D=channels) changes; H and W stay the same
    v_up = F.interpolate(
        v,
        size=(new_C, H, W),  # (D_out, H_out, W_out)
        mode=mode,
        align_corners=align_corners,
    )

    return v_up.squeeze(0).squeeze(0).numpy()


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

    v = interp_channels(v, rho.shape[0], mode="trilinear", align_corners=True)

    if height != v.shape[1] or width != v.shape[2]:
        v = resize_3d(v, height, width)
        rho = resize_3d(rho, height, width)

    return v, rho, p[1:, :, :]


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
        vs, rhos, ps = get_sims(sim_paths, height, width)
        self.r = get_radii(sim_paths[0])[1:]
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
