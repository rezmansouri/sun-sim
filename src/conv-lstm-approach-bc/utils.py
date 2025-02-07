import os
import torch
import cv2 as cv
import numpy as np
from pyhdf.SD import SD, SDC
from os.path import join as path_join


FILE_NAMES = ["vr_r0.hdf", "vr002.hdf", "rho002.hdf"]  # , "br_r0.hdf", "br002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def pad_3d_array_to_square(three_d_array):
    # assuming h > w
    h, w = three_d_array.shape[:2]
    assert h >= w, f"h is not geq than w, h: {h}, w:{w}"
    pad_left = (h - w) // 2
    pad_right = (h - w) - pad_left
    if pad_left > 0 or pad_right > 0:
        three_d_array = np.pad(
            three_d_array, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge"
        )
    return three_d_array


def pad_2d_array_to_square(two_d_array):
    # assuming h > w
    h, w = two_d_array.shape
    assert h >= w, f"h is not geq than w, h: {h}, w:{w}"
    pad_left = (h - w) // 2
    pad_right = (h - w) - pad_left
    if pad_left > 0 or pad_right > 0:
        two_d_array = np.pad(two_d_array, ((0, 0), (pad_left, pad_right)), mode="edge")
    return two_d_array


def resize(two_d_array, size):
    if two_d_array.shape == (size, size):
        return two_d_array
    return cv.resize(two_d_array, (size, size), interpolation=cv.INTER_LANCZOS4)


def get_sim(sim_path):
    v_bc_path, v_path, rho_path = [
        path_join(sim_path, file_name) for file_name in FILE_NAMES
    ]
    v_bc, v_bc_i, v_bc_j = read_hdf(v_bc_path, ["Data-Set-2", "fakeDim0", "fakeDim1"])
    v, v_i, v_j, v_k = read_hdf(
        v_path, ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"]
    )
    rho, rho_i, rho_j, rho_k = read_hdf(
        rho_path, ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"]
    )

    v_bc = pad_2d_array_to_square(v_bc)
    v_bc = resize(v_bc, v.shape[0])
    v = pad_3d_array_to_square(v)

    rho = pad_3d_array_to_square(rho)

    # Reshape to (K, I, J)
    v = np.transpose(v, (2, 0, 1))
    rho = np.transpose(rho, (2, 0, 1))

    v_bc = np.expand_dims(v_bc, axis=0)
    v = np.concatenate([v_bc, v], axis=0)

    sim = np.stack([v, rho], axis=1)

    return sim


def get_sims(cr_paths, instruments):
    sims = []
    for cr_path in cr_paths:
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if not os.path.exists(instrument_path):
                continue
            sims.append(get_sim(instrument_path))
    sims = np.stack(sims, axis=0)
    return sims


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = array - min_ / (max_ - min_)
    return array, min_, max_


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cr_paths,
        teacher_forcing=True,
        starting_slice=0,
        instruments=[
            "kpo_mas_mas_std_0101",
            "mdi_mas_mas_std_0101",
            "hmi_mast_mas_std_0101",
            "hmi_mast_mas_std_0201",
            "hmi_masp_mas_std_0201",
            "mdi_mas_mas_std_0201",
        ],
        v_min=None,
        v_max=None,
        rho_min=None,
        rho_max=None,
    ):
        super().__init__()
        self.instruments = instruments
        self.teacher_forcing = teacher_forcing
        self.starting_slice = starting_slice
        sims = get_sims(cr_paths, self.instruments)
        sims[:, :, 0, :, :], self.v_min, self.v_max = min_max_normalize(
            sims[:, :, 0, :, :], v_min, v_max
        )
        sims[:, :, 1, :, :], self.rho_min, self.rho_max = min_max_normalize(
            sims[:, :, 1, :, :], rho_min, rho_max
        )
        _, self.k, _, self.i, self.j = sims.shape
        self.sims = sims

    def __getitem__(self, index):
        cube = self.sims[index]
        # with teacher forcing
        if self.teacher_forcing:
            return cube[:-1, :, :, :], cube[1:, :, :, :]
        # without teacher forcing
        return (
            cube[self.starting_slice : self.starting_slice + 1, :, :, :],
            cube[self.starting_slice + 1 :, :, :, :],
        )

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "rho_min": float(self.rho_min),
            "rho_max": float(self.rho_max),
        }
