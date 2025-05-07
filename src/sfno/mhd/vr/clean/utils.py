import torch
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC
from torch.utils.data import Dataset
from os.path import join as path_join
import os

FILE_NAMES = ["vr002.hdf"]

DEFAULT_INSTRUMENTS = [
    "kpo_mas_mas_std_0101",
    "mdi_mas_mas_std_0101",
    "hmi_mast_mas_std_0101",
    "hmi_mast_mas_std_0201",
    "hmi_masp_mas_std_0201",
    "mdi_mas_mas_std_0201",
]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_sim(sim_path):
    (v_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    v = read_hdf(v_path, ["Data-Set-2"])[0]
    v = v.transpose(2, 1, 0)

    return v


def get_sims(sim_paths):
    sims = []
    for sim_path in tqdm(sim_paths, desc="Loading simulations"):
        sims.append(get_sim(sim_path))
    sims = np.stack(sims, axis=0)
    return sims


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = (array - min_) / (max_ - min_)
    return array, min_, max_


def compute_climatology(data: np.ndarray) -> np.ndarray:
    """
    Compute per-voxel climatology (mean field) from a dataset.

    Args:
        data (np.ndarray): Array of shape (n, 139, 111, 128)

    Returns:
        np.ndarray: Climatology array of shape (139, 111, 128)
    """
    assert data.ndim == 4 and data.shape[1:] == (
        139,
        111,
        128,
    ), "Unexpected input shape."
    climatology = np.mean(data, axis=0)
    climatology = torch.tensor(climatology, dtype=torch.float32)
    return climatology


def get_cr_dirs(data_path):
    """Return list of CR directories (crXXXX) inside data_path."""
    cr_dirs = sorted(
        [
            d
            for d in os.listdir(data_path)
            if d.startswith("cr") and os.path.isdir(os.path.join(data_path, d))
        ]
    )
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cr_dirs)
    return cr_dirs


def collect_sim_paths(data_path, cr_list, instruments=None):
    """Collect simulation paths given a list of CR directories."""
    if instruments is None:
        instruments = DEFAULT_INSTRUMENTS

    sim_paths = []
    for cr in cr_list:
        cr_path = os.path.join(data_path, cr)
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                sim_paths.append(instrument_path)
    return sim_paths


class SphericalNODataset(Dataset):
    def __init__(
        self,
        data_path,
        cr_list,
        v_min=None,
        v_max=None,
        instruments=None,
    ):
        super().__init__()
        sim_paths = collect_sim_paths(data_path, cr_list, instruments)
        sims = get_sims(sim_paths)
        sims, self.v_min, self.v_max = min_max_normalize(sims, v_min, v_max)
        self.sims = sims
        self.climatology = compute_climatology(sims[:, 1:, :, :])

    def __getitem__(self, index):
        cube = self.sims[index]
        return {
            "x": torch.tensor(cube[0, :, :], dtype=torch.float32).unsqueeze(0),
            "y": torch.tensor(cube[1:, :, :], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}
