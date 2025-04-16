import torch
import cv2 as cv
import numpy as np
from pyhdf.SD import SD, SDC
from tqdm import tqdm
from torch.utils.data import Dataset
from os.path import join as path_join


FILE_NAMES = ["v002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def resize_3d(array, new_height, new_width):
    resized_array = np.zeros((array.shape[0], new_height, new_width))

    for i in range(array.shape[0]):
        resized_array[i] = cv.resize(
            array[i], (new_width, new_height), interpolation=cv.INTER_LINEAR
        )

    return resized_array


def get_sim(sim_path, new_height, new_width):
    (v_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    v = read_hdf(v_path, ["Data-Set-2"])[0]
    v = v.transpose(2, 1, 0)

    v = resize_3d(v, new_height, new_width)

    return v


def get_sims(sim_paths, new_height, new_width):
    sims = []
    for sim_path in tqdm(sim_paths):
        sims.append(get_sim(sim_path, new_height, new_width))
    sims = np.stack(sims, axis=0)
    return sims


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = (array - min_) / (max_ - min_)
    return array, min_, max_


class SphericalNODataset(Dataset):
    def __init__(
        self,
        sim_paths,
        height=111,
        width=128,
        v_min=None,
        v_max=None,
    ):
        super().__init__()
        sims = get_sims(sim_paths, height, width)
        sims, self.v_min, self.v_max = min_max_normalize(sims, v_min, v_max)
        self.sims = sims

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
