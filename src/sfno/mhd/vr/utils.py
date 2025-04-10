import torch
import cv2 as cv
import numpy as np
from pyhdf.SD import SD, SDC
from tqdm import tqdm
from torch.utils.data import Dataset
from os.path import join as path_join


FILE_NAMES = ["vr_r0.hdf", "vr002.hdf"]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def resize_3d(array, new_height_y, new_width_y):
    resized_array = np.zeros(
        (array.shape[0], new_height_y, new_width_y), dtype=array.dtype
    )

    for i in range(array.shape[0]):
        resized_array[i] = cv.resize(
            array[i], (new_width_y, new_height_y), interpolation=cv.INTER_LINEAR
        )

    return resized_array


def resize_2d(array, new_height_x, new_width_x):
    resized_array = cv.resize(
        array, (new_width_x, new_height_x), interpolation=cv.INTER_LINEAR
    )

    return resized_array


def get_sim(sim_path, height_y, width_y, height_x, width_x):
    (v0_path, v_path) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    [v0] = read_hdf(v0_path, ["Data-Set-2"])
    [v] = read_hdf(v_path, ["Data-Set-2"])

    v = v.transpose(2, 1, 0)

    if height_y != v.shape[1] or width_y != v.shape[2]:
        v = resize_3d(v, height_y, width_y)

    v0 = v0.transpose(1, 0)

    if height_x != v0.shape[0] or width_x != v0.shape[1]:
        v0 = resize_2d(v0, height_x, width_x)

    return v0, v


def get_sims(sim_paths, height_y, width_y, height_x, width_x):
    slices, cubes = [], []
    for sim_path in tqdm(sim_paths):
        slc, cube = get_sim(sim_path, height_y, width_y, height_x, width_x)
        slices.append(slc)
        cubes.append(cube)
    return slices, cubes


def min_max_normalize(slices, cubes, min_=None, max_=None):
    if min_ is None or max_ is None:
        slices_min, slices_max = np.min(slices), np.max(slices)
        cubes_min, cubes_max = np.min(cubes), np.max(cubes)
        min_ = min(slices_min, cubes_min)
        max_ = max(slices_max, cubes_max)
    slices = (np.array(slices, dtype=np.float32) - min_) / (max_ - min_)
    cubes = (np.array(cubes, dtype=np.float32) - min_) / (max_ - min_)
    return slices, cubes, min_, max_


class SphericalNODataset(Dataset):
    def __init__(
        self,
        sim_paths,
        height_y,
        width_y,
        height_x,
        width_x,
        v_min=None,
        v_max=None,
    ):
        super().__init__()
        slices, cubes = get_sims(sim_paths, height_y, width_y, height_x, width_x)
        slices, cubes, self.v_min, self.v_max = min_max_normalize(
            slices, cubes, v_min, v_max
        )
        self.slices, self.cubes = slices, cubes

    def __getitem__(self, index):
        slc = self.slices[index]
        cube = self.cubes[index]
        return {
            "x": torch.tensor(slc, dtype=torch.float32).unsqueeze(0),
            "y": torch.tensor(cube, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.slices)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}
