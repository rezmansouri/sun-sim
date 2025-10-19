import torch
import math
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC
from torch.utils.data import Dataset
from os.path import join as path_join
from scipy.ndimage import zoom
import os
from neuralop.losses import H1Loss

FILE_NAMES = ["jr002.hdf", "vr002.hdf"]

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


def get_coords(sim_path):
    (
        _,
        v_path,
    ) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    radii, thetas, phis = read_hdf(v_path, ["fakeDim2", "fakeDim1", "fakeDim0"])
    return radii, thetas, phis


def enlarge_cube(cube, scale):
    """
    Enlarge the spatial dimensions (axis 1 and 2) of a 3D cube using bilinear interpolation.

    Parameters:
    - cube: np.ndarray of shape (140, 110, 128)
    - scale: int or float (e.g., 2)

    Returns:
    - enlarged_cube: np.ndarray of shape (140, 110 * scale, 128 * scale)
    """
    return zoom(cube, (1, scale, scale), order=1)


def get_sim(sim_path, scale_up, radii):
    (
        j_path,
        v_path,
    ) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    v = read_hdf(v_path, ["Data-Set-2"])[0]
    v = v.transpose(2, 1, 0)
    j = read_hdf(j_path, ["Data-Set-2"])[0]
    j = j.transpose(2, 1, 0)

    broadcasted_radii = np.broadcast_to(radii[:, np.newaxis, np.newaxis], (140, 111, 128))
    j = j * (broadcasted_radii**2)

    if scale_up != 1:
        v = enlarge_cube(v, scale_up)

    v = np.expand_dims(v, axis=1)  # (140, 1, H, W)
    j = np.expand_dims(j, axis=1)  # (140, 1, H, W)
    
    return v, j


def get_sims(sim_paths, scale_up):  # , pos_emb):
    vs = []
    js = []
    print(sim_paths)
    radii, _, _ = get_coords(sim_paths[0])  # (140,), (111,), (128,)

    # # Broadcast coordinate grids
    # R, T, P = np.meshgrid(radii, thetas, phis, indexing="ij")  # shapes (140, 111, 128)

    # # Normalize angles for embeddings
    # T_norm = T / np.pi  # θ ∈ [0, π] → [0,1]
    # P_cos = np.cos(P)  # periodic encoding
    # P_sin = np.sin(P)

    for sim_path in tqdm(sim_paths, desc="Loading simulations"):
        v, j = get_sim(sim_path, scale_up, radii)  # (140, 111, 128)

        # if pos_emb == "pt":
        #     # Embed only angular coords
        #     # stack channels: [sim, θ, cos φ, sin φ]
        #     sim_emb = np.stack(
        #         [sim, T_norm, P_cos, P_sin], axis=0
        #     )  # (C=4, 140, 111, 128)

        # elif pos_emb == "ptr":
        #     # Embed radius too
        #     R_norm = (R - R.min()) / (R.max() - R.min())
        #     sim_emb = np.stack(
        #         [sim, R_norm, T_norm, P_cos, P_sin], axis=0
        #     )  # (C=5, 140, 111, 128)

        # else:
        # v_emb = v[None, ...]  # (1, 140, 111, 128)
        # j_emb = j[None, ...]  # (1, 140, 111, 128)

        vs.append(v)
        js.append(j)

    vs = np.stack(vs, axis=0)  # (N, 140, 111, 128)
    js = np.stack(js, axis=0)  # (N, 140, 111, 128)
    return vs, js  # (radii, thetas, phis)


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = (array - min_) / (max_ - min_ + 1e-9)
    return array, min_, max_


def get_cr_dirs(data_path):
    """Return list of CR directories (crXXXX) inside data_path."""
    cr_dirs = sorted(
        [
            d
            for d in os.listdir(data_path)
            if d.startswith("cr") and os.path.isdir(os.path.join(data_path, d))
        ]
    )
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
        scale_up,
        v_min=None,
        v_max=None,
        j_min=None,
        j_max=None,
        instruments=None,
        # positional_embedding=None,
    ):
        super().__init__()
        self.sim_paths = collect_sim_paths(data_path, cr_list, instruments)
        vs, js = get_sims(self.sim_paths, scale_up)  # , positional_embedding)
        vs, self.v_min, self.v_max = min_max_normalize(vs, v_min, v_max)
        js, self.j_min, self.j_max = min_max_normalize(js, j_min, j_max)
        self.vs = vs
        self.js = js

    def __getitem__(self, index):
        v = self.vs[index]
        j = self.js[index]
        
        print(f"v shape in dataset: {v.shape}")
        print(f"j shape in dataset: {j.shape}")
        return {
            "vx": torch.tensor(v[0, :, :, :], dtype=torch.float32),
            "vy": torch.tensor(v[1:, :, :, :], dtype=torch.float32),
            "jx": torch.tensor(j[0, :, :, :], dtype=torch.float32),
            "jy": torch.tensor(j[1:, :, :, :], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.vs)

    def get_min_max(self):
        return {
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "j_min": float(self.j_min),
            "j_max": float(self.j_max),
        }

    def get_grid_points(self):
        return get_coords(self.sim_paths[0])


class H1LossSpherical(H1Loss):
    def __init__(
        self,
        r_grid,
        theta_grid,
        phi_grid,
        reduction="sum",
        fix_x_bnd=True,
        fix_y_bnd=True,
        fix_z_bnd=False,
    ):
        # spherical is always 3D
        super().__init__(
            d=3,
            measure=[float(207.94533), float(3.1704147), float(6.234098)],
            reduction=reduction,
            fix_x_bnd=fix_x_bnd,
            fix_y_bnd=fix_y_bnd,
            fix_z_bnd=fix_z_bnd,
        )

        # Store coordinate grids (1D arrays of r, theta, phi)
        r_grid = torch.tensor(r_grid, dtype=torch.float32)
        theta_grid = torch.tensor(theta_grid, dtype=torch.float32)
        phi_grid = torch.tensor(phi_grid, dtype=torch.float32)

        # Build Jacobian weights r^2 sin(theta)
        R, Theta, Phi = torch.meshgrid(r_grid, theta_grid, phi_grid, indexing="ij")
        self.jacobian = (R**2) * torch.sin(Theta)

    def abs(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        # Differential cell volume = dr * dθ * dφ
        const = math.prod(quadrature)

        # Apply Jacobian pointwise
        J = self.jacobian.to(x.device)

        diff = (
            const * torch.norm((dict_x[0] - dict_y[0]) * J.flatten(), p=2, dim=-1) ** 2
        )
        for j in range(1, self.d + 1):
            diff += (
                const
                * torch.norm((dict_x[j] - dict_y[j]) * J.flatten(), p=2, dim=-1) ** 2
            )

        diff = diff**0.5
        return self.reduce_all(diff).squeeze()

    def rel(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)
        const = math.prod(quadrature)
        J = self.jacobian.to(x.device)

        diff = torch.norm((dict_x[0] - dict_y[0]) * J.flatten(), p=2, dim=-1) ** 2
        ynorm = torch.norm(dict_y[0] * J.flatten(), p=2, dim=-1) ** 2

        for j in range(1, self.d + 1):
            diff += torch.norm((dict_x[j] - dict_y[j]) * J.flatten(), p=2, dim=-1) ** 2
            ynorm += torch.norm(dict_y[j] * J.flatten(), p=2, dim=-1) ** 2

        diff = (diff**0.5) / (ynorm**0.5)
        return self.reduce_all(diff).squeeze()


class H1LossSphericalMAE(H1Loss):
    def __init__(
        self,
        r_grid,
        theta_grid,
        phi_grid,
        reduction="sum",
        fix_x_bnd=True,
        fix_y_bnd=True,
        fix_z_bnd=False,
    ):
        # spherical is always 3D
        super().__init__(
            d=3,
            measure=[float(207.94533), float(3.1704147), float(6.234098)],
            reduction=reduction,
            fix_x_bnd=fix_x_bnd,
            fix_y_bnd=fix_y_bnd,
            fix_z_bnd=fix_z_bnd,
        )

        # Store coordinate grids (1D arrays of r, theta, phi)
        r_grid = torch.tensor(r_grid, dtype=torch.float32)
        theta_grid = torch.tensor(theta_grid, dtype=torch.float32)
        phi_grid = torch.tensor(phi_grid, dtype=torch.float32)

        # Build Jacobian weights r^2 sin(theta)
        R, Theta, Phi = torch.meshgrid(r_grid, theta_grid, phi_grid, indexing="ij")
        self.jacobian = (R**2) * torch.sin(Theta)  # shape (Nr, Nθ, Nφ)

    def abs(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        # Differential cell volume = dr * dθ * dφ
        const = math.prod(quadrature)

        # Jacobian with same shape as spatial grid
        J = self.jacobian.to(x.device)
        J_flat = J.reshape(-1)  # match flattened dict_x entries

        # --- Absolute H1 loss with MAE ---
        diff = const * torch.mean(torch.abs((dict_x[0] - dict_y[0]) * J_flat), dim=-1)

        for j in range(1, self.d + 1):
            diff += const * torch.mean(
                torch.abs((dict_x[j] - dict_y[j]) * J_flat), dim=-1
            )

        return self.reduce_all(diff).squeeze()

    def rel(self, x, y, quadrature=None):
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)
        const = math.prod(quadrature)
        J = self.jacobian.to(x.device)
        J_flat = J.reshape(-1)

        # numerator = MAE of difference
        diff = torch.mean(torch.abs((dict_x[0] - dict_y[0]) * J_flat), dim=-1)
        # denominator = MAE of target
        ynorm = torch.mean(torch.abs(dict_y[0] * J_flat), dim=-1)

        for j in range(1, self.d + 1):
            diff += torch.mean(torch.abs((dict_x[j] - dict_y[j]) * J_flat), dim=-1)
            ynorm += torch.mean(torch.abs(dict_y[j] * J_flat), dim=-1)

        diff = (diff / ynorm) * const
        return self.reduce_all(diff).squeeze()
