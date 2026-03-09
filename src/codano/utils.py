import torch
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC
from torch.utils.data import Dataset
from os.path import join as path_join
from scipy.interpolate import RegularGridInterpolator
import os

FILE_NAMES = [
    "jr002.hdf",
    "jt002.hdf",
    "jr002.hdf",
    "br002.hdf",
    "bt002.hdf",
    "bp002.hdf",
]

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


def read_sim(sim_path):
    dataset_names = ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"]
    br, br_phi, br_theta, br_r = read_hdf(
        f"{sim_path}/br002.hdf", dataset_names
    )  # (128, 110, 141)
    bt, _, bt_theta, bt_r = read_hdf(
        f"{sim_path}/bt002.hdf", dataset_names
    )  # (128, 111, 140)
    bp, _, bp_theta, bp_r = read_hdf(
        f"{sim_path}/bp002.hdf", dataset_names
    )  # (128, 111, 140)
    jr, _, jr_theta, jr_r = read_hdf(
        f"{sim_path}/jr002.hdf", dataset_names
    )  # (128, 111, 140)
    jt, _, jt_theta, jt_r = read_hdf(
        f"{sim_path}/jt002.hdf", dataset_names
    )  # (128, 110, 141)
    jp, _, jp_theta, jp_r = read_hdf(
        f"{sim_path}/jp002.hdf", dataset_names
    )  # (128, 111, 141)
    return (
        {
            "br": {"data": br, "theta": br_theta, "r": br_r},
            "bt": {"data": bt, "theta": bt_theta, "r": bt_r},
            "bp": {"data": bp, "theta": bp_theta, "r": bp_r},
            "jr": {"data": jr, "theta": jr_theta, "r": jr_r},
            "jt": {"data": jt, "theta": jt_theta, "r": jt_r},
            "jp": {"data": jp, "theta": jp_theta, "r": jp_r},
        },
        bp_r,
        br_theta,
        br_phi,
    )  # {data}, new_r, new_theta, phi


def interpolate_cube(data, x_old, y_old, z_old, x_new, y_new, z_new):
    interp_func = RegularGridInterpolator(
        (x_old, y_old, z_old),
        data,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    xg, yg, zg = np.meshgrid(x_new, y_new, z_new, indexing="ij")
    points_new = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=-1)
    data_new_flat = interp_func(points_new)
    data_new = data_new_flat.reshape((len(x_new), len(y_new), len(z_new)))
    return data_new


def get_sim(sim_path):

    out, r, theta, phi = read_sim(sim_path)

    final = dict()
    for component in out:
        data_old, y_old, z_old = (
            out[component]["data"],
            out[component]["theta"],
            out[component]["r"],
        )
        data_new = interpolate_cube(data_old, phi, y_old, z_old, phi, theta, r)
        data_new = np.transpose(data_new, (2, 1, 0))  # (r, theta, phi)
        final[component] = data_new

    broadcasted_radii = np.broadcast_to(r[:, np.newaxis, np.newaxis], (140, 110, 128))
    br = final["br"] * (broadcasted_radii**2)
    bt = final["bt"] * (broadcasted_radii**2)
    bp = final["bp"] * (broadcasted_radii**2)
    jr = final["jr"] * (broadcasted_radii**2)
    jt = final["jt"] * (broadcasted_radii**2)
    jp = final["jp"] * (broadcasted_radii**2)

    br = np.expand_dims(br, axis=1)  # (140, 1, H, W)
    bt = np.expand_dims(bt, axis=1)  # (140, 1, H, W)
    bp = np.expand_dims(bp, axis=1)  # (140, 1, H, W)
    jr = np.expand_dims(jr, axis=1)  # (140, 1, H, W)
    jt = np.expand_dims(jt, axis=1)  # (140, 1, H, W)
    jp = np.expand_dims(jp, axis=1)  # (140, 1, H, W)

    return jr, jt, jp, br, bt, bp, (r, theta, phi)


def get_sims(sim_paths):  # , pos_emb):
    brs = []
    bts = []
    bps = []
    jrs = []
    jts = []
    jps = []

    # # Broadcast coordinate grids
    # R, T, P = np.meshgrid(radii, thetas, phis, indexing="ij")  # shapes (140, 111, 128)

    # # Normalize angles for embeddings
    # T_norm = T / np.pi  # θ ∈ [0, π] → [0,1]
    # P_cos = np.cos(P)  # periodic encoding
    # P_sin = np.sin(P)

    for sim_path in tqdm(sim_paths, desc="Loading simulations"):
        jr, jt, jp, br, bt, bp, _ = get_sim(sim_path)  # (140, 111, 128)

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
        # b_emb = b[None, ...]  # (1, 140, 111, 128)
        # j_emb = j[None, ...]  # (1, 140, 111, 128)

        brs.append(br)
        bts.append(bt)
        bps.append(bp)
        jrs.append(jr)
        jts.append(jt)
        jps.append(jp)

    brs = np.stack(brs, axis=0)  # (N, 140, 111, 128)
    bts = np.stack(bts, axis=0)  # (N, 140, 111, 128)
    bps = np.stack(bps, axis=0)  # (N, 140, 111, 128)
    jrs = np.stack(jrs, axis=0)  # (N, 140, 111, 128)
    jts = np.stack(jts, axis=0)  # (N, 140, 111, 128)
    jps = np.stack(jps, axis=0)  # (N, 140, 111, 128)
    return brs, bts, bps, jrs, jts, jps


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


class CODANODataset(Dataset):
    def __init__(
        self,
        data_path,
        cr_list,
        br_min=None,
        bt_min=None,
        bp_min=None,
        br_max=None,
        bt_max=None,
        bp_max=None,
        jr_min=None,
        jr_max=None,
        jt_min=None,
        jt_max=None,
        jp_min=None,
        jp_max=None,
        instruments=None,
    ):
        super().__init__()
        self.sim_paths = collect_sim_paths(data_path, cr_list, instruments)
        brs, bts, bps, jrs, jts, jps = get_sims(
            self.sim_paths
        )  # , positional_embedding)
        _, _, _, _, _, _, (r, theta, phi) = get_sim(self.sim_paths[0])
        self.r = torch.tensor(r, dtype=torch.float32)
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.phi = torch.tensor(phi, dtype=torch.float32)
        brs, self.br_min, self.br_max = min_max_normalize(brs, br_min, br_max)
        bts, self.bt_min, self.bt_max = min_max_normalize(bts, bt_min, bt_max)
        bps, self.bp_min, self.bp_max = min_max_normalize(bps, bp_min, bp_max)
        jrs, self.jr_min, self.jr_max = min_max_normalize(jrs, jr_min, jr_max)
        jts, self.jt_min, self.jt_max = min_max_normalize(jts, jt_min, jt_max)
        jps, self.jp_min, self.jp_max = min_max_normalize(jps, jp_min, jp_max)
        self.brs = brs
        self.bts = bts
        self.bps = bps
        self.jrs = jrs
        self.jts = jts
        self.jps = jps

    def __getitem__(self, index):
        br = torch.tensor(self.brs[index], dtype=torch.float32)
        bt = torch.tensor(self.bts[index], dtype=torch.float32)
        bp = torch.tensor(self.bps[index], dtype=torch.float32)
        jr = torch.tensor(self.jrs[index], dtype=torch.float32)
        jt = torch.tensor(self.jts[index], dtype=torch.float32)
        jp = torch.tensor(self.jps[index], dtype=torch.float32)

        x = torch.cat(
            [
                br[0, :, :, :],
                bt[0, :, :, :],
                bp[0, :, :, :],
                jr[0, :, :, :],
                jt[0, :, :, :],
                jp[0, :, :, :],
            ],
            dim=0,
        )

        y = torch.stack(
            [
                br[1:, 0, :, :],
                bt[1:, 0, :, :],
                bp[1:, 0, :, :],
                jr[1:, 0, :, :],
                jt[1:, 0, :, :],
                jp[1:, 0, :, :],
            ],
            dim=1,
        )

        return x, y

    def __len__(self):
        return len(self.brs)

    def get_min_max(self):
        return {
            "br_min": float(self.br_min),
            "br_max": float(self.br_max),
            "bt_min": float(self.bt_min),
            "bt_max": float(self.bt_max),
            "bp_min": float(self.bp_min),
            "bp_max": float(self.bp_max),
            "jr_min": float(self.jr_min),
            "jr_max": float(self.jr_max),
            "jt_min": float(self.jt_min),
            "jt_max": float(self.jt_max),
            "jp_min": float(self.jp_min),
            "jp_max": float(self.jp_max),
        }
