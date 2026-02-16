import torch
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator
import imageio
from io import BytesIO
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_sim(sim_path, system="cgs"):
    out, new_r, new_theta, phi = read_sim(sim_path)
    final = dict()
    for component in out:
        data_old, y_old, z_old = (
            out[component]["data"],
            out[component]["theta"],
            out[component]["r"],
        )
        data_new = interpolate_cube(data_old, phi, y_old, z_old, phi, new_theta, new_r)
        data_new = np.transpose(data_new, (2, 1, 0))  # (r, theta, phi)
        print(f"{component} shape after interpolation and transpose: ", data_new.shape)
        final[component] = data_new
    system_func = cgs if system == "cgs" else mks
    final["r"] = new_r
    final, coefficients = system_func(final)
    return (
        final,
        coefficients,
        torch.tensor(new_theta, dtype=torch.float64),
        torch.tensor(phi, dtype=torch.float64),
    )


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
        bt_theta,
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


def cgs(data_dict):
    # Equation 1
    data_dict["r"] = (
        torch.tensor(data_dict["r"], dtype=torch.float64) * 6.96 * 1e10
    )  # cm
    data_dict["jr"] = (
        torch.tensor(data_dict["jr"], dtype=torch.float64) * 0.07558
    )  # statamp / cm^2
    data_dict["jt"] = (
        torch.tensor(data_dict["jt"], dtype=torch.float64) * 0.07558
    )  # statamp / cm^2
    data_dict["jp"] = (
        torch.tensor(data_dict["jp"], dtype=torch.float64) * 0.07558
    )  # statamp / cm^2
    data_dict["br"] = (
        torch.tensor(data_dict["br"], dtype=torch.float64) * 2.2068908
    )  # Gauss
    data_dict["bt"] = (
        torch.tensor(data_dict["bt"], dtype=torch.float64) * 2.2068908
    )  # Gauss
    data_dict["bp"] = (
        torch.tensor(data_dict["bp"], dtype=torch.float64) * 2.2068908
    )  # Gauss
    coefficients = {
        "G": 6.67430 * 1e-8,  # cm^3 / (g * s^2)
        "M": 1.9885 * 1e33,  # g
        "omega_rot": 2.84 * 1e-6,  # rad / s
        "c": 2.99792458e10,  # cm / s
        "nu": 5.0 * 1e-3 * 3.350342628857710e18,  # cm^2 / s
    }
    # jr = jr * 0.07558           # statamp / cm^2
    return data_dict, coefficients


def mks(data_dict):
    # Equation 1
    data_dict["r"] = torch.tensor(data_dict["r"], dtype=torch.float64) * 6.96 * 1e8  # m
    data_dict["jr"] = (
        torch.tensor(data_dict["jr"], dtype=torch.float64) * 2.52 * 1e-7
    )  # A / m^2
    data_dict["jt"] = (
        torch.tensor(data_dict["jt"], dtype=torch.float64) * 2.52 * 1e-7
    )  # A / m^2
    data_dict["jp"] = (
        torch.tensor(data_dict["jp"], dtype=torch.float64) * 2.52 * 1e-7
    )  # A / m^2
    data_dict["br"] = (
        torch.tensor(data_dict["br"], dtype=torch.float64) * 2.2068908 * 1e-4
    )  # Tesla
    data_dict["bt"] = (
        torch.tensor(data_dict["bt"], dtype=torch.float64) * 2.2068908 * 1e-4
    )  # Tesla
    data_dict["bp"] = (
        torch.tensor(data_dict["bp"], dtype=torch.float64) * 2.2068908 * 1e-4
    )  # Tesla
    coefficients = {
        "G": 6.67430 * 1e-11,  # m^3 / (kg * s^2)
        "M": 1.9885 * 1e30,  # kg
        "omega_rot": 2.84 * 1e-6,  # rad / s
        "c": 2.99792458 * 1e8,  # m / s
        "nu": 5.0 * 1e-7 * 3.350342628857710e18,  # m^2 / s
    }
    # jr = jr * 2.52 * 1e-7  # * 2.267 * 1e4 (MAS Guide)       # A / m^2
    return data_dict, coefficients


# def mks(r, vr, rho, p, jr, br):
#     # Equation 6
#     r = r * 6.96 * 1e8  # m
#     vr = vr * 481.3711 * 1e3  # m / s
#     rho = rho * 1.6726 * 1e-13  # kg / m^3
#     p = p * 0.03875717  # Pascals (N / m^2)
#     jr = jr * 2.52 * 1e-7  # * 2.267 * 1e4 (MAS Guide)       # A / m^2
#     br = br * 2.2068908 * 1e-4  # Tesla
#     G = 6.67430 * 1e-11  # m^3 / (kg * s^2)
#     M_sun = 1.9885 * 1e30  # kg
#     omega_rot = 2.84 * 1e-6  # rad / s
#     c = 2.99792458 * 1e8  # m / s
#     viscosity = 5.0 * 1e-7 * 3.350342628857710e18  # m^2 / s
#     return r, vr, rho, p, jr, br, G, M_sun, omega_rot, c, viscosity


def build_analysis_mask(residual, n_exclude_r=0, n_exclude_theta=2):

    mask = torch.ones_like(residual, dtype=torch.bool)

    mask[:, :n_exclude_theta] = False
    mask[:, -n_exclude_theta:] = False

    return mask


def detailed_residual_metrics(lhs, rhs, residual, mask=None):

    lhs_r, lhs_t, lhs_p = lhs
    rhs_r, rhs_t, rhs_p = rhs
    Rr, Rt, Rp = residual

    mag_lhs = torch.sqrt(lhs_r**2 + lhs_t**2 + lhs_p**2)
    mag_rhs = torch.sqrt(rhs_r**2 + rhs_t**2 + rhs_p**2)
    mag_R = torch.sqrt(Rr**2 + Rt**2 + Rp**2)

    if mask is not None:
        mask = mask.bool()
        mag_lhs = mag_lhs[mask]
        mag_rhs = mag_rhs[mask]
        mag_R = mag_R[mask]

    rms_lhs = torch.sqrt(torch.mean(mag_lhs**2))
    rms_rhs = torch.sqrt(torch.mean(mag_rhs**2))
    rms_R = torch.sqrt(torch.mean(mag_R**2))

    rel = mag_R / (mag_lhs + mag_rhs + 1e-30)

    metrics = {
        "RMS_LHS": rms_lhs.item(),
        "RMS_RHS": rms_rhs.item(),
        "RMS_RES": rms_R.item(),
        "RMS_RES/LHS": (rms_R / (rms_lhs + 1e-30)).item(),
        "RMS_RES/RHS": (rms_R / (rms_rhs + 1e-30)).item(),
        "rel_p50": torch.quantile(rel, 0.50).item(),
        "rel_p90": torch.quantile(rel, 0.90).item(),
        "rel_p95": torch.quantile(rel, 0.95).item(),
        "rel_p99": torch.quantile(rel, 0.99).item(),
        "rel_max": torch.max(rel).item(),
    }

    return metrics


def radial_rms_profile(residual, mask=None):

    Rr, Rt, Rp = residual
    mag_R = torch.sqrt(Rr**2 + Rt**2 + Rp**2)

    Nr = mag_R.shape[0]
    profile = []

    for i in range(Nr):

        slice_R = mag_R[i]

        if mask is not None:
            m2 = mask[i].bool()
            slice_R = slice_R[m2]

        if slice_R.numel() == 0:
            profile.append(float("nan"))
        else:
            profile.append(torch.sqrt(torch.mean(slice_R**2)).item())

    return profile


def make_residual_gif(
    output_filename, Rr, Rt, Rp, mask, fps=8, cmap="coolwarm", scale="global"
):

    Rr = Rr.detach().cpu()
    Rt = Rt.detach().cpu()
    Rp = Rp.detach().cpu()
    mask = mask.detach().cpu().bool()

    Rmag = torch.sqrt(Rr**2 + Rt**2 + Rp**2)
    Nr = Rmag.shape[0]

    if scale == "global":
        vmax = torch.max(Rmag[mask])
        vmin = 0.0

    frames = []

    for i in trange(Nr):

        slice_R = Rmag[i]
        m2 = mask[i]

        if scale == "local":
            vmax = torch.max(slice_R[m2])
            vmin = 0.0

        img = slice_R.clone()
        img[~m2] = float("nan")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"|Residual| r-slice {i+1}/{Nr}")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(im, orientation="horizontal", pad=0)
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    with imageio.get_writer(output_filename, fps=fps, loop=0) as writer:
        for f in frames:
            writer.append_data(f)

    print("Saved:", output_filename)


def make_lhs_rhs_residual_gif(
    output_filename, lhs, rhs, residual, mask, fps=8, cmap="coolwarm"
):

    lhs_r, lhs_t, lhs_p = lhs
    rhs_r, rhs_t, rhs_p = rhs
    Rr, Rt, Rp = residual

    # Magnitudes
    L = torch.sqrt(lhs_r**2 + lhs_t**2 + lhs_p**2).detach().cpu()
    H = torch.sqrt(rhs_r**2 + rhs_t**2 + rhs_p**2).detach().cpu()
    R = torch.sqrt(Rr**2 + Rt**2 + Rp**2).detach().cpu()
    mask = mask.detach().cpu().bool()

    # ---- GLOBAL COLOR LIMITS (masked region only) ----
    all_values = torch.cat([L[mask], H[mask], R[mask]])

    vmin = torch.min(all_values)
    vmax = torch.max(all_values)

    Nr = L.shape[0]
    frames = []

    for i in trange(Nr):

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        im0 = axes[0].imshow(L[i], vmin=vmin, vmax=vmax, cmap=cmap)
        axes[0].set_title("|LHS|")

        im1 = axes[1].imshow(H[i], vmin=vmin, vmax=vmax, cmap=cmap)
        axes[1].set_title("|RHS|")

        im2 = axes[2].imshow(R[i], vmin=vmin, vmax=vmax, cmap=cmap)
        axes[2].set_title("|RES|")

        fig.suptitle(f"r-slice {i+1}/{Nr}")

        # ---- SINGLE SHARED HORIZONTAL COLORBAR ----
        cbar = fig.colorbar(
            im2, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08
        )
        cbar.set_label("Magnitude")

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    with imageio.get_writer(output_filename, fps=fps, loop=0) as writer:
        for f in frames:
            writer.append_data(f)

    print("Saved:", output_filename)


def full_numerical_analysis(
    lhs,
    rhs,
    residual,
    r=None,
    theta=None,
    phi=None,
    mask=None,
    volume_weighted=False,
):

    Lr, Lt, Lp = lhs
    Rr, Rt, Rp = rhs
    Res_r, Res_t, Res_p = residual

    # --------------------------------------------------
    # Magnitudes (full 3D)
    # --------------------------------------------------

    mag_L = torch.sqrt(Lr**2 + Lt**2 + Lp**2)
    mag_R = torch.sqrt(Rr**2 + Rt**2 + Rp**2)
    mag_Res = torch.sqrt(Res_r**2 + Res_t**2 + Res_p**2)

    # --------------------------------------------------
    # Volume weighting
    # --------------------------------------------------

    if volume_weighted:

        if r is None or theta is None or phi is None:
            raise ValueError("r, theta, phi must be provided for volume weighting.")

        # Build full 3D spherical element
        rr, tt, pp = torch.meshgrid(r, theta, phi, indexing="ij")

        weight = rr**2 * torch.sin(tt)

        if mask is not None:
            mask = mask.bool()
            mag_L = mag_L[mask]
            mag_R = mag_R[mask]
            mag_Res = mag_Res[mask]
            weight = weight[mask]
        else:
            weight = weight.flatten()
            mag_L = mag_L.flatten()
            mag_R = mag_R.flatten()
            mag_Res = mag_Res.flatten()

        weight = weight / torch.sum(weight)

        L2_L = torch.sqrt(torch.sum(weight * mag_L**2))
        L2_R = torch.sqrt(torch.sum(weight * mag_R**2))
        L2_Res = torch.sqrt(torch.sum(weight * mag_Res**2))

    else:

        if mask is not None:
            mask = mask.bool()
            mag_L = mag_L[mask]
            mag_R = mag_R[mask]
            mag_Res = mag_Res[mask]

        L2_L = torch.sqrt(torch.mean(mag_L**2))
        L2_R = torch.sqrt(torch.mean(mag_R**2))
        L2_Res = torch.sqrt(torch.mean(mag_Res**2))

    # --------------------------------------------------
    # Relative errors
    # --------------------------------------------------

    rel = mag_Res / (mag_L + mag_R + 1e-30)

    # --------------------------------------------------
    # Stats
    # --------------------------------------------------

    stats = {
        "N_points": mag_L.numel(),
        "L2_LHS": L2_L.item(),
        "L2_RHS": L2_R.item(),
        "L2_Residual": L2_Res.item(),
        "L2_Residual/LHS": (L2_Res / (L2_L + 1e-30)).item(),
        "Mean_|LHS|": torch.mean(mag_L).item(),
        "Mean_|RHS|": torch.mean(mag_R).item(),
        "Mean_|Res|": torch.mean(mag_Res).item(),
        "Max_|Res|": torch.max(mag_Res).item(),
        "Median_rel_error": torch.quantile(rel, 0.5).item(),
        "P90_rel_error": torch.quantile(rel, 0.9).item(),
        "P95_rel_error": torch.quantile(rel, 0.95).item(),
        "P99_rel_error": torch.quantile(rel, 0.99).item(),
        "Max_rel_error": torch.max(rel).item(),
    }

    df = pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])

    return df


import torch
import numpy as np


def term_budget_table(terms, mask=None, print_table=True):

    def compute_stats(x):

        absx = torch.abs(x)

        rms = torch.sqrt(torch.mean(x**2)).item()
        mean_abs = torch.mean(absx).item()
        p95 = torch.quantile(absx, 0.95).item()
        p99 = torch.quantile(absx, 0.99).item()
        maxv = torch.max(absx).item()

        return rms, mean_abs, p95, p99, maxv


    rows = []

    for name, comps in terms.items():

        r, t, p = comps

        if mask is not None:
            mask_bool = mask.bool()
            r = r[mask_bool]
            t = t[mask_bool]
            p = p[mask_bool]

        # Component rows
        for label, comp in zip(
            [f"{name}_r", f"{name}_t", f"{name}_p"],
            [r, t, p]
        ):
            rows.append((label,) + compute_stats(comp))

        # Magnitude row
        mag = torch.sqrt(r**2 + t**2 + p**2)
        rows.append((f"{name}_mag",) + compute_stats(mag))

    # -------------------------
    # Print formatted table
    # -------------------------

    if print_table:

        header = (
            "Term budgets (masked interior)\n"
            "          term |        rms |    mean|x| |     p95|x| |     p99|x| |     max|x|\n"
            "-------------------------------------------------------------------------------"
        )
        print(header)

        for row in rows:
            name, rms, mean_abs, p95, p99, maxv = row
            print(
                f"{name:>15} | "
                f"{rms: .3e} | "
                f"{mean_abs: .3e} | "
                f"{p95: .3e} | "
                f"{p99: .3e} | "
                f"{maxv: .3e}"
            )

        print()

    # -------------------------
    # Global balance summary
    # -------------------------

    if "lhs" in terms and "rhs" in terms and "residual" in terms:

        lhs = terms["lhs"]
        rhs = terms["rhs"]
        res = terms["residual"]

        if mask is not None:
            mask_bool = mask.bool()
            lhs_mag = torch.sqrt(
                lhs[0][mask_bool]**2 +
                lhs[1][mask_bool]**2 +
                lhs[2][mask_bool]**2
            )
            rhs_mag = torch.sqrt(
                rhs[0][mask_bool]**2 +
                rhs[1][mask_bool]**2 +
                rhs[2][mask_bool]**2
            )
            res_mag = torch.sqrt(
                res[0][mask_bool]**2 +
                res[1][mask_bool]**2 +
                res[2][mask_bool]**2
            )
        else:
            lhs_mag = torch.sqrt(lhs[0]**2 + lhs[1]**2 + lhs[2]**2)
            rhs_mag = torch.sqrt(rhs[0]**2 + rhs[1]**2 + rhs[2]**2)
            res_mag = torch.sqrt(res[0]**2 + res[1]**2 + res[2]**2)

        rms_lhs = torch.sqrt(torch.mean(lhs_mag**2)).item()
        rms_rhs = torch.sqrt(torch.mean(rhs_mag**2)).item()
        rms_res = torch.sqrt(torch.mean(res_mag**2)).item()

        rel = res_mag / (lhs_mag + rhs_mag + 1e-30)

        print(f"  RMS(LHS)   = {rms_lhs:.3e}")
        print(f"  RMS(RHS)   = {rms_rhs:.3e}")
        print(f"  RMS(RES)   = {rms_res:.3e}")
        print(f"  RMS(RES)/RMS(LHS) = {rms_res/rms_lhs:.3e}")
        print(f"  RMS(RES)/RMS(RHS) = {rms_res/rms_rhs:.3e}")

        print("  Pointwise relative residual  |R|/(|LHS|+|RHS|):")
        print(
            f"    p50={torch.quantile(rel,0.5).item():.3e}  "
            f"p90={torch.quantile(rel,0.9).item():.3e}  "
            f"p95={torch.quantile(rel,0.95).item():.3e}  "
            f"p99={torch.quantile(rel,0.99).item():.3e}  "
            f"max={torch.max(rel).item():.3e}"
        )

    return rows
