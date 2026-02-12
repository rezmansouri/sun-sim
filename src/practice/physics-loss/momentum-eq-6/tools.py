import torch
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator
import imageio
from io import BytesIO
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


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
    rho, _, rho_theta, rho_r = read_hdf(
        f"{sim_path}/rho002.hdf", dataset_names
    )  # (128, 111, 141)
    vr, vr_phi, vr_theta, vr_r = read_hdf(
        f"{sim_path}/vr002.hdf", dataset_names
    )  # (128, 111, 140)
    vt, _, vt_theta, vt_r = read_hdf(
        f"{sim_path}/vt002.hdf", dataset_names
    )  # (128, 110, 141)
    vp, _, vp_theta, vp_r = read_hdf(
        f"{sim_path}/vp002.hdf", dataset_names
    )  # (128, 111, 141)
    br, _, br_theta, br_r = read_hdf(
        f"{sim_path}/br002.hdf", dataset_names
    )  # (128, 110, 141)
    bt, _, bt_theta, bt_r = read_hdf(
        f"{sim_path}/bt002.hdf", dataset_names
    )  # (128, 111, 140)
    bp, _, bp_theta, bp_r = read_hdf(
        f"{sim_path}/bp002.hdf", dataset_names
    )  # (128, 111, 140)
    p, _, p_theta, p_r = read_hdf(
        f"{sim_path}/p002.hdf", dataset_names
    )  # (128, 111, 141)
    return (
        {
            "rho": {"data": rho, "theta": rho_theta, "r": rho_r},
            "vr": {"data": vr, "theta": vr_theta, "r": vr_r},
            "vt": {"data": vt, "theta": vt_theta, "r": vt_r},
            "vp": {"data": vp, "theta": vp_theta, "r": vp_r},
            "br": {"data": br, "theta": br_theta, "r": br_r},
            "bt": {"data": bt, "theta": bt_theta, "r": bt_r},
            "bp": {"data": bp, "theta": bp_theta, "r": bp_r},
            "p": {
                "data": p,
                "theta": p_theta,
                "r": p_r,
            },
        },
        vr_r,
        vt_theta,
        vr_phi,
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
    # Equation 6
    data_dict["r"] = (
        torch.tensor(data_dict["r"], dtype=torch.float64) * 6.96 * 1e10
    )  # cm
    data_dict["vr"] = (
        torch.tensor(data_dict["vr"], dtype=torch.float64) * 481.3711 * 1e5
    )  # cm / s
    data_dict["vt"] = (
        torch.tensor(data_dict["vt"], dtype=torch.float64) * 481.3711 * 1e5
    )  # cm / s
    data_dict["vp"] = (
        torch.tensor(data_dict["vp"], dtype=torch.float64) * 481.3711 * 1e5
    )  # cm / s
    data_dict["rho"] = (
        torch.tensor(data_dict["rho"], dtype=torch.float64) * 1.6726 * 1e-16
    )  # g / cm^3
    data_dict["p"] = (
        torch.tensor(data_dict["p"], dtype=torch.float64) * 0.3875717
    )  # dyn / cm^2
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
    # Equation 6
    data_dict["r"] = torch.tensor(data_dict["r"], dtype=torch.float64) * 6.96 * 1e8  # m
    data_dict["vr"] = (
        torch.tensor(data_dict["vr"], dtype=torch.float64) * 481.3711 * 1e3
    )  # m / s
    data_dict["vt"] = (
        torch.tensor(data_dict["vt"], dtype=torch.float64) * 481.3711 * 1e3
    )  # m / s
    data_dict["vp"] = (
        torch.tensor(data_dict["vp"], dtype=torch.float64) * 481.3711 * 1e3
    )  # m / s
    data_dict["rho"] = (
        torch.tensor(data_dict["rho"], dtype=torch.float64) * 1.6726 * 1e-13
    )  # kg / m^3
    data_dict["p"] = (
        torch.tensor(data_dict["p"], dtype=torch.float64) * 0.03875717
    )  # Pascals (N / m^2)
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


def detailed_residual_metrics(lhs, rhs, residual, mask=None):

    lhs_r, lhs_t, lhs_p = lhs
    rhs_r, rhs_t, rhs_p = rhs
    Rr, Rt, Rp = residual

    if mask is not None:
        lhs_r = lhs_r[mask]
        lhs_t = lhs_t[mask]
        lhs_p = lhs_p[mask]
        rhs_r = rhs_r[mask]
        rhs_t = rhs_t[mask]
        rhs_p = rhs_p[mask]
        Rr = Rr[mask]
        Rt = Rt[mask]
        Rp = Rp[mask]

    mag_lhs = torch.sqrt(lhs_r**2 + lhs_t**2 + lhs_p**2)
    mag_rhs = torch.sqrt(rhs_r**2 + rhs_t**2 + rhs_p**2)
    mag_R = torch.sqrt(Rr**2 + Rt**2 + Rp**2)

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
            slice_R = slice_R[mask[i]]

        profile.append(torch.sqrt(torch.mean(slice_R**2)).item())

    return profile


def plot_radial_profile(profile):
    plt.figure()
    plt.plot(profile)
    plt.xlabel("Radial Index")
    plt.ylabel("RMS Residual")
    plt.title("Residual RMS vs Radius")
    plt.show()


def make_residual_gif(
    output_filename, Rr, Rt, Rp, mask, fps=8, cmap="coolwarm", scale="global"
):

    Rr = Rr.detach().cpu()
    Rt = Rt.detach().cpu()
    Rp = Rp.detach().cpu()
    mask = mask.detach().cpu()

    Rmag = torch.sqrt(Rr**2 + Rt**2 + Rp**2)
    Nr = Rmag.shape[0]

    if scale == "global":
        vmax = torch.max(torch.abs(Rmag[mask]))
        vmin = 0
    frames = []

    for i in trange(Nr):

        slice_R = Rmag[i]
        m2 = mask[i]

        if scale == "local":
            vmax = torch.max(torch.abs(slice_R[m2]))
            vmin = 0

        img = slice_R.clone()
        img[~m2] = float("nan")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Residual |R|, r-slice {i+1}/{Nr}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im)

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    with imageio.get_writer(output_filename, fps=fps) as writer:
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

    with imageio.get_writer(output_filename, fps=fps) as writer:
        for f in frames:
            writer.append_data(f)

    print("Saved:", output_filename)


def build_analysis_mask(residual, n_exclude_r=0, n_exclude_theta=2):

    mask = torch.ones_like(residual, dtype=torch.bool)

    mask[:, :n_exclude_theta] = False
    mask[:, -n_exclude_theta:] = False

    return mask
