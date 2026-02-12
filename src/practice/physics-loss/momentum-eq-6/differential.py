import torch
import math

print("MAKE SURE EVERYTHING IS IN CGS")


FOUR_PI = 4.0 * math.pi


def build_geometry(r, theta):
    """
    r: (Nr,)
    theta: (Nth,)
    """
    R = r.view(-1, 1, 1)
    T = theta.view(1, -1, 1)

    sinT = torch.sin(T)
    cosT = torch.cos(T)

    sinT_safe = sinT.clone()
    sinT_safe[sinT_safe.abs() < 1e-8] = 1e-8

    return R, sinT, sinT_safe, cosT


def d_dr(f, dr):
    return torch.gradient(f, spacing=dr, dim=0)[0]


def d_dtheta(f, dtheta):
    return torch.gradient(f, spacing=dtheta, dim=1)[0]


def d_dphi_periodic(f, dphi):
    return (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2.0 * dphi)


def build_mhd_stress_tensor(rho, vr, vt, vp, br, bt, bp, p):

    B2 = br**2 + bt**2 + bp**2

    # Isotropic pressure + magnetic pressure
    p_tot = p + B2 / (8.0 * math.pi)

    # Maxwell anisotropic part
    fac = 1.0 / (4.0 * math.pi)

    Trr = rho * vr * vr + p_tot - fac * br * br
    Trt = rho * vr * vt - fac * br * bt
    Trp = rho * vr * vp - fac * br * bp

    Ttr = Trt
    Ttt = rho * vt * vt + p_tot - fac * bt * bt
    Ttp = rho * vt * vp - fac * bt * bp

    Tpr = Trp
    Tpt = Ttp
    Tpp = rho * vp * vp + p_tot - fac * bp * bp

    return {
        "rr": Trr,
        "rt": Trt,
        "rp": Trp,
        "tr": Ttr,
        "tt": Ttt,
        "tp": Ttp,
        "pr": Tpr,
        "pt": Tpt,
        "pp": Tpp,
    }


def divergence_tensor(T, dr, dtheta, dphi, R, sinT, sinT_safe):

    # Shortcuts
    Trr, Trt, Trp = T["rr"], T["rt"], T["rp"]
    Ttr, Ttt, Ttp = T["tr"], T["tt"], T["tp"]
    Tpr, Tpt, Tpp = T["pr"], T["pt"], T["pp"]

    # ----------------------
    # Radial component
    # ----------------------
    term_r1 = d_dr(R**2 * Trr, dr) / (R**2)
    term_r2 = d_dtheta(sinT * Trt, dtheta) / (R * sinT_safe)
    term_r3 = d_dphi_periodic(Trp, dphi) / (R * sinT_safe)

    geom_r = -(Ttt + Tpp) / R

    div_r = term_r1 + term_r2 + term_r3 + geom_r

    # ----------------------
    # Theta component
    # ----------------------
    term_t1 = d_dr(R**2 * Ttr, dr) / (R**2)
    term_t2 = d_dtheta(sinT * Ttt, dtheta) / (R * sinT_safe)
    term_t3 = d_dphi_periodic(Ttp, dphi) / (R * sinT_safe)

    geom_t = (Ttr - Tpp * torch.cos(torch.acos(sinT))) / R

    div_t = term_t1 + term_t2 + term_t3 + geom_t

    # ----------------------
    # Phi component
    # ----------------------
    term_p1 = d_dr(R**2 * Tpr, dr) / (R**2)
    term_p2 = d_dtheta(sinT * Tpt, dtheta) / (R * sinT_safe)
    term_p3 = d_dphi_periodic(Tpp, dphi) / (R * sinT_safe)

    geom_p = (Tpr + Tpt * torch.cos(torch.acos(sinT))) / R

    div_p = term_p1 + term_p2 + term_p3 + geom_p

    return div_r, div_t, div_p


def mas_momentum_residual_conservative(
    rho,
    vr,
    vt,
    vp,
    br,
    bt,
    bp,
    p,
    r,
    theta,
    dr,
    dtheta,
    dphi,
    G,
    M,
    rho_old=None,
    vr_old=None,
    vt_old=None,
    vp_old=None,
    dt=None,
):

    R, sinT, sinT_safe, _ = build_geometry(r, theta)

    # ----------------------
    # Time derivative
    # ----------------------
    if rho_old is not None:
        dt_r = (rho * vr - rho_old * vr_old) / dt
        dt_t = (rho * vt - rho_old * vt_old) / dt
        dt_p = (rho * vp - rho_old * vp_old) / dt
    else:
        dt_r = torch.zeros_like(vr)
        dt_t = torch.zeros_like(vt)
        dt_p = torch.zeros_like(vp)

    # ----------------------
    # Total MHD stress tensor
    # ----------------------
    T = build_mhd_stress_tensor(rho, vr, vt, vp, br, bt, bp, p)

    div_r, div_t, div_p = divergence_tensor(T, dr, dtheta, dphi, R, sinT, sinT_safe)

    # ----------------------
    # Gravity
    # ----------------------
    g_r = -rho * G * M / (R**2)
    g_t = torch.zeros_like(g_r)
    g_p = torch.zeros_like(g_r)

    # ----------------------
    # Residual
    # ----------------------
    Rr = dt_r + div_r - g_r
    Rt = dt_t + div_t - g_t
    Rp = dt_p + div_p - g_p

    return {
        "lhs": (dt_r + div_r, dt_t + div_t, dt_p + div_p),
        "rhs": (g_r, g_t, g_p),
        "residual": (Rr, Rt, Rp),
    }
