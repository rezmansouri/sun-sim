import torch
import math

print("MAKE SURE EVERYTHING IS IN CGS")


def ampere_residual(br, bt, bp, jr, jt, jp, r, theta, dr, dtheta, dphi, c):

    R = r.view(-1, 1, 1)
    T = theta.view(1, -1, 1)

    sinT = torch.sin(T)
    sinT_safe = torch.clamp(sinT, min=1e-12)

    # ----- derivatives -----
    d_bp_sin_dtheta = torch.gradient(bp * sinT, spacing=dtheta, dim=1)[0]
    d_bt_dphi = torch.gradient(bt, spacing=dphi, dim=2)[0]

    d_br_dphi = torch.gradient(br, spacing=dphi, dim=2)[0]
    d_rbp_dr = torch.gradient(R * bp, spacing=dr, dim=0)[0]

    d_rbt_dr = torch.gradient(R * bt, spacing=dr, dim=0)[0]
    d_br_dtheta = torch.gradient(br, spacing=dtheta, dim=1)[0]

    # ----- curl(B) -----
    curl_r = (1.0 / (R * sinT_safe)) * (d_bp_sin_dtheta - d_bt_dphi)

    curl_t = (1.0 / R) * ((1.0 / sinT_safe) * d_br_dphi - d_rbp_dr)

    curl_p = (1.0 / R) * (d_rbt_dr - d_br_dtheta)

    # ----- RHS -----
    factor = (4.0 * torch.pi) / c

    rhs_r = factor * jr
    rhs_t = factor * jt
    rhs_p = factor * jp

    # ----- residual -----
    Rr = curl_r - rhs_r
    Rt = curl_t - rhs_t
    Rp = curl_p - rhs_p

    return {
        "lhs": (curl_r, curl_t, curl_p),
        "rhs": (rhs_r, rhs_t, rhs_p),
        "residual": (Rr, Rt, Rp),
    }
