import numpy as np
import torch, math


def min_max_inverse(cubes, min_, max_):
    cubes = cubes * (max_ - min_) + min_
    return cubes


def fft_gradient(f: torch.Tensor, dr: float, radial_dim: int = 0) -> torch.Tensor:
    """
    Spectral derivative ∂v/∂r for a batched tensor.

    Parameters
    ----------
    f : (batch, Nr, Nθ, Nφ)  or any layout where `radial_dim` is Nr
    dr : float
    radial_dim : which axis in `f` is the radial coordinate (default 1)

    Returns
    -------
    dv_dr : tensor with same shape as vf
    """
    Nr = f.shape[radial_dim]

    # --- FFT along the radial axis ------------------------------------------
    v_hat = torch.fft.fft(f, dim=radial_dim)  # complex spectrum

    # --- build wavenumber array k -------------------------------------------
    k_1d = 2 * math.pi * torch.fft.fftfreq(Nr, d=dr, device=f.device)  # (Nr,)
    # reshape to broadcast into v_hat (insert singleton dims)
    shape = [1] * f.ndim
    shape[radial_dim] = Nr
    k = k_1d.reshape(*shape)

    # --- multiply by i·k in spectral space -----------------------------------
    dv_hat = 1j * k * v_hat

    # --- back-transform -------------------------------------------------------
    dv_dr = torch.fft.ifft(dv_hat, dim=radial_dim).real  # discard imag noise

    return dv_dr


class PhysicalLaw:

    # in MKS units (but with km instead of m)
    def __init__(
        self,
        r,
        v_min,
        v_max,
        rho_min,
        rho_max,
        G=6.6743e-20,
        sun_mass=1.989e30,
        v_constant=481.3711,
        rho_constant=1.6726e-13,
        p_constant=3.875717e-2,
        r_constant=697_500,
    ):
        self.dr = float(r[1] - r[0]) * r_constant
        self.v_min = v_min
        self.v_max = v_max
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.v_constant = v_constant
        self.rho_constant = rho_constant
        self.p_constant = p_constant
        g = G * sun_mass / r**2
        self.g = np.broadcast_to(g[:, np.newaxis, np.newaxis], (140, 111, 128))

    def forward(self, v_pred, rho_pred, p, v_noise=False, rho_noise=False):

        v_pred = min_max_inverse(v_pred, self.v_min, self.v_max)

        rho_pred = min_max_inverse(rho_pred, self.rho_min, self.rho_max)

        v_pred = v_pred * self.v_constant

        rho_pred = rho_pred * self.rho_constant

        if v_noise:
            v_noise = np.random.normal(
                torch.mean(v_pred), torch.std(v_pred), v_pred.shape
            )
            v_pred = v_pred + v_noise
        if rho_noise:
            rho_noise = np.random.normal(
                torch.mean(rho_pred), torch.std(rho_pred), rho_pred.shape
            )
            rho_pred = rho_pred + rho_noise

        p = p * self.p_constant

        radial_dim = 1 if v_pred.ndim == 4 else 0

        dv_r_dr = fft_gradient(v_pred, self.dr, radial_dim=radial_dim)
        # First derivative of v_r (radial velocity) w.r.t. r (along axis 0)
        d_p_dr = fft_gradient(p, self.dr, radial_dim=radial_dim)
        # First derivative of pressure w.r.t. r (1D array)

        term1 = rho_pred * v_pred * dv_r_dr  # Convective term: rho * v_r * (dv_r / dr)
        term2 = -d_p_dr  # Pressure gradient term: - dp / dr
        term3 = rho_pred * self.g  # Gravitational term: rho * g

        value = term2 + term3 - term1

        return value, v_pred, rho_pred
