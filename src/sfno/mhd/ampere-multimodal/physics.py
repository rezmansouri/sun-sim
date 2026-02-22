import torch


class PhysicsLoss(torch.nn.Module):
    def __init__(
        self,
        r,
        theta,
        phi,
        br_min,
        bt_min,
        bp_min,
        br_max,
        bt_max,
        bp_max,
        jr_min,
        jt_min,
        jp_min,
        jr_max,
        jt_max,
        jp_max,
        c=2.99792458e10,
    ):
        super().__init__()
        self.r = r
        self.theta = theta
        self.dr = r[1] - r[0]
        self.dtheta = theta[1] - theta[0]
        self.dphi = phi[1] - phi[0]
        self.br_min = br_min
        self.bt_min = bt_min
        self.bp_min = bp_min
        self.br_max = br_max
        self.bt_max = bt_max
        self.bp_max = bp_max
        self.jr_min = jr_min
        self.jt_min = jt_min
        self.jp_min = jp_min
        self.jr_max = jr_max
        self.jt_max = jt_max
        self.jp_max = jp_max
        self.c = c
        self.r_cube = torch.broadcast_to(
            self.r[1:, torch.newaxis, torch.newaxis], (139, 110, 128)
        )

    def forward(self, br, bt, bp, jr, jt, jp):

        ## inverse min-max
        bt = bt * (self.bt_max - self.bt_min) + self.bt_min
        bp = bp * (self.bp_max - self.bp_min) + self.bp_min
        br = br * (self.br_max - self.br_min) + self.br_min
        jr = jr * (self.jr_max - self.jr_min) + self.jr_min
        jt = jt * (self.jt_max - self.jt_min) + self.jt_min
        jp = jp * (self.jp_max - self.jp_min) + self.jp_min

        ## radius-variant
        bt = bt / self.r_cube**2
        bp = bp / self.r_cube**2
        br = br / self.r_cube**2
        jr = jr / self.r_cube**2
        jt = jt / self.r_cube**2
        jp = jp / self.r_cube**2

        ## cgs
        r = self.r * 6.96 * 1e10
        dr = self.dr * 6.96 * 1e10
        jr = jr * 0.07558
        jt = jt * 0.07558
        jp = jp * 0.07558
        br = br * 2.2068908
        bt = bt * 2.2068908
        bp = bp * 2.2068908

        R = r.view(-1, 1, 1)
        T = self.theta.view(1, -1, 1)

        sinT = torch.sin(T)
        sinT_safe = torch.clamp(sinT, min=1e-12)

        # ----- derivatives -----
        d_bp_sin_dtheta = torch.gradient(bp * sinT, spacing=self.dtheta, dim=1)[0]
        d_bt_dphi = torch.gradient(bt, spacing=self.dphi, dim=2)[0]

        d_br_dphi = torch.gradient(br, spacing=self.dphi, dim=2)[0]
        d_rbp_dr = torch.gradient(R * bp, spacing=dr, dim=0)[0]

        d_rbt_dr = torch.gradient(R * bt, spacing=dr, dim=0)[0]
        d_br_dtheta = torch.gradient(br, spacing=self.dtheta, dim=1)[0]

        # ----- curl(B) -----
        curl_r = (1.0 / (R * sinT_safe)) * (d_bp_sin_dtheta - d_bt_dphi)

        curl_t = (1.0 / R) * ((1.0 / sinT_safe) * d_br_dphi - d_rbp_dr)

        curl_p = (1.0 / R) * (d_rbt_dr - d_br_dtheta)

        # ----- RHS -----
        factor = (4.0 * torch.pi) / self.c

        rhs_r = factor * jr
        rhs_t = factor * jt
        rhs_p = factor * jp

        # ----- residual -----
        Rr = curl_r - rhs_r
        Rt = curl_t - rhs_t
        Rp = curl_p - rhs_p

        R_mag = torch.sqrt(Rr**2 + Rt**2 + Rp**2)
        R_scalar = torch.mean(R_mag)

        return R_scalar
