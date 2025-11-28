from neuralop.models import SFNO


import torch
import torch.nn as nn

class ComponentAttention(nn.Module):
    def __init__(self, n_comp, d_hidden, n_heads=4):
        super().__init__()
        self.n_comp = n_comp
        self.d_hidden = d_hidden
        self.attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)

    def forward(self, comps):
        """
        comps: list of (B, d_hidden, H, W)
        return: same shapes after attention
        """
        B, C, H, W = comps[0].shape

        x = torch.stack(comps, dim=1)  # (B, n_comp, d_hidden, H, W)
        x = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, self.n_comp, C)

        out, _ = self.attn(x, x, x)

        out = out.reshape(B, H, W, self.n_comp, C).permute(0, 3, 4, 1, 2)
        return list(out)



class MultiModalSFNO(nn.Module):
    def __init__(
        self,
        in_comp=1,
        out_comp=1,
        n_radii=139,
        d_hidden=64,
        sfno=None,
        use_attention=True,     # <---- add switch
        n_heads=4
    ):
        super().__init__()
        self.in_comp = in_comp
        self.out_comp = out_comp
        self.n_radii = n_radii
        self.d_hidden = d_hidden
        self.use_attention = use_attention   # <---- store it

        # 1) Per-component encoders
        self.encoders = nn.ModuleList([
            nn.Conv1d(1, d_hidden, kernel_size=1) for _ in range(in_comp)
        ])

        # 2) Cross-component attention block (may be bypassed)
        self.comp_attn = ComponentAttention(in_comp, d_hidden, n_heads)

        # 3) Shared SFNO trunk
        self.sfno = sfno

        # 4) Per-component output heads
        self.heads = nn.ModuleList([
            nn.Conv1d(d_hidden, n_radii, kernel_size=1) for _ in range(out_comp)
        ])

    def forward(self, comps):
        B, _, H, W = comps[0].shape

        # --- Encode each component ---
        encoded = []
        for x_i, enc in zip(comps, self.encoders):
            x_i = x_i.reshape(B, 1, H * W)
            z_i = enc(x_i)
            z_i = z_i.view(B, self.d_hidden, H, W)
            encoded.append(z_i)

        # --- Optional cross-component attention ---
        if self.use_attention:
            encoded = self.comp_attn(encoded)

        # --- Concatenate channels ---
        latent = torch.cat(encoded, dim=1)  # (B, in_comp*d_hidden, H, W)

        # --- SFNO trunk ---
        z = self.sfno(latent)

        # --- Split back into per-component latent chunks ---
        z_split = torch.split(z, self.d_hidden, dim=1)

        # --- Decode each component ---
        outputs = []
        for z_i, head in zip(z_split, self.heads):
            z_i_flat = z_i.view(B, self.d_hidden, H * W)
            o_i = head(z_i_flat)
            o_i = o_i.view(B, self.n_radii, H, W)
            outputs.append(o_i)

        return outputs
