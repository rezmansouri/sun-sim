from neuralop.models import SFNO


import torch
import torch.nn as nn


class MultiModalSFNO(nn.Module):
    def __init__(self, in_comp=1, out_comp=1, n_radii=139, d_hidden=64, sfno=None):
        super().__init__()
        self.in_comp = in_comp
        self.out_comp = out_comp
        self.n_radii = n_radii
        self.d_hidden = d_hidden

        # One encoder per input component (maps 1→d_hidden)
        self.encoders = nn.ModuleList(
            [nn.Conv1d(1, d_hidden, kernel_size=1) for _ in range(in_comp)]
        )

        # Shared SFNO trunk (operates on concatenated components)
        assert sfno is not None, "Pass an initialized SFNO module"
        # Expect: SFNO(in_channels = in_comp * d_hidden, out_channels = in_comp * d_hidden)
        self.sfno = sfno

        # One decoder head per output component (maps its latent→n_radii)
        self.heads = nn.ModuleList(
            [nn.Conv1d(d_hidden, n_radii, kernel_size=1) for _ in range(out_comp)]
        )

    def forward(self, comps):
        """
        comps: list of tensors, each (B, 1, H, W)
        returns: (B, out_comp, n_radii, H, W)
        """
        assert (
            len(comps) == self.in_comp
        ), f"Expected {self.in_comp} input components, got {len(comps)}"
        B, _, H, W = comps[0].shape
        
        # for comp in comps:
        #     print(comp.shape)

        # --- Encode each component separately ---
        encoded = []
        for x_i, enc in zip(comps, self.encoders):
            x_i = x_i.view(B, 1, H * W)
            z_i = enc(x_i)  # (B, d_hidden, H*W)
            z_i = z_i.view(B, self.d_hidden, H, W)
            encoded.append(z_i)

        # --- Concatenate along channel dimension (no merging) ---
        # shape: (B, in_comp*d_hidden, H, W)
        latent = torch.cat(encoded, dim=1)

        # --- Shared SFNO trunk (one forward pass for all components) ---
        z = self.sfno(latent)  # (B, in_comp*d_hidden, H, W)

        # --- Split latent back per component ---
        z_split = torch.split(z, self.d_hidden, dim=1)

        # --- Decode each component separately ---
        outputs = []
        for z_i, head in zip(z_split, self.heads):
            z_flat = z_i.view(B, self.d_hidden, H * W)
            o_i = head(z_flat)  # (B, n_radii, H*W)
            o_i = o_i.view(B, self.n_radii, H, W)
            outputs.append(o_i)

        # # Stack to (B, out_comp, n_radii, H, W)
        # out = torch.stack(outputs, dim=1)
        # return out
        return outputs
