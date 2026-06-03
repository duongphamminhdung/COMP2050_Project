# [OWN WORK] Attention gate module for Attention U-Net
# Based on the concept from Oktay et al., 2018 (arXiv:1804.03999)
# Implementation is original -- the specific code is my own work

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections.

    Takes a gating signal g (from the decoder, lower resolution) and a skip
    connection feature x (from the encoder, higher resolution). Computes
    attention coefficients in [0, 1] that weight x to suppress irrelevant
    features and highlight relevant ones.

    Args:
        f_g: Number of channels in the gating signal g.
        f_l: Number of channels in the skip feature x.
        f_int: Number of intermediate channels (typically f_g // 2).
    """

    def __init__(self, f_g, f_l, f_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

        # Store attention coefficients for visualization
        self.attention_coeffs = None

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder, shape (B, f_g, H, W).
               Must be upsampled to match x spatial dims before calling.
            x: Skip connection features from encoder, shape (B, f_l, H, W).

        Returns:
            Weighted skip features, shape (B, f_l, H, W).
        """
        g1 = self.W_g(g)       # (B, f_int, H, W)
        x1 = self.W_x(x)       # (B, f_int, H, W)
        psi = self.relu(g1 + x1)   # (B, f_int, H, W)
        psi = self.psi(psi)         # (B, 1, H, W) in [0, 1]

        self.attention_coeffs = psi.detach()  # Save for visualization

        return x * psi  # Element-wise attention weighting
