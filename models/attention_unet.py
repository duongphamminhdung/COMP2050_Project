# [OWN WORK] Attention U-Net implementation from scratch
# Combines baseline U-Net architecture with attention gates on skip connections
# Reference concept: Oktay et al., 2018 (arXiv:1804.03999)
# The specific implementation and integration choices are my own work

import torch
import torch.nn as nn

from .unet import ConvBlock, UpConv
from .attention_modules import AttentionGate


class AttentionUNet(nn.Module):
    """Attention U-Net for binary image segmentation.

    Same structure as baseline U-Net but with attention gates on each
    skip connection. The attention gates use the decoder's gating signal
    to weight encoder features, suppressing irrelevant regions.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        num_classes: Number of output classes (default: 1 for binary).
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder (same as baseline U-Net)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Attention gates (one per skip connection level)
        # f_g = decoder channels at this level, f_l = encoder channels, f_int = f_g // 2
        self.ag4 = AttentionGate(f_g=512, f_l=512, f_int=256)
        self.ag3 = AttentionGate(f_g=256, f_l=256, f_int=128)
        self.ag2 = AttentionGate(f_g=128, f_l=128, f_int=64)
        self.ag1 = AttentionGate(f_g=64, f_l=64, f_int=32)

        # Decoder
        self.upconv4 = UpConv(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = UpConv(512, 256)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = UpConv(256, 128)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = UpConv(128, 64)
        self.dec1 = ConvBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)              # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, H/16, W/16)

        # Decoder path with attention-gated skip connections
        d4 = self.upconv4(b)                      # (B, 512, H/8, W/8)
        e4_att = self.ag4(g=d4, x=e4)             # Attention-weighted skip
        d4 = torch.cat([d4, e4_att], dim=1)       # (B, 1024, H/8, W/8)
        d4 = self.dec4(d4)                         # (B, 512, H/8, W/8)

        d3 = self.upconv3(d4)                      # (B, 256, H/4, W/4)
        e3_att = self.ag3(g=d3, x=e3)              # Attention-weighted skip
        d3 = torch.cat([d3, e3_att], dim=1)        # (B, 512, H/4, W/4)
        d3 = self.dec3(d3)                          # (B, 256, H/4, W/4)

        d2 = self.upconv2(d3)                      # (B, 128, H/2, W/2)
        e2_att = self.ag2(g=d2, x=e2)              # Attention-weighted skip
        d2 = torch.cat([d2, e2_att], dim=1)        # (B, 256, H/2, W/2)
        d2 = self.dec2(d2)                          # (B, 128, H/2, W/2)

        d1 = self.upconv1(d2)                      # (B, 64, H, W)
        e1_att = self.ag1(g=d1, x=e1)              # Attention-weighted skip
        d1 = torch.cat([d1, e1_att], dim=1)        # (B, 128, H, W)
        d1 = self.dec1(d1)                          # (B, 64, H, W)

        return self.out_conv(d1)  # (B, 1, H, W) logits

    def get_attention_maps(self):
        """Return attention coefficient maps from all attention gates.

        Returns a list of tensors, one per decoder level (from deepest to
        shallowest), each of shape (B, 1, H, W).
        """
        return [
            self.ag4.attention_coeffs,
            self.ag3.attention_coeffs,
            self.ag2.attention_coeffs,
            self.ag1.attention_coeffs,
        ]


if __name__ == "__main__":
    model = AttentionUNet(in_channels=3, num_classes=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    att_maps = model.get_attention_maps()
    for i, am in enumerate(att_maps):
        print(f"Attention gate {i+1}: {am.shape}")
