# [OWN WORK] Baseline U-Net implementation from scratch
# Reference architecture: Ronneberger et al., 2015 (arXiv:1505.04597)

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    """Upsampling block: bilinear upsample + conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    """Baseline U-Net for binary image segmentation.

    Architecture:
        Encoder: 4 levels (64, 128, 256, 512 channels) with MaxPool downsampling
        Bottleneck: 1024 channels
        Decoder: 4 levels mirroring encoder with skip connections
        Output: 1x1 conv (logits, no sigmoid)

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        num_classes: Number of output classes (default: 1 for binary segmentation).
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.upconv4 = UpConv(1024, 512)
        self.dec4 = ConvBlock(1024, 512)  # 512 + 512 (skip) = 1024 input
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
        e1 = self.enc1(x)           # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, H/16, W/16)

        # Decoder path with skip connections
        d4 = self.upconv4(b)                    # (B, 512, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)         # (B, 1024, H/8, W/8)
        d4 = self.dec4(d4)                       # (B, 512, H/8, W/8)

        d3 = self.upconv3(d4)                    # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)          # (B, 512, H/4, W/4)
        d3 = self.dec3(d3)                        # (B, 256, H/4, W/4)

        d2 = self.upconv2(d3)                    # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)          # (B, 256, H/2, W/2)
        d2 = self.dec2(d2)                        # (B, 128, H/2, W/2)

        d1 = self.upconv1(d2)                    # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)          # (B, 128, H, W)
        d1 = self.dec1(d1)                        # (B, 64, H, W)

        return self.out_conv(d1)  # (B, 1, H, W) logits


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
