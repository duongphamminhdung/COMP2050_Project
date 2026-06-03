# [OWN WORK] Residual U-Net implementation from scratch
# Replaces standard ConvBlock with ResBlock (residual connections)
# Inspired by He et al., 2016 (ResNet) applied to U-Net architecture

import torch
import torch.nn as nn

from .unet import UpConv


class ResBlock(nn.Module):
    """Residual block: Conv-BN-ReLU-Conv-BN + identity shortcut."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResUNet(nn.Module):
    """Residual U-Net for binary image segmentation.

    Same encoder-decoder structure as baseline U-Net but uses residual
    blocks instead of plain double-convolution blocks. This improves
    gradient flow through the network.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        num_classes: Number of output classes (default: 1 for binary).
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder with residual blocks
        self.enc1 = ResBlock(in_channels, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.enc4 = ResBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResBlock(512, 1024)

        # Decoder
        self.upconv4 = UpConv(1024, 512)
        self.dec4 = ResBlock(1024, 512)
        self.upconv3 = UpConv(512, 256)
        self.dec3 = ResBlock(512, 256)
        self.upconv2 = UpConv(256, 128)
        self.dec2 = ResBlock(256, 128)
        self.upconv1 = UpConv(128, 64)
        self.dec1 = ResBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


if __name__ == "__main__":
    model = ResUNet(in_channels=3, num_classes=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
