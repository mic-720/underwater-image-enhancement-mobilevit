"""
models/cnn_model.py

CNN Encoder-Decoder with MobileViT-style transformer bottleneck.

Architecture:
    Input (3, H, W)
      → enc1 (3→32, double conv + BN)
      → MaxPool → enc2 (32→64, double conv + BN)
      → MaxPool → MobileViT bottleneck (64)
      → Upsample + concat(skip2) → dec1 (128→64)
      → Upsample + concat(skip1) → dec2 (96→32)
      → final_conv 1×1 (32→3)
      → sigmoid(output + input)   ← residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MOBILEVIT_DIM, MOBILEVIT_DEPTH, MOBILEVIT_HEADS


# ─────────────────────────────────────────────────────────────────────────────
# MobileViT Block
# ─────────────────────────────────────────────────────────────────────────────
class MobileViTBlock(nn.Module):
    def __init__(self, dim: int, depth: int = 2, heads: int = 2, dropout: float = 0.1):
        super().__init__()

        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x
        x = self.local_conv(x)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_flat = self.transformer(x_flat)
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        x = self.proj_conv(x)
        return x + identity


# ─────────────────────────────────────────────────────────────────────────────
# Helper: double conv block with BN
# ─────────────────────────────────────────────────────────────────────────────
def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement CNN
# ─────────────────────────────────────────────────────────────────────────────
class EnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MobileViTBlock(
            dim=MOBILEVIT_DIM,
            depth=MOBILEVIT_DEPTH,
            heads=MOBILEVIT_HEADS,
        )

        self.dec1 = conv_block(64 + 64, 64)
        self.dec2 = conv_block(64 + 32, 32)

        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d1 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.final_conv(d2)
        return torch.sigmoid(out + residual)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EnhancementCNN()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
