"""
model.py
--------
U-Net architecture for binary hair segmentation.

Architecture:
  - Encoder: 4 downsampling blocks (double conv + max pool)
  - Bottleneck: deepest feature block
  - Decoder: 4 upsampling blocks (transposed conv + skip concat + double conv)
  - Output head: 1×1 conv → single channel logits (no sigmoid here)

Input  : (N, 3, H, W)   float32 RGB image, normalized
Output : (N, 1, H, W)   float32 raw logits (apply sigmoid for prediction)

Note: Batch Normalization is included for training stability.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  Basic building block: Double Conv
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive: Conv3×3 → BatchNorm → ReLU blocks.
    Standard U-Net building block.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────
#  Encoder block: DoubleConv + MaxPool
# ─────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv    = DoubleConv(in_channels, out_channels)
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)   # save for skip connection
        down = self.pool(skip)
        return skip, down


# ─────────────────────────────────────────────
#  Decoder block: TransposedConv + concat + DoubleConv
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                       kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch from odd input dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:],
                                          mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
#  Full U-Net
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net for binary hair segmentation.

    Channel progression:
      Encoder : 3 → 64 → 128 → 256 → 512
      Bottleneck:    512 → 1024
      Decoder : 1024 → 512 → 256 → 128 → 64
      Output  : 64  → 1   (raw logit)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: list = None):
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256]

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(ch, feat))
            ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (reversed features)
        self.decoders = nn.ModuleList()
        rev_feats = list(reversed(features))
        in_ch = features[-1] * 2   # bottleneck output channels
        for feat in rev_feats:
            self.decoders.append(DecoderBlock(in_ch, feat, feat))
            in_ch = feat

        # Output head: 1×1 conv → single channel logit
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Encoding path
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding path (skip connections in reverse order)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_conv(x)   # raw logits, shape (N, 1, H, W)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Returns a binary mask (0 or 1) from input image tensor.
        Applies sigmoid then thresholds.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            return (probs > threshold).float()


# ─────────────────────────────────────────────
#  Model info utility
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {count_parameters(model):,}")
    # Expected: ~31 million parameters (standard U-Net)
