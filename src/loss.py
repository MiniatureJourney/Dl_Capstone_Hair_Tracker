"""
loss.py
-------
Combined Binary Cross-Entropy + Dice Loss for binary hair segmentation.

Why combine both?
  BCE  : Penalizes per-pixel classification errors (good for overall accuracy)
  Dice : Maximizes mask overlap (robust against class imbalance — hair vs. background)
  Combined = best of both worlds

Input:
  logits  : raw model output (N, 1, H, W) — NO sigmoid applied yet
  targets : ground truth binary mask (N, 1, H, W) — values 0.0 or 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Differentiable Dice Loss for binary segmentation.
    Applies sigmoid internally to convert logits to probabilities.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dims for efficient computation
        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.

    L_total = λ_bce * L_BCE + λ_dice * L_Dice

    Both lambdas default to 1.0 (equal weighting, standard choice for U-Net).
    """
    def __init__(self, smooth: float = 1.0,
                 lambda_bce: float = 1.0,
                 lambda_dice: float = 1.0):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.lambda_bce  = lambda_bce
        self.lambda_dice = lambda_dice

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.lambda_bce * bce_loss + self.lambda_dice * dice_loss


if __name__ == "__main__":
    # Quick sanity check
    criterion = BCEDiceLoss()
    logits  = torch.randn(2, 1, 256, 256)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    loss    = criterion(logits, targets)
    print(f"Loss value: {loss.item():.4f}  (should be around 0.8–1.5 for random)")
