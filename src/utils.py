"""
utils.py
--------
Shared utility functions used across the project:
  - save_sample_predictions : saves a grid of input|GT|pred images
  - plot_curves             : saves training loss & metric curves
  - overlay_mask            : blends a binary mask over an image (for GUI/reports)
  - denormalize             : reverses ImageNet normalization for display
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from PIL import Image


# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized image tensor (C, H, W) back to uint8 numpy array (H, W, C).
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * STD + MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color: tuple = (255, 50, 50), alpha: float = 0.45) -> np.ndarray:
    """
    Overlays a binary mask on an RGB image.

    Args:
        image : (H, W, 3) uint8 RGB image
        mask  : (H, W) binary array (0 or 1)
        color : RGB tuple for the overlay color
        alpha : transparency (0=invisible, 1=opaque)

    Returns:
        (H, W, 3) uint8 blended image
    """
    overlay = image.copy().astype(float)
    color_arr = np.array(color, dtype=float)

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask > 0,
            overlay[:, :, c] * (1 - alpha) + color_arr[c] * alpha,
            overlay[:, :, c]
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


@torch.no_grad()
def save_sample_predictions(model, loader, device: torch.device,
                             save_dir: str, epoch: int, n_samples: int = 6):
    """
    Saves a figure with n_samples rows, each showing:
      Column 1: Input image
      Column 2: Ground truth mask
      Column 3: Predicted mask
      Column 4: Overlay (pred mask on image)
    """
    model.eval()
    images_list, masks_list, preds_list = [], [], []

    for images, masks, _ in loader:
        images = images.to(device)
        logits = model(images)
        preds  = (torch.sigmoid(logits) > 0.5).float()

        images_list.append(images.cpu())
        masks_list.append(masks.cpu())
        preds_list.append(preds.cpu())

        if sum(x.shape[0] for x in images_list) >= n_samples:
            break

    images_all = torch.cat(images_list)[:n_samples]
    masks_all  = torch.cat(masks_list)[:n_samples]
    preds_all  = torch.cat(preds_list)[:n_samples]

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, n_samples * 4))
    fig.suptitle(f"Sample Predictions ? Epoch {epoch}", fontsize=14, y=1.01)

    col_titles = ["Input Image", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for row in range(n_samples):
        img  = denormalize(images_all[row])
        gt   = masks_all[row][0].numpy()
        pred = preds_all[row][0].numpy()
        ovl  = overlay_mask(img, pred)

        axes[row][0].imshow(img)
        axes[row][1].imshow(gt,   cmap="gray")
        axes[row][2].imshow(pred, cmap="gray")
        axes[row][3].imshow(ovl)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history: dict, save_dir: str):
    """
    Plots and saves:
      1. loss_curve.png     : train loss & val loss
      2. metric_curves.png  : Dice, IoU, Precision, Recall
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # -- Loss curve -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCE + Dice)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # -- Metric curves --------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_info = [
        ("val_dice",      "Dice Score",  "#2ca02c"),
        ("val_iou",       "IoU Score",   "#9467bd"),
        ("val_precision", "Precision",   "#d62728"),
        ("val_recall",    "Recall",      "#8c564b"),
    ]
    for ax, (key, label, color) in zip(axes.flatten(), metrics_info):
        if key in history:
            ax.plot(epochs, history[key], label=label, color=color, linewidth=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=12)
            ax.set_ylim(0, 1)
            ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="Target 0.85")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Validation Metrics Over Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metric_curves.png"), dpi=150)
    plt.close(fig)

    print(f"  ? Plots saved -> {save_dir}/loss_curve.png  &  metric_curves.png")
