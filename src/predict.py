"""
predict.py
----------
Single-image inference utility.
Loads the trained U-Net model and runs hair segmentation on any image.

Usage:
    python src/predict.py \
        --image  path/to/image.jpg \
        --model  outputs/models/best_model.pth \
        --output outputs/results/comparison_outputs/prediction.png

Output: side-by-side PNG  [Input | Hair Mask | Overlay]
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import UNet
from utils import denormalize, overlay_mask

# ImageNet normalization constants
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image hair segmentation inference")
    parser.add_argument("--image",     type=str, required=True, help="Path to input image")
    parser.add_argument("--model",     type=str, default="outputs/models/best_model.pth")
    parser.add_argument("--output",    type=str, default=None,
                        help="Output path for result PNG (optional)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--size",      type=int, default=256)
    parser.add_argument("--show",      action="store_true",
                        help="Display result (requires GUI)")
    return parser.parse_args()


def preprocess_image(image_path: str, size: int = 256) -> torch.Tensor:
    """
    Loads an image, resizes to (size × size), normalizes with ImageNet stats.
    Returns a (1, 3, H, W) float tensor ready for the model.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0

    # Normalize
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()  # (1,3,H,W)
    return tensor, img


@torch.no_grad()
def predict_hair_mask(model: torch.nn.Module, image_tensor: torch.Tensor,
                      device: torch.device, threshold: float = 0.5) -> np.ndarray:
    """
    Runs the model and returns a binary mask as a numpy array (H, W) with values 0 or 1.
    """
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs  = torch.sigmoid(logits)
    mask   = (probs > threshold).float().squeeze().cpu().numpy()
    return mask


def compute_coverage(mask: np.ndarray) -> float:
    """Returns the percentage of pixels classified as hair."""
    return float(mask.sum() / mask.size * 100)


def save_result(original_img: Image.Image, mask: np.ndarray, output_path: str):
    """
    Saves a 3-panel PNG: Input | Binary Mask | Overlay
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_arr = np.array(original_img)
    ovl      = overlay_mask(orig_arr, mask, color=(220, 50, 50), alpha=0.45)
    coverage = compute_coverage(mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_arr);          axes[0].set_title("Input Image")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title(f"Hair Mask\nCoverage: {coverage:.1f}%")
    axes[2].imshow(ovl);               axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Preprocess & predict
    image_tensor, pil_img = preprocess_image(args.image, size=args.size)
    mask     = predict_hair_mask(model, image_tensor, device, args.threshold)
    coverage = compute_coverage(mask)

    print(f"\n  Image    : {args.image}")
    print(f"  Coverage : {coverage:.2f}% of image is hair")

    # Save or display
    if args.output:
        save_result(pil_img, mask, args.output)
    else:
        # Auto-generate output path next to input image
        base   = os.path.splitext(args.image)[0]
        output = base + "_hair_prediction.png"
        save_result(pil_img, mask, output)

    return mask, coverage


if __name__ == "__main__":
    main()
