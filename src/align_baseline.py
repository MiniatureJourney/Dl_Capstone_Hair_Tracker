"""
align_baseline.py
-----------------
Traditional computer-vision baseline using:
  - AKAZE feature detection (non-linear scale space, superior for fine details)
  - Brute-Force Hamming matching (AKAZE uses binary MLDB descriptors)
  - RANSAC homography estimation  (robust to outlier matches)
  - warpPerspective alignment

Also applies U-Net segmentation to both images and computes
quantitative change metrics for comparison with the DL-based approach.

Usage:
    python src/align_baseline.py \
        --baseline path/to/before.jpg \
        --query    path/to/after.jpg \
        --model    outputs/models/best_model.pth \
        --output   outputs/results/comparison_outputs/

Outputs:
    aligned_query.png          : query image warped to match baseline perspective
    comparison_result.png      : 4-panel side-by-side comparison
    change_metrics.txt         : quantitative change report
"""

import os
import sys
import argparse

import cv2
import numpy as np
from PIL import Image

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model   import UNet
from predict import preprocess_image, predict_hair_mask, compute_coverage
from utils   import overlay_mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------
#  Argument parser
# ----------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AKAZE+RANSAC baseline + U-Net comparison")
    parser.add_argument("--baseline",  type=str, required=True,
                        help="Path to baseline (BEFORE) image")
    parser.add_argument("--query",     type=str, required=True,
                        help="Path to query (AFTER) image")
    parser.add_argument("--model",     type=str,
                        default="outputs/models/best_model.pth")
    parser.add_argument("--output",    type=str,
                        default="outputs/results/comparison_outputs/")
    parser.add_argument("--size",      type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


# ----------------------------------------------
#  AKAZE + RANSAC alignment
# ----------------------------------------------

def akaze_ransac_align(baseline_img: np.ndarray,
                       query_img:    np.ndarray,
                       size: int = 256):
    """
    Aligns query_img to baseline_img using AKAZE feature matching + RANSAC.

    Args:
        baseline_img : (H, W, 3) uint8 BGR image (OpenCV format)
        query_img    : (H, W, 3) uint8 BGR image (OpenCV format)
        size         : resize both images to (size ? size) first

    Returns:
        aligned_query : (H, W, 3) uint8 BGR warped image
        n_matches     : number of good AKAZE matches found
        homography_ok : True if homography was successfully estimated
    """
    # Resize
    h, w = size, size
    base = cv2.resize(baseline_img, (w, h))
    qry  = cv2.resize(query_img,    (w, h))

    # Convert to grayscale for feature detection
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    qry_gray  = cv2.cvtColor(qry,  cv2.COLOR_BGR2GRAY)

    # -- AKAZE detection -----------------------
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,      # full size
        descriptor_channels=3,
        threshold=1e-4,         # lower = more keypoints
        nOctaves=4,
        nOctaveLayers=4,
    )
    kp1, des1 = akaze.detectAndCompute(base_gray, None)
    kp2, des2 = akaze.detectAndCompute(qry_gray,  None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("  WARN  AKAZE: insufficient keypoints ? returning unaligned query")
        return qry, 0, False

    # -- Brute-Force Hamming matching ----------
    matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches  = matcher.match(des1, des2)
    matches  = sorted(matches, key=lambda x: x.distance)

    # Keep top 15% of matches (filter outliers)
    n_keep = max(10, int(len(matches) * 0.15))
    good   = matches[:n_keep]

    if len(good) < 8:
        print(f"  WARN  AKAZE: only {len(good)} good matches ? too few for RANSAC")
        return qry, len(good), False

    # -- RANSAC Homography ---------------------
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,
                                 ransacReprojThreshold=5.0)

    if H is None:
        print("  WARN  RANSAC failed to find homography ? returning unaligned query")
        return qry, len(good), False

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  AKAZE: {len(kp1)} + {len(kp2)} keypoints | "
          f"{len(good)} matches | {inliers} inliers")

    # -- Warp query to match baseline ----------
    aligned = cv2.warpPerspective(qry, H, (w, h))
    return aligned, len(good), True


# ----------------------------------------------
#  SSIM computation (structural similarity)
# ----------------------------------------------

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes mean SSIM between two BGR images."""
    from skimage.metrics import structural_similarity as ssim
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)


# ----------------------------------------------
#  Main comparison pipeline
# ----------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*62}")
    print(f"  AKAZE+RANSAC Baseline + U-Net Comparison")
    print(f"  Baseline : {args.baseline}")
    print(f"  Query    : {args.query}")
    print(f"  Device   : {device}")
    print(f"{'='*62}\n")

    # -- Load model -----------------------------
    checkpoint = torch.load(args.model, map_location=device)
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # -- Load images ----------------------------
    base_bgr  = cv2.imread(args.baseline)
    query_bgr = cv2.imread(args.query)

    if base_bgr is None or query_bgr is None:
        raise FileNotFoundError("Could not read one or both input images.")

    # -- Step 1: AKAZE + RANSAC alignment -------
    print("[Step 1] AKAZE + RANSAC alignment...")
    aligned_bgr, n_matches, align_ok = akaze_ransac_align(
        base_bgr, query_bgr, size=args.size)

    # Save aligned image
    aligned_path = os.path.join(args.output, "aligned_query.png")
    cv2.imwrite(aligned_path, aligned_bgr)
    print(f"  Aligned query saved -> {aligned_path}")

    # -- Step 2: U-Net segmentation -------------
    print("\n[Step 2] U-Net hair segmentation...")
    # Baseline mask
    base_tensor, base_pil = preprocess_image(args.baseline, size=args.size)
    base_mask = predict_hair_mask(model, base_tensor, device, args.threshold)
    base_cov  = compute_coverage(base_mask)

    # Aligned query mask
    aligned_pil = Image.fromarray(cv2.cvtColor(
        cv2.resize(aligned_bgr, (args.size, args.size)), cv2.COLOR_BGR2RGB))
    aligned_pil.save(os.path.join(args.output, "_tmp_aligned.png"))
    aligned_tensor, _ = preprocess_image(
        os.path.join(args.output, "_tmp_aligned.png"), size=args.size)
    aligned_mask = predict_hair_mask(model, aligned_tensor, device, args.threshold)
    aligned_cov  = compute_coverage(aligned_mask)

    # -- Step 3: Change metrics -----------------
    print("\n[Step 3] Computing change metrics...")
    coverage_change = aligned_cov - base_cov
    coverage_change_pct = (coverage_change / base_cov * 100) if base_cov > 0 else 0.0

    # XOR map: pixels that changed (gained or lost hair)
    xor_map = np.logical_xor(base_mask > 0, aligned_mask > 0).astype(np.uint8)
    changed_area_pct = float(xor_map.sum() / xor_map.size * 100)

    # SSIM between original and aligned (alignment quality check)
    base_resized = cv2.resize(base_bgr, (args.size, args.size))
    ssim_score   = compute_ssim(base_resized, aligned_bgr)

    # -- Print metrics --------------------------
    direction = "^ INCREASE" if coverage_change > 0 else "v DECREASE"
    print(f"\n  {'-'*45}")
    print(f"  Baseline hair coverage  : {base_cov:.2f}%")
    print(f"  Current hair coverage   : {aligned_cov:.2f}%")
    print(f"  Coverage change         : {coverage_change:+.2f}% ({direction})")
    print(f"  % Change (relative)     : {coverage_change_pct:+.1f}%")
    print(f"  Changed region area     : {changed_area_pct:.2f}% of image")
    print(f"  AKAZE matches           : {n_matches}")
    print(f"  Alignment SSIM          : {ssim_score:.4f}")
    print(f"  {'-'*45}")

    # -- Save metrics text file -----------------
    metrics_path = os.path.join(args.output, "change_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Hair Change Analysis Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Baseline image    : {args.baseline}\n")
        f.write(f"Query image       : {args.query}\n")
        f.write(f"Baseline coverage : {base_cov:.2f}%\n")
        f.write(f"Current coverage  : {aligned_cov:.2f}%\n")
        f.write(f"Coverage change   : {coverage_change:+.2f}%\n")
        f.write(f"Relative change   : {coverage_change_pct:+.1f}%\n")
        f.write(f"Changed area      : {changed_area_pct:.2f}%\n")
        f.write(f"AKAZE matches     : {n_matches}\n")
        f.write(f"Alignment SSIM    : {ssim_score:.4f}\n")
    print(f"  Metrics saved -> {metrics_path}")

    # -- Save comparison figure -----------------
    base_rgb    = cv2.cvtColor(cv2.resize(base_bgr, (args.size, args.size)), cv2.COLOR_BGR2RGB)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    base_ovl    = overlay_mask(base_rgb,    base_mask,    color=(220, 50, 50))
    aligned_ovl = overlay_mask(aligned_rgb, aligned_mask, color=(220, 50, 50))

    # XOR visualization: red=lost hair, green=gained hair
    xor_vis = np.zeros((*xor_map.shape, 3), dtype=np.uint8)
    lost_mask   = np.logical_and(xor_map > 0, base_mask > 0)
    gained_mask = np.logical_and(xor_map > 0, aligned_mask > 0)
    xor_vis[lost_mask]   = [220, 50, 50]    # red = lost
    xor_vis[gained_mask] = [50, 200, 50]    # green = gained

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panels = [
        (base_rgb,    "Baseline Image",           ""),
        (aligned_rgb, "Aligned Query Image",      f"SSIM: {ssim_score:.3f}"),
        (xor_vis,     "Change Map\n(Red=Lost, Green=Gained)", f"Changed: {changed_area_pct:.1f}%"),
        (base_mask,   f"Baseline Hair Mask",       f"Coverage: {base_cov:.1f}%"),
        (aligned_mask,f"Current Hair Mask",        f"Coverage: {aligned_cov:.1f}%"),
        (aligned_ovl, "Current Overlay",           f"Change: {coverage_change:+.1f}%"),
    ]
    for ax, (img, title, subtitle) in zip(axes.flatten(), panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Hair Change Analysis ? AKAZE+RANSAC + U-Net Pipeline",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    cmp_path = os.path.join(args.output, "comparison_result.png")
    plt.savefig(cmp_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure -> {cmp_path}")
    print(f"\n  Pipeline complete!")


if __name__ == "__main__":
    main()
