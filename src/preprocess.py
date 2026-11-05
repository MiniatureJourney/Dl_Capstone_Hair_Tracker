"""
preprocess.py
-------------
Extracts binary hair masks from CelebAMask-HQ dataset annotations
and resizes images + masks to a standard size for U-Net training.

CelebAMask-HQ structure:
  CelebA-HQ-img/          -> 0.jpg, 1.jpg, ..., 29999.jpg
  CelebAMask-HQ-mask-anno/
      0/                  -> 00000_hair.png, 00000_skin.png, ...
      1/                  -> 02000_hair.png, ...
      ...
      14/                 -> 28000_hair.png, ...

Mask folder for image index i  = i // 2000
Hair mask filename              = f"{i:05d}_hair.png"

Usage:
    python src/preprocess.py \
        --img_dir  data/raw/CelebA-HQ-img \
        --mask_dir data/raw/CelebAMask-HQ-mask-anno \
        --out_dir  data/processed \
        --size     256
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CelebAMask-HQ for hair segmentation")
    parser.add_argument("--img_dir",  type=str,
                        default=os.path.join("data", "raw", "CelebA-HQ-img"),
                        help="Path to CelebA-HQ-img folder")
    parser.add_argument("--mask_dir", type=str,
                        default=os.path.join("data", "raw", "CelebAMask-HQ-mask-anno"),
                        help="Path to CelebAMask-HQ-mask-anno folder")
    parser.add_argument("--out_dir",  type=str,
                        default=os.path.join("data", "processed"),
                        help="Output directory for processed images and masks")
    parser.add_argument("--size",     type=int, default=256,
                        help="Resize target (square). Default: 256")
    return parser.parse_args()


def get_hair_mask_path(mask_dir: str, img_idx: int) -> str:
    """Returns the expected path for the hair mask of a given image index."""
    folder   = img_idx // 2000
    filename = f"{img_idx:05d}_hair.png"
    return os.path.join(mask_dir, str(folder), filename)


def process_single(img_path: str, mask_path: str,
                   out_img_path: str, out_mask_path: str,
                   size: int) -> bool:
    """
    Processes one image-mask pair.
    Returns True on success, False on failure.
    """
    try:
        # --- Image ---
        img = Image.open(img_path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(out_img_path, format="PNG")

        # --- Mask ---
        mask     = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask)
        binary   = (mask_arr > 0).astype(np.uint8) * 255
        mask_out = Image.fromarray(binary)
        mask_out = mask_out.resize((size, size), Image.NEAREST)
        mask_out.save(out_mask_path, format="PNG")
        return True
    except Exception:
        return False


def main():
    args = parse_args()

    # Resolve to absolute paths (fixes relative path issues)
    img_dir  = os.path.abspath(args.img_dir)
    mask_dir = os.path.abspath(args.mask_dir)
    out_dir  = os.path.abspath(args.out_dir)

    out_img_dir  = os.path.join(out_dir, "images")
    out_mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(out_img_dir,  exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    total             = 30000
    processed         = 0
    skipped_no_mask   = 0
    skipped_error     = 0

    print("=" * 60)
    print("  CelebAMask-HQ Preprocessing")
    print(f"  Image dir : {img_dir}")
    print(f"  Mask dir  : {mask_dir}")
    print(f"  Output    : {out_dir}")
    print(f"  Resize to : {args.size}x{args.size}")
    print("=" * 60)
    print()

    # Quick sanity check
    if not os.path.isdir(img_dir):
        print(f"ERROR: Image directory not found: {img_dir}")
        sys.exit(1)
    if not os.path.isdir(mask_dir):
        print(f"ERROR: Mask directory not found: {mask_dir}")
        sys.exit(1)

    for i in tqdm(range(total), desc="Processing", unit="img"):
        img_path  = os.path.join(img_dir, f"{i}.jpg")
        mask_path = get_hair_mask_path(mask_dir, i)

        if not os.path.exists(img_path):
            skipped_error += 1
            continue

        if not os.path.exists(mask_path):
            skipped_no_mask += 1
            continue

        out_img_path  = os.path.join(out_img_dir,  f"{i:05d}.png")
        out_mask_path = os.path.join(out_mask_dir, f"{i:05d}.png")

        success = process_single(
            img_path, mask_path, out_img_path, out_mask_path, args.size)
        if success:
            processed += 1
        else:
            skipped_error += 1

    # Save list of valid processed image IDs
    valid_ids_path = os.path.join(out_dir, "valid_ids.txt")
    all_valid = [
        f"{i:05d}" for i in range(total)
        if os.path.exists(os.path.join(out_img_dir, f"{i:05d}.png"))
    ]
    with open(valid_ids_path, "w") as f:
        f.write("\n".join(all_valid))

    print()
    print("=" * 60)
    print(f"  Processed successfully : {processed:,}")
    print(f"  Skipped (no hair mask) : {skipped_no_mask:,}")
    print(f"  Skipped (errors)       : {skipped_error:,}")
    print(f"  Valid IDs saved to     : {valid_ids_path}")
    print("=" * 60)
    print()
    print("Next step: python src/split_data.py")


if __name__ == "__main__":
    main()
