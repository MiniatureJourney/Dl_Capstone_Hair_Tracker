"""
dataset.py
----------
PyTorch Dataset class for CelebAMask-HQ binary hair segmentation.
Loads preprocessed 256×256 images and binary masks.
Applies Albumentations augmentations during training.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
#  Augmentation pipelines
# ─────────────────────────────────────────────

def get_train_transforms(size: int = 256) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Affine(
            rotate=(-15, 15),
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            p=0.4
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_val_transforms(size: int = 256) -> A.Compose:
    """No augmentation for validation/test — only normalize."""
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────

class HairSegDataset(Dataset):
    """
    Dataset for binary hair segmentation.

    Args:
        split_file     (str): Path to split .txt file (train/val/test.txt)
        processed_dir  (str): Root of data/processed/
        transform      (A.Compose): Albumentations transform pipeline
    """

    def __init__(self, split_file: str, processed_dir: str,
                 transform: A.Compose = None):
        self.img_dir  = os.path.join(processed_dir, "images")
        self.mask_dir = os.path.join(processed_dir, "masks")
        self.transform = transform

        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        if len(self.ids) == 0:
            raise ValueError(f"No sample IDs found in {split_file}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]

        img_path  = os.path.join(self.img_dir,  f"{img_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")

        # Load image as RGB numpy array (H, W, 3)
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask as binary float32 (H, W) — values 0.0 or 1.0
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # binarize

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # (3, H, W) float tensor
            mask  = augmented["mask"]    # (H, W) float tensor

        # Add channel dim to mask → (1, H, W)
        mask = mask.unsqueeze(0)

        return image, mask, img_id


# ─────────────────────────────────────────────
#  Factory function
# ─────────────────────────────────────────────

def build_dataloaders(processed_dir: str, splits_dir: str,
                      batch_size: int = 8, num_workers: int = 4,
                      size: int = 256):
    """
    Builds train, val, and test DataLoaders.
    Returns a dict: {"train": ..., "val": ..., "test": ...}
    """
    from torch.utils.data import DataLoader

    loaders = {}
    configs = [
        ("train", "train.txt", get_train_transforms(size), True),
        ("val",   "val.txt",   get_val_transforms(size),   False),
        ("test",  "test.txt",  get_val_transforms(size),   False),
    ]

    for split, fname, transform, shuffle in configs:
        split_file = os.path.join(splits_dir, fname)
        dataset = HairSegDataset(split_file, processed_dir, transform)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        loaders[split] = loader
        print(f"  {split:<6}: {len(dataset):>6,} samples | {len(loader):>4} batches")

    return loaders
