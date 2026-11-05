# 📂 CelebAMask-HQ Dataset Setup Instructions
## Deep Learning Capstone — Hair Progress Tracker (Group 4, V-C)

This document provides instructions for downloading, organizing, and preprocessing the **CelebAMask-HQ** dataset for training and reproducing the **U-Net Hair Segmentation** pipeline.

---

## 1. Dataset Overview

* **Dataset Name**: CelebAMask-HQ
* **Original Paper**: *MaskGAN: Towards Diverse and Interactive Facial Image Manipulation* (Lee et al., CVPR 2020)
* **Total Images**: 30,000 high-resolution portrait photographs ($1024 \times 1024$ pixels)
* **Annotation Classes**: 19 semantic categories (including `hair`, `skin`, `l_brow`, `r_brow`, `l_eye`, `r_eye`, `nose`, `cloth`, etc.)
* **Hair Task**: Binary semantic segmentation separating cranial and facial hair from background, facial skin, clothing, and accessories.

---

## 2. Download Sources

The CelebAMask-HQ dataset can be acquired through either of the following official mirrors:

1. **Kaggle Mirror (Direct Fast Download)**:
   * URL: [https://www.kaggle.com/datasets/oslandlibrary/celebamaskhq](https://www.kaggle.com/datasets/oslandlibrary/celebamaskhq)
   * Archive size: ~3.2 GB compressed
2. **Official GitHub Repository (Google Drive / Baidu)**:
   * URL: [https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
   * Follow the Google Drive links provided in the official repository README.

---

## 3. Directory Layout

Extract the downloaded archives into the `data/raw/` directory according to this exact folder structure:

```
data/
├── raw/
│   ├── CelebA-HQ-img/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   └── 29999.jpg
│   ├── CelebAMask-HQ-mask-anno/
│   │   ├── 0/
│   │   │   ├── 00000_hair.png
│   │   │   ├── ...
│   │   ├── 1/
│   │   ├── ...
│   │   └── 14/
│   ├── CelebA-HQ-to-CelebA-mapping.txt
│   └── CelebAMask-HQ-pose-anno.txt
├── processed/              <-- Generated automatically by src/preprocess.py
│   ├── images/             # 256x256 RGB JPEG/PNG
│   └── masks/              # 256x256 binary single-channel masks (0 or 255)
└── splits/                 <-- Pre-computed and included in the repository
    ├── train.txt           # 24,000 training samples (80%)
    ├── val.txt             # 3,000 validation samples (10%)
    └── test.txt            # 3,000 testing samples (10%)
```

---

## 4. Preprocessing & Mask Extraction

CelebAMask-HQ stores individual category masks in separate subfolders. The script `src/preprocess.py` parses these folder hierarchies, aggregates all hair annotations, resizes images and masks to $256 \times 256$ pixels, and formats them for high-throughput training:

```powershell
python src/preprocess.py `
  --img_dir data/raw/CelebA-HQ-img `
  --mask_dir data/raw/CelebAMask-HQ-mask-anno `
  --out_dir data/processed `
  --size 256
```

### Preprocessing Parameters:
* `--img_dir`: Path to the raw 1024x1024 face images.
* `--mask_dir`: Path to the 15 annotation folders (`0/` to `14/`).
* `--out_dir`: Destination directory for processed images and merged masks.
* `--size`: Target resolution (default: `256` for $256 \times 256$ pixels).

---

## 5. Dataset Partitioning (80 / 10 / 10 Split)

To ensure reproducibility across all experiments, deterministic train/val/test splits are saved in `data/splits/`.
If you ever need to re-generate the partition text files:

```powershell
python src/split_data.py `
  --processed_dir data/processed `
  --splits_dir data/splits `
  --seed 42 `
  --train_ratio 0.8 `
  --val_ratio 0.1 `
  --test_ratio 0.1
```

* **Training Set**: 24,000 images (80%)
* **Validation Set**: 3,000 images (10%)
* **Test Evaluation Set**: 3,000 images (10%)

---

## 6. Pre-Trained Model Weights

If you only wish to run inference, the desktop application, or the web dashboard without retraining on 30,000 images:
* Pre-trained weights are pre-packaged at:
  * `outputs/models/best_model.pth` (or root `best_model.pth`)
* Cloud mirror: [Hugging Face Model Hub](https://huggingface.co/AlgomCorp/hair-segmentation-unet)
