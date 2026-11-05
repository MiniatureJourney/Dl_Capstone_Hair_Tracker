"""
pipeline_visualizer.py
----------------------
Interactive Model Training Pipeline & Architecture Visualizer for DL Capstone.

Features:
  - Dynamic interactive slider across all 8 training pipeline stages
  - Clickable, interactive custom canvas flowchart with glowing active stages
  - Interactive stepper controls (Previous, Next, Auto-Play Tour)
  - Detailed technical deep-dive per stage:
      * Raw data acquisition (CelebAMask-HQ 30k)
      * Data cleaning & preprocessing (700 bald filtered -> 29,300 @ 256x256)
      * Stratified 80/10/10 split (23,440 / 2,930 / 2,930) & DataLoader (batch=16)
      * Data augmentation pipeline (Albumentations transforms + ImageNet norm)
      * Deep U-Net architecture (7.76M parameters, DoubleConv, skip connections)
      * Hybrid loss formulation (BCEWithLogits + Soft Dice)
      * Optimization & training convergence (Adam, ReduceLROnPlateau, RTX 5060, best ep 15)
      * Final evaluation benchmark (Dice 95.19%, IoU 90.84%) & Hair Tracker inference
"""

import sys
import os
import tkinter as tk
import customtkinter as ctk

# ──────────────────────────────────────────────────────────────
#  Stage Definitions & Technical Specifications
# ──────────────────────────────────────────────────────────────

STAGES = [
    {
        "id": 1,
        "title": "Raw Data Acquisition",
        "short_title": "1. Raw Data",
        "subtitle": "CelebAMask-HQ Origin & Multi-Class Parsing",
        "node_text": "CelebAMask-HQ Raw\n30,000 Facial Images\n(1024 x 1024 x 3 RGB)",
        "output_shape": "30,000 pairs (1024x1024)",
        "card_color": "#1e3a5f",
        "highlight_color": "#38bdf8",
        "overview": (
            "The project utilizes the CelebAMask-HQ dataset, a large-scale high-resolution "
            "face attribute database selected from CelebA-HQ. Each subject contains extensive "
            "facial variations across age, ethnicity, lighting, hair texture, length, and scalp coverage.\n\n"
            "Key Challenge: Raw annotations contain 19 individual facial components spread across "
            "15 disk folders (0/ to 14/). The hair class had to be isolated, merged across multi-part "
            "hair annotations, and mapped into a clean single-target segmentation objective."
        ),
        "specs": [
            ("Raw Dataset", "CelebAMask-HQ (High-Resolution Facial Images)"),
            ("Total Raw Images", "30,000 RGB Portrait Images"),
            ("Native Resolution", "1024 x 1024 pixels, 24-bit RGB"),
            ("Annotation Structure", "19 semantic classes across 15 partition folders"),
            ("Target Class Isolated", "'hair' category (facial & scalp hair)"),
            ("Subject Diversity", "Male & female, multi-ethnic, varying hair volume/balding")
        ],
        "math_title": "Data Selection & Class Extraction Formulation",
        "math_formula": (
            "Mask Extraction Function:\n"
            "  M_hair(x, y) = 1  if pixel (x, y) in Hair_Annotation_Set\n"
            "  M_hair(x, y) = 0  otherwise (skin, eyes, clothing, background)\n\n"
            "Original Storage Representation:\n"
            "  Image Tensor:  X_raw in [0, 255]^(1024 x 1024 x 3)\n"
            "  Binary Mask:   Y_raw in {0, 1}^(1024 x 1024)"
        ),
        "code_snippet": (
            "# Class extraction logic from CelebAMask-HQ\n"
            "for idx in range(30000):\n"
            "    folder = idx // 2000\n"
            "    mask_path = f'data/raw/CelebAMask-HQ-mask-anno/{folder}/{idx:05d}_hair.png'\n"
            "    # Read 1024x1024 single-channel hair label\n"
            "    hair_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)"
        ),
        "metrics": [
            ("Initial Dataset Size", "30,000 images"),
            ("Annotation Resolution", "1024 x 1024 px"),
            ("Target Classes", "1 (Hair vs Background)"),
            ("Class Balance", "~26.4% mean hair pixel coverage")
        ]
    },
    {
        "id": 2,
        "title": "Preprocessing & Data Cleaning",
        "short_title": "2. Preprocessing",
        "subtitle": "Dead Sample Filtering & 256x256 Spatial Standardization",
        "node_text": "Cleaning & Resizing\n700 Bald Filtered Out\n29,300 Valid (256 x 256)",
        "output_shape": "29,300 pairs (256x256)",
        "card_color": "#1e3d34",
        "highlight_color": "#34d399",
        "overview": (
            "High-resolution 1024x1024 images require immense GPU memory and contain redundant high-frequency "
            "noise. All images and masks were standardized to 256x256 pixels.\n\n"
            "Critical Data Cleaning Step: Automated verification detected 700 images with zero hair pixels "
            "(completely bald subjects or misannotations). Including non-hair images causes zero-gradient "
            "trivial local minima. Exactly 700 dead samples were eliminated, yielding 29,300 pristine pairs."
        ),
        "specs": [
            ("Filtering Outcome", "700 bald/non-hair samples removed (2.33% filtered)"),
            ("Cleaned Dataset", "29,300 verified (image, mask) pairs"),
            ("Target Spatial Grid", "256 x 256 pixels"),
            ("Image Interpolation", "cv2.INTER_LINEAR (Bilinear - smooth textures)"),
            ("Mask Interpolation", "cv2.INTER_NEAREST (Preserves crisp binary 0/1 edges)"),
            ("Storage Format", "PNG pairs in data/processed/images/ & masks/")
        ],
        "math_title": "Spatial Downsampling & Thresholding Operators",
        "math_formula": (
            "Image Downsampling (Bilinear Continuous Operator):\n"
            "  I_256(u, v) = Sum_i Sum_j w_ij * I_1024(x_i, y_j)\n\n"
            "Mask Downsampling (Nearest-Neighbor Discrete Operator):\n"
            "  M_256(u, v) = M_1024(round(u * s), round(v * s)),  s = 1024/256 = 4\n\n"
            "Binary Enforcing:\n"
            "  M_clean(u, v) = 1  if M_256(u, v) > 127  else 0\n\n"
            "Validation Condition:\n"
            "  Valid_Sample = True  if Sum(M_clean) >= 50 pixels  else Discard"
        ),
        "code_snippet": (
            "# Preprocessing pipeline execution (preprocess.py)\n"
            "img_resized = cv2.resize(img_raw, (256, 256), interpolation=cv2.INTER_LINEAR)\n"
            "mask_resized = cv2.resize(mask_raw, (256, 256), interpolation=cv2.INTER_NEAREST)\n"
            "binary_mask = (mask_resized > 127).astype(np.uint8)\n\n"
            "if binary_mask.sum() < 50:  # Skip bald instances\n"
            "    skipped_count += 1\n"
            "    continue"
        ),
        "metrics": [
            ("Raw Input Count", "30,000 images"),
            ("Clean Retained Count", "29,300 images (97.67%)"),
            ("Discarded Count", "700 images (2.33%)"),
            ("Spatial Reduction", "16x memory footprint reduction")
        ]
    },
    {
        "id": 3,
        "title": "Dataset Split & DataLoader Pipeline",
        "short_title": "3. Dataset Split",
        "subtitle": "Deterministic 80 / 10 / 10 Partitioning & Batching",
        "node_text": "Stratified Split\n80% Train | 10% Val | 10% Test\nBatch Size 16 DataLoader",
        "output_shape": "Batch (16, 3, 256, 256)",
        "card_color": "#3f2d54",
        "highlight_color": "#c084fc",
        "overview": (
            "To guarantee unbiased evaluation and scientific rigor, the 29,300 processed samples were "
            "partitioned into disjoint subsets using a deterministic random seed (42).\n\n"
            "The training set (23,440 images) exposes the model to broad demographic diversity. The validation "
            "set (2,930 images) guides learning rate scheduling and early stopping. The test set (2,930 images) "
            "remains strictly isolated until the final feature audit."
        ),
        "specs": [
            ("Split Proportion", "80% Train / 10% Validation / 10% Test"),
            ("Training Split", "23,440 images (80.0%)"),
            ("Validation Split", "2,930 images (10.0%)"),
            ("Test Split (Holdout)", "2,930 images (10.0%)"),
            ("Random Seed", "42 (deterministic reproducibility)"),
            ("DataLoader Config", "Batch size = 16, num_workers = 0 (Windows stable), pin_memory = True")
        ],
        "math_title": "Partition Cardinality & Batch Tensor Dimensions",
        "math_formula": (
            "Partition Cardinality:\n"
            "  N_total = 29,300\n"
            "  N_train = floor(0.80 * N_total) = 23,440\n"
            "  N_val   = floor(0.10 * N_total) = 2,930\n"
            "  N_test  = N_total - (N_train + N_val) = 2,930\n\n"
            "Batch Iteration Step:\n"
            "  Batches per Epoch = ceil(23,440 / 16) = 1,465 batches/epoch\n"
            "  Input Tensor Batch:   B_x in R^(16 x 3 x 256 x 256)\n"
            "  Target Tensor Batch:  B_y in {0, 1}^(16 x 1 x 256 x 256)"
        ),
        "code_snippet": (
            "# Data splitting and DataLoader creation\n"
            "train_ids, temp_ids = train_test_split(valid_ids, test_size=0.20, random_state=42)\n"
            "val_ids, test_ids   = train_test_split(temp_ids,  test_size=0.50, random_state=42)\n\n"
            "train_loader = DataLoader(\n"
            "    HairDataset(train_ids, augment=True),\n"
            "    batch_size=16, shuffle=True, num_workers=0, pin_memory=True\n"
            ")"
        ),
        "metrics": [
            ("Train Samples", "23,440 images (1,465 batches)"),
            ("Validation Samples", "2,930 images (184 batches)"),
            ("Test Samples", "2,930 images (184 batches)"),
            ("Total Processed", "29,300 clean samples")
        ]
    },
    {
        "id": 4,
        "title": "Data Augmentation & Normalization",
        "short_title": "4. Augmentation",
        "subtitle": "Albumentations Invariance Transforms & ImageNet Standardization",
        "node_text": "Augmentation Pipeline\nHorizontal Flip + Affine + Jitter\nImageNet Standardization",
        "output_shape": "Normalized (16, 3, 256, 256)",
        "card_color": "#4a3319",
        "highlight_color": "#fb923c",
        "overview": (
            "To prevent the U-Net from overfitting and ensure robust generalization across webcam angles, "
            "lighting variations, and head poses, dynamic data augmentations were applied on-the-fly "
            "during training using Albumentations 1.4.18.\n\n"
            "Transforms are applied jointly to both image and mask, preserving spatial alignment. "
            "Validation and testing sets are strictly kept clean with deterministic zero-augmentation."
        ),
        "specs": [
            ("Horizontal Flip", "p = 0.5 (exploits lateral bilateral facial symmetry)"),
            ("ShiftScaleRotate", "p = 0.5 (shift +-5%, scale +-10%, rotate +-15 deg, BORDER_CONSTANT)"),
            ("ColorJitter", "p = 0.3 (brightness +-0.2, contrast +-0.2, sat +-0.2, hue +-0.1)"),
            ("Normalization Mean", "mu = [0.485, 0.456, 0.406] (ImageNet RGB standard)"),
            ("Normalization Std", "sigma = [0.229, 0.224, 0.225] (ImageNet RGB standard)"),
            ("Target Mask Channel", "Added channel dimension -> (1, 256, 256) float32")
        ],
        "math_title": "Geometric & Color Transformation Equations",
        "math_formula": (
            "Affine Coordinate Transformation (Shift, Scale, Rotate):\n"
            "  [x']   [ s*cos(theta)  -s*sin(theta)  dx ] [x]\n"
            "  [y'] = [ s*sin(theta)   s*cos(theta)  dy ] [y]\n"
            "  [ 1]   [      0               0        1 ] [1]\n"
            "  where theta in [-15 deg, +15 deg], s in [0.9, 1.1], dx,dy in [-5%, +5%]\n\n"
            "Pixel Channel Normalization (Z-score standardization):\n"
            "  x_norm[c] = (x_raw[c] / 255.0 - mu[c]) / sigma[c],  c in {R, G, B}"
        ),
        "code_snippet": (
            "# Albumentations augmentation pipeline (dataset.py)\n"
            "train_transform = A.Compose([\n"
            "    A.HorizontalFlip(p=0.5),\n"
            "    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.5),\n"
            "    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),\n"
            "    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n"
            "    ToTensorV2()\n"
            "])"
        ),
        "metrics": [
            ("Active Transforms", "HorizontalFlip, ShiftScaleRotate, ColorJitter"),
            ("Input Range", "Raw uint8 [0..255] -> Normalized float32 [-2.1..+2.6]"),
            ("Evaluation Mode", "Validation & Test use purely deterministic Normalize()"),
            ("Alignment Guarantee", "Joint image+mask transform preserves boundary ground truth")
        ]
    },
    {
        "id": 5,
        "title": "Deep U-Net Model Architecture",
        "short_title": "5. U-Net Architecture",
        "subtitle": "4-Level Encoder-Decoder with Skip Connections (~7.76M Params)",
        "node_text": "Deep U-Net Architecture\nChannels: [32, 64, 128, 256, 512]\n7,763,041 Total Parameters",
        "output_shape": "Logits (16, 1, 256, 256)",
        "card_color": "#1e293b",
        "highlight_color": "#60a5fa",
        "overview": (
            "A customized deep U-Net architecture designed specifically for high-precision scalp hair "
            "segmentation. It uses 4 contracting encoder stages to capture multi-scale semantic context "
            "(edges -> strands -> global scalp shape) and 4 expanding decoder stages to reconstruct fine boundaries.\n\n"
            "Direct Skip Connections transfer high-resolution spatial feature maps from each encoder to its "
            "corresponding decoder level, bypassing information loss from downsampling and eliminating vanishing gradients."
        ),
        "specs": [
            ("Feature Progression", "[32, 64, 128, 256] with Bottleneck = 512 channels"),
            ("Encoder 1", "3 -> 32 -> 32 (10,208 params) | Pool 2x2 -> 128x128"),
            ("Encoder 2", "32 -> 64 -> 64 (55,552 params) | Pool 2x2 -> 64x64"),
            ("Encoder 3", "64 -> 128 -> 128 (221,696 params) | Pool 2x2 -> 32x32"),
            ("Encoder 4", "128 -> 256 -> 256 (885,760 params) | Pool 2x2 -> 16x16"),
            ("Bottleneck", "256 -> 512 -> 512 (3,540,992 params) at 16x16 resolution"),
            ("Decoder 4..1", "UpConv + Concat Skip + DoubleConv -> 3,048,800 params total"),
            ("Output Layer", "Conv2d(32 -> 1, kernel=1x1) (33 params) -> Raw Logits"),
            ("Total Parameters", "7,763,041 Trainable Parameters (89 MB on disk)")
        ],
        "math_title": "DoubleConv Block & Skip Concatenation Formulation",
        "math_formula": (
            "DoubleConv Building Block:\n"
            "  f(x) = ReLU( BatchNorm2d( Conv2d_3x3(x, bias=False) ) )\n"
            "  DoubleConv(x) = f( f(x) )\n\n"
            "Encoder Contracting Step:\n"
            "  e_k = DoubleConv_k(x_k)\n"
            "  x_{k+1} = MaxPool_2x2(e_k)\n\n"
            "Decoder Expanding Step (Skip Connection Fusion):\n"
            "  u_k = ConvTranspose2d_2x2(d_{k+1})\n"
            "  fused_k = Concat( [u_k, e_k], dim=channel )\n"
            "  d_k = DoubleConv_k(fused_k)\n\n"
            "Final Classification Projection:\n"
            "  z = Conv2d_1x1(d_1) in R^(B x 1 x 256 x 256)  [Raw Logits]"
        ),
        "code_snippet": (
            "# U-Net Architecture snippet (model.py)\n"
            "class UNet(nn.Module):\n"
            "    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):\n"
            "        super().__init__()\n"
            "        self.downs = nn.ModuleList()     # Encoders\n"
            "        self.ups   = nn.ModuleList()     # Decoders\n"
            "        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)\n"
            "        self.bottleneck = DoubleConv(256, 512)\n"
            "        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)"
        ),
        "metrics": [
            ("Total Parameters", "7,763,041 (100% trainable)"),
            ("Encoder Depth", "4 levels (256 -> 128 -> 64 -> 32 -> 16 px)"),
            ("Bottleneck Channels", "512 channels"),
            ("Model File Size", "88.9 MB (.pth checkpoint)")
        ]
    },
    {
        "id": 6,
        "title": "Hybrid Loss Function Formulation",
        "short_title": "6. Loss Function",
        "subtitle": "Binary Cross-Entropy with Logits + Soft Dice Loss",
        "node_text": "Hybrid Loss Objective\nBCE (Pixel Accuracy)\n+ Soft Dice (IoU Overlap)",
        "output_shape": "Scalar Loss L_total",
        "card_color": "#3b1e2a",
        "highlight_color": "#f43f5e",
        "overview": (
            "Semantic hair segmentation faces an inherent foreground-background imbalance: scalp hair occupies "
            "only ~25% of the total facial canvas. Standard Cross-Entropy alone produces conservative boundaries "
            "biased toward background.\n\n"
            "To solve this, a dual-objective loss was implemented: Binary Cross-Entropy (BCE) enforces sharp "
            "pixel-level boundary gradients, while Soft Dice Loss directly maximizes volumetric region overlap."
        ),
        "specs": [
            ("Dual Objective", "L_total = L_BCE + L_Dice"),
            ("BCE Formulation", "BCEWithLogitsLoss (numerically stable via log-sum-exp)"),
            ("Soft Dice Loss", "Differentiable approximation of Dice Similarity Coefficient"),
            ("Smoothing Factor", "epsilon = 1e-6 (prevents division by zero in zero-coverage cases)"),
            ("Gradient Dynamics", "Dice loss penalizes false negatives; BCE penalizes boundary ambiguity"),
            ("Optimization Behavior", "Smooth monotonic loss reduction across all 15 epochs")
        ],
        "math_title": "Mathematical Derivation of the Hybrid Loss",
        "math_formula": (
            "1. Binary Cross-Entropy with Logits Loss:\n"
            "   L_BCE(z, y) = - 1/N * Sum_i [ y_i * log(sigma(z_i)) + (1 - y_i) * log(1 - sigma(z_i)) ]\n"
            "   where sigma(z) = 1 / (1 + exp(-z))\n\n"
            "2. Soft Dice Loss (Continuous Differentiable):\n"
            "   L_Dice(p, y) = 1 - ( 2 * Sum(p_i * y_i) + eps ) / ( Sum(p_i) + Sum(y_i) + eps )\n"
            "   where p_i = sigma(z_i) in (0, 1), eps = 1e-6\n\n"
            "3. Total Joint Optimization Objective:\n"
            "   L_Total(z, y) = L_BCE(z, y) + L_Dice(sigma(z), y)"
        ),
        "code_snippet": (
            "# Combined loss function implementation (loss.py)\n"
            "class BCEDiceLoss(nn.Module):\n"
            "    def __init__(self, smooth=1e-6):\n"
            "        super().__init__()\n"
            "        self.bce = nn.BCEWithLogitsLoss()\n"
            "        self.smooth = smooth\n\n"
            "    def forward(self, logits, targets):\n"
            "        bce_loss = self.bce(logits, targets)\n"
            "        probs = torch.sigmoid(logits)\n"
            "        intersection = (probs * targets).sum()\n"
            "        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)\n"
            "        return bce_loss + (1.0 - dice)"
        ),
        "metrics": [
            ("Epoch 1 Train Loss", "0.3120"),
            ("Epoch 15 Train Loss", "0.1189"),
            ("Epoch 15 Val Loss", "0.1429"),
            ("Final Test Loss", "0.1434")
        ]
    },
    {
        "id": 7,
        "title": "Training & Optimization Dynamics",
        "short_title": "7. Optimization",
        "subtitle": "Adam Optimizer, Plateau Decay & RTX 5060 GPU Execution",
        "node_text": "Training Dynamics\nAdam lr=1e-4 | 15 Epochs\nBest Model Checkpoint (Ep 15)",
        "output_shape": "Checkpoint best_model.pth",
        "card_color": "#2a3b1e",
        "highlight_color": "#a3e635",
        "overview": (
            "Training was conducted on an NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7 VRAM, Blackwell architecture) "
            "running CUDA 12.8 and PyTorch 2.11.0. The optimizer employed was Adam with an initial learning rate of 1e-4.\n\n"
            "Adaptive Scheduling: ReduceLROnPlateau monitored validation Dice score. Checkpoints were dynamically saved "
            "whenever validation Dice achieved a new peak. Peak performance was unlocked at Epoch 15 with Val Dice = 95.20%."
        ),
        "specs": [
            ("Compute Hardware", "NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7, Blackwell arch)"),
            ("CUDA / PyTorch", "CUDA 12.8 | PyTorch 2.11.0+cu128"),
            ("Optimizer", "Adam (lr = 1e-4, beta1 = 0.9, beta2 = 0.999, weight_decay = 1e-4)"),
            ("Learning Rate Schedule", "ReduceLROnPlateau (mode='max', factor=0.5, patience=5)"),
            ("Completed Epochs", "15 full epochs (21,975 backprop steps across 23,440 images)"),
            ("Epoch Execution Time", "~10 minutes per epoch (~2.5 hours total training duration)"),
            ("Best Checkpoint Saved", "outputs/models/best_model.pth (Epoch 15: Val Dice = 0.9520)")
        ],
        "math_title": "Optimization Update Rule & Convergence Metrics",
        "math_formula": (
            "Adam Parameter Update Rule:\n"
            "  m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t\n"
            "  v_t = beta2 * v_{t-1} + (1 - beta2) * (grad_t)^2\n"
            "  m_hat_t = m_t / (1 - beta1^t),   v_hat_t = v_t / (1 - beta2^t)\n"
            "  theta_t = theta_{t-1} - ( lr / (sqrt(v_hat_t) + eps) ) * m_hat_t\n\n"
            "Plateau Adaptation:\n"
            "  If Val_Dice does not increase for 5 epochs: lr_new = lr_old * 0.5"
        ),
        "code_snippet": (
            "# Training loop execution (train.py)\n"
            "optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)\n"
            "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n"
            "    optimizer, mode='max', factor=0.5, patience=5\n"
            ")\n"
            "for epoch in range(1, 16):\n"
            "    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)\n"
            "    val_loss, val_dice, val_iou = evaluate(model, val_loader, device)\n"
            "    if val_dice > best_val_dice:\n"
            "        torch.save({'model_state_dict': model.state_dict()}, 'best_model.pth')"
        ),
        "metrics": [
            ("Epoch 01 Val Dice", "0.9021 | Val IoU: 0.8223"),
            ("Epoch 05 Val Dice", "0.9380 | Val IoU: 0.8834"),
            ("Epoch 10 Val Dice", "0.9472 | Val IoU: 0.9001"),
            ("Epoch 15 BEST Dice", "0.9520 (95.20%) | Val IoU: 0.9085")
        ]
    },
    {
        "id": 8,
        "title": "Unseen Test Evaluation & Hair Tracker",
        "short_title": "8. Inference & Tracker",
        "subtitle": "Independent Test Benchmark & AKAZE Progress Tracking",
        "node_text": "Inference & Tracking\nTest Dice: 95.19% | IoU: 90.84%\nAKAZE Alignment + Change Map",
        "output_shape": "Scalp Mask + Progress Delta",
        "card_color": "#1e3b37",
        "highlight_color": "#2dd4bf",
        "overview": (
            "The best checkpoint was comprehensively evaluated on 2,930 unseen test images, achieving an "
            "outstanding 95.19% Dice Score and 90.84% IoU. The model reliably identifies fine hairlines, "
            "diffuse hair partings, and various hairstyles across diverse subjects.\n\n"
            "Desktop GUI Integration: In the Hair Progress Tracker application, follow-up photos are registered "
            "to baseline coordinates via AKAZE+RANSAC homography. The system computes exact scalp hair coverage "
            "percentages, net change delta (Delta = Current - Baseline), and generates an intuitive differential "
            "Change Map (Red = Hair Lost, Green = Hair Gained)."
        ),
        "specs": [
            ("Test Set Evaluation", "2,930 independent test images"),
            ("Test Dice Similarity", "0.9519 (95.19% spatial overlap)"),
            ("Test IoU (Jaccard)", "0.9084 (90.84% intersection over union)"),
            ("Precision / Recall", "Precision = 95.12% | Recall = 95.27%"),
            ("Mean Test Loss", "0.1434"),
            ("Inference Speed", "~12 milliseconds per frame on RTX 5060 GPU"),
            ("AKAZE Alignment", "Non-linear scale space keypoints + Hamming matcher + RANSAC homography"),
            ("Differential Change Map", "Red = Hair Lost | Green = Hair Gained | Overlaid on face")
        ],
        "math_title": "Inference Thresholding & Progress Tracking Equations",
        "math_formula": (
            "1. Inference Binarization:\n"
            "   Y_pred(x, y) = 1  if sigma( z(x, y) ) >= 0.5  else 0\n\n"
            "2. Scalp Hair Coverage Percentage:\n"
            "   Coverage(M) = ( Sum_{x,y} M(x, y) ) / ( 256 * 256 ) * 100 %\n\n"
            "3. Progress Change Metrics:\n"
            "   Absolute Delta = Coverage(Current_Aligned) - Coverage(Baseline)\n"
            "   Relative Change % = ( Absolute Delta / Coverage(Baseline) ) * 100 %\n\n"
            "4. Differential Change Map:\n"
            "   Lost_Mask(x, y)   = (Baseline == 1) and (Current_Aligned == 0)   -> RED [230, 60, 60]\n"
            "   Gained_Mask(x, y) = (Baseline == 0) and (Current_Aligned == 1)   -> GREEN [50, 230, 70]"
        ),
        "code_snippet": (
            "# Downstream clinical progress tracking\n"
            "aligned_current = akaze_align(baseline_rgb, current_rgb)\n"
            "base_mask  = predict_hair_mask(model, baseline_tensor, device)\n"
            "cur_mask   = predict_hair_mask(model, aligned_tensor, device)\n\n"
            "delta_coverage = compute_coverage(cur_mask) - compute_coverage(base_mask)\n"
            "lost_hair_pixels   = np.logical_and(base_mask > 0, cur_mask == 0)\n"
            "gained_hair_pixels = np.logical_and(base_mask == 0, cur_mask > 0)"
        ),
        "metrics": [
            ("Test Dice Score", "95.19% (Target exceeded)"),
            ("Test IoU Score", "90.84%"),
            ("Precision", "95.12%"),
            ("Recall", "95.27% (High sensitivity to hair boundaries)")
        ]
    }
]


# ──────────────────────────────────────────────────────────────
#  Interactive Visualizer Window Class
# ──────────────────────────────────────────────────────────────

class TrainingPipelineVisualizer(ctk.CTkToplevel):
    """
    Modern, interactive window presenting the complete end-to-end model training
    pipeline with a dynamic slider, interactive flowchart canvas, and deep-dive
    technical inspector.
    """

    def __init__(self, master=None):
        super().__init__(master)

        self.title("Model Training Pipeline & Architecture Visualizer | DL Capstone")
        self.geometry("1200x780")
        self.minsize(1050, 700)

        # State
        self.current_stage_idx = 0   # 0 to 7
        self.is_playing = False
        self.play_timer = None
        self.card_coords = []        # Store canvas bounding boxes for click hit-testing

        # Build UI layout
        self._build_header()
        self._build_slider_bar()
        self._build_main_workspace()

        # Render first stage
        self._select_stage(0, update_slider=True)

        # Handle close cleanup
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Header ─────────────────────────────────────────────────
    def _build_header(self):
        header_frame = ctk.CTkFrame(self, fg_color=("gray85", "#181b22"), corner_radius=0, height=56)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        left_titles = ctk.CTkFrame(header_frame, fg_color="transparent")
        left_titles.pack(side="left", padx=16, pady=8)

        title_lbl = ctk.CTkLabel(
            left_titles, text="End-to-End Model Training Pipeline & Architecture Visualizer",
            font=ctk.CTkFont(size=16, weight="bold"), text_color=("gray10", "gray95")
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            left_titles,
            text="Interactive Walkthrough: Raw Data -> Cleaning -> 80/10/10 Split -> Augmentation -> U-Net 7.8M -> BCE+Dice -> RTX 5060 Training -> 95.19% Dice",
            font=ctk.CTkFont(size=11), text_color=("gray40", "gray65")
        )
        sub_lbl.pack(anchor="w")

        # Close / Return button
        btn_close = ctk.CTkButton(
            header_frame, text="Close Window", width=100, height=30,
            fg_color=("gray70", "gray30"), hover_color=("gray60", "gray25"),
            command=self._on_close
        )
        btn_close.pack(side="right", padx=16, pady=12)

    # ── Slider & Stepper Bar ───────────────────────────────────
    def _build_slider_bar(self):
        slider_container = ctk.CTkFrame(self, fg_color=("gray90", "#1e222b"), corner_radius=8)
        slider_container.pack(fill="x", padx=14, pady=(10, 6))

        # Top pill buttons row
        pills_frame = ctk.CTkFrame(slider_container, fg_color="transparent")
        pills_frame.pack(fill="x", padx=12, pady=(10, 4))

        self.pill_buttons = []
        for i, stage in enumerate(STAGES):
            btn = ctk.CTkButton(
                pills_frame,
                text=stage["short_title"],
                height=26,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="transparent",
                text_color="gray70",
                hover_color=("gray80", "#282e3c"),
                corner_radius=6,
                command=lambda s=i: self._select_stage(s, update_slider=True)
            )
            btn.pack(side="left", expand=True, fill="x", padx=2)
            self.pill_buttons.append(btn)

        # Dynamic Slider row
        ctrl_frame = ctk.CTkFrame(slider_container, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=12, pady=(4, 10))

        # Previous button
        self.btn_prev = ctk.CTkButton(
            ctrl_frame, text="< Prev Step", width=95, height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._prev_stage
        )
        self.btn_prev.pack(side="left", padx=(0, 10))

        # Slider widget
        self.stage_slider = ctk.CTkSlider(
            ctrl_frame,
            from_=1, to=8,
            number_of_steps=7,
            command=self._on_slider_moved
        )
        self.stage_slider.set(1)
        self.stage_slider.pack(side="left", expand=True, fill="x", padx=10)

        # Next button
        self.btn_next = ctk.CTkButton(
            ctrl_frame, text="Next Step >", width=95, height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._next_stage
        )
        self.btn_next.pack(side="left", padx=(10, 10))

        # Auto-Play Tour button
        self.btn_play = ctk.CTkButton(
            ctrl_frame, text="Auto-Play Tour", width=120, height=32,
            fg_color="#10b981", hover_color="#059669",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._toggle_auto_play
        )
        self.btn_play.pack(side="right", padx=(4, 0))

        # Status text below slider
        self.lbl_progress = ctk.CTkLabel(
            slider_container,
            text="Stage 1 of 8: Raw Data Acquisition",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8"
        )
        self.lbl_progress.pack(pady=(0, 6))

    # ── Main Workspace ─────────────────────────────────────────
    def _build_main_workspace(self):
        workspace = ctk.CTkFrame(self, fg_color="transparent")
        workspace.pack(fill="both", expand=True, padx=14, pady=(4, 12))
        workspace.columnconfigure(0, weight=4)   # Left flowchart canvas (~40%)
        workspace.columnconfigure(1, weight=6)   # Right technical inspector (~60%)
        workspace.rowconfigure(0, weight=1)

        # ── Left: Interactive Flowchart Canvas ────────────────
        left_box = ctk.CTkFrame(workspace, fg_color=("gray92", "#181b22"), corner_radius=8)
        left_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        canvas_header = ctk.CTkFrame(left_box, fg_color="transparent")
        canvas_header.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            canvas_header, text="Pipeline Flow Diagram",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left")
        ctk.CTkLabel(
            canvas_header, text="(Click any node to inspect)",
            font=ctk.CTkFont(size=10), text_color="gray60"
        ).pack(side="right")

        # Scrollable Canvas
        canvas_container = tk.Frame(left_box, bg="#13151b")
        canvas_container.pack(fill="both", expand=True, padx=10, pady=(2, 10))

        self.canvas_scroll = tk.Scrollbar(canvas_container, orient="vertical")
        self.flow_canvas = tk.Canvas(
            canvas_container,
            bg="#13151b",
            highlightthickness=0,
            yscrollcommand=self.canvas_scroll.set
        )
        self.canvas_scroll.config(command=self.flow_canvas.yview)
        self.canvas_scroll.pack(side="right", fill="y")
        self.flow_canvas.pack(side="left", fill="both", expand=True)

        # Canvas event bindings
        self.flow_canvas.bind("<Button-1>", self._on_canvas_click)
        self.flow_canvas.bind("<Motion>", self._on_canvas_motion)

        # ── Right: Deep-Dive Technical Inspector ──────────────
        right_box = ctk.CTkFrame(workspace, fg_color=("gray92", "#181b22"), corner_radius=8)
        right_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)

        self.detail_scroll = ctk.CTkScrollableFrame(right_box, fg_color="transparent")
        self.detail_scroll.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_detail_cards()

    # ── Detail Inspector Cards ─────────────────────────────────
    def _build_detail_cards(self):
        # Header banner inside inspector
        self.card_header = ctk.CTkFrame(self.detail_scroll, fg_color="#1e293b", corner_radius=8)
        self.card_header.pack(fill="x", padx=6, pady=(4, 8))

        self.lbl_card_stage_num = ctk.CTkLabel(
            self.card_header, text="STAGE 1 / 8",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#38bdf8"
        )
        self.lbl_card_stage_num.pack(anchor="w", padx=14, pady=(8, 0))

        self.lbl_card_title = ctk.CTkLabel(
            self.card_header, text="Raw Data Acquisition",
            font=ctk.CTkFont(size=17, weight="bold")
        )
        self.lbl_card_title.pack(anchor="w", padx=14, pady=(2, 0))

        self.lbl_card_subtitle = ctk.CTkLabel(
            self.card_header, text="CelebAMask-HQ Origin & Multi-Class Parsing",
            font=ctk.CTkFont(size=11), text_color="gray65"
        )
        self.lbl_card_subtitle.pack(anchor="w", padx=14, pady=(0, 10))

        # Card 1: Overview & Role
        self.card_overview = ctk.CTkFrame(self.detail_scroll, fg_color=("gray88", "#1e222b"), corner_radius=8)
        self.card_overview.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(
            self.card_overview, text="Stage Overview & Objective",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#38bdf8"
        ).pack(anchor="w", padx=14, pady=(10, 4))

        self.lbl_overview_text = ctk.CTkLabel(
            self.card_overview, text="",
            font=ctk.CTkFont(size=11),
            justify="left", wraplength=560
        )
        self.lbl_overview_text.pack(anchor="w", padx=14, pady=(0, 12))

        # Card 2: Technical Specifications & Data Dimensions
        self.card_specs = ctk.CTkFrame(self.detail_scroll, fg_color=("gray88", "#1e222b"), corner_radius=8)
        self.card_specs.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(
            self.card_specs, text="Technical Specifications & Dimensions",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#38bdf8"
        ).pack(anchor="w", padx=14, pady=(10, 6))

        self.specs_table_frame = ctk.CTkFrame(self.card_specs, fg_color="transparent")
        self.specs_table_frame.pack(fill="x", padx=14, pady=(0, 12))

        # Card 3: Mathematical Formulation & Operations
        self.card_math = ctk.CTkFrame(self.detail_scroll, fg_color=("gray88", "#1e222b"), corner_radius=8)
        self.card_math.pack(fill="x", padx=6, pady=4)
        self.lbl_math_title = ctk.CTkLabel(
            self.card_math, text="Mathematical Formulation & Operations",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#38bdf8"
        )
        self.lbl_math_title.pack(anchor="w", padx=14, pady=(10, 4))

        self.lbl_math_formula = ctk.CTkLabel(
            self.card_math, text="",
            font=ctk.CTkFont(family="Consolas", size=10),
            justify="left", wraplength=560, text_color="#e2e8f0"
        )
        self.lbl_math_formula.pack(anchor="w", padx=14, pady=(0, 12))

        # Card 4: Code & Implementation Execution
        self.card_code = ctk.CTkFrame(self.detail_scroll, fg_color=("gray88", "#1e222b"), corner_radius=8)
        self.card_code.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(
            self.card_code, text="Implementation Code Snippet",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#38bdf8"
        ).pack(anchor="w", padx=14, pady=(10, 4))

        code_box = ctk.CTkFrame(self.card_code, fg_color="#0f1117", corner_radius=6)
        code_box.pack(fill="x", padx=14, pady=(0, 12))
        self.lbl_code_text = ctk.CTkLabel(
            code_box, text="",
            font=ctk.CTkFont(family="Consolas", size=10),
            justify="left", text_color="#a5f3fc"
        )
        self.lbl_code_text.pack(anchor="w", padx=12, pady=10)

        # Card 5: Experimental Benchmarks & Results
        self.card_metrics = ctk.CTkFrame(self.detail_scroll, fg_color=("gray88", "#1e222b"), corner_radius=8)
        self.card_metrics.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(
            self.card_metrics, text="Key Metrics & Verification Status",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#38bdf8"
        ).pack(anchor="w", padx=14, pady=(10, 6))

        self.metrics_grid_frame = ctk.CTkFrame(self.card_metrics, fg_color="transparent")
        self.metrics_grid_frame.pack(fill="x", padx=14, pady=(0, 12))

    # ── Render Flowchart Canvas ────────────────────────────────
    def _draw_flowchart(self):
        self.flow_canvas.delete("all")
        self.card_coords = []

        card_w = 340
        card_h = 76
        start_x = 24
        start_y = 16
        spacing_y = 108   # 76 card + 32 arrow gap

        total_canvas_h = start_y + len(STAGES) * spacing_y + 40
        self.flow_canvas.config(scrollregion=(0, 0, 400, total_canvas_h))

        for i, stage in enumerate(STAGES):
            y1 = start_y + i * spacing_y
            y2 = y1 + card_h
            x1 = start_x
            x2 = start_x + card_w

            is_active = (i == self.current_stage_idx)
            self.card_coords.append((x1, y1, x2, y2, i))

            # Connective arrow to next card
            if i < len(STAGES) - 1:
                arrow_y1 = y2
                arrow_y2 = y1 + spacing_y
                arrow_x = start_x + card_w // 2

                # Directional arrow line
                arrow_color = "#38bdf8" if (i == self.current_stage_idx) else "#334155"
                self.flow_canvas.create_line(
                    arrow_x, arrow_y1 + 2, arrow_x, arrow_y2 - 6,
                    fill=arrow_color, width=2, arrow=tk.LAST, arrowshape=(8, 10, 4)
                )
                # Output shape tag beside arrow
                self.flow_canvas.create_text(
                    arrow_x + 10, (arrow_y1 + arrow_y2) // 2,
                    text=stage["output_shape"],
                    fill="#64748b", font=("Consolas", 8), anchor="w"
                )

            # Card background
            if is_active:
                bg_color = "#1e293b"
                border_color = stage["highlight_color"]
                border_width = 3
            else:
                bg_color = "#151821"
                border_color = "#2e384d"
                border_width = 1

            # Draw rounded card outline (simulated with polygon/rect)
            self.flow_canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=bg_color, outline=border_color, width=border_width
            )

            # Glowing indicator bar on left edge of active card
            accent_color = stage["highlight_color"] if is_active else "#475569"
            self.flow_canvas.create_rectangle(
                x1, y1, x1 + 6, y2,
                fill=accent_color, outline=""
            )

            # Stage number badge
            badge_bg = stage["highlight_color"] if is_active else "#283347"
            badge_fg = "#000000" if is_active else "#94a3b8"
            self.flow_canvas.create_rectangle(
                x1 + 14, y1 + 12, x1 + 42, y1 + 38,
                fill=badge_bg, outline=""
            )
            self.flow_canvas.create_text(
                x1 + 28, y1 + 25,
                text=f"{stage['id']:02d}",
                fill=badge_fg, font=("Segoe UI", 10, "bold")
            )

            # Node title & summary text
            title_color = "#ffffff" if is_active else "#cbd5e1"
            self.flow_canvas.create_text(
                x1 + 52, y1 + 22,
                text=stage["title"],
                fill=title_color, font=("Segoe UI", 11, "bold"), anchor="w"
            )

            sub_color = "#38bdf8" if is_active else "#64748b"
            self.flow_canvas.create_text(
                x1 + 52, y1 + 46,
                text=stage["subtitle"][:42] + ("..." if len(stage["subtitle"]) > 42 else ""),
                fill=sub_color, font=("Segoe UI", 9), anchor="w"
            )

    # ── Interaction & Selection ────────────────────────────────
    def _select_stage(self, stage_idx: int, update_slider: bool = False):
        if not (0 <= stage_idx < len(STAGES)):
            return

        self.current_stage_idx = stage_idx
        stage = STAGES[stage_idx]

        # Update slider
        if update_slider:
            self.stage_slider.set(stage_idx + 1)

        # Update stage text
        self.lbl_progress.configure(
            text=f"Stage {stage['id']} of 8: {stage['title']}",
            text_color=stage["highlight_color"]
        )

        # Update pill button appearances
        for i, btn in enumerate(self.pill_buttons):
            if i == stage_idx:
                btn.configure(
                    fg_color=stage["highlight_color"],
                    text_color="#000000",
                    hover_color=stage["highlight_color"]
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color="gray70",
                    hover_color=("gray80", "#282e3c")
                )

        # Redraw flowchart with highlighted active card
        self._draw_flowchart()

        # Update right inspector details
        self._populate_inspector(stage)

        # Scroll flowchart canvas to keep active node visible
        frac = max(0.0, min(1.0, stage_idx / (len(STAGES) - 1)))
        self.flow_canvas.yview_moveto(max(0.0, frac - 0.15))

    def _populate_inspector(self, stage: dict):
        # Header banner
        self.lbl_card_stage_num.configure(
            text=f"STAGE {stage['id']} OF 8",
            text_color=stage["highlight_color"]
        )
        self.lbl_card_title.configure(text=stage["title"])
        self.lbl_card_subtitle.configure(text=stage["subtitle"])

        # Card 1: Overview
        self.lbl_overview_text.configure(text=stage["overview"])

        # Card 2: Specs table
        for widget in self.specs_table_frame.winfo_children():
            widget.destroy()

        for idx, (param, val) in enumerate(stage["specs"]):
            row = ctk.CTkFrame(
                self.specs_table_frame,
                fg_color="#181e28" if idx % 2 == 0 else "transparent",
                corner_radius=4
            )
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(
                row, text=f"{param}:",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="gray65", width=190, anchor="w"
            ).pack(side="left", padx=8, pady=3)
            ctk.CTkLabel(
                row, text=val,
                font=ctk.CTkFont(size=10),
                text_color="gray95", anchor="w"
            ).pack(side="left", padx=4, pady=3)

        # Card 3: Math formula
        self.lbl_math_title.configure(text=stage["math_title"])
        self.lbl_math_formula.configure(text=stage["math_formula"])

        # Card 4: Code
        self.lbl_code_text.configure(text=stage["code_snippet"])

        # Card 5: Metrics grid
        for widget in self.metrics_grid_frame.winfo_children():
            widget.destroy()

        for idx, (m_title, m_val) in enumerate(stage["metrics"]):
            box = ctk.CTkFrame(self.metrics_grid_frame, fg_color="#141a24", corner_radius=6)
            box.pack(fill="x", pady=2)
            ctk.CTkLabel(
                box, text=f"{m_title}:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="gray65", width=200, anchor="w"
            ).pack(side="left", padx=10, pady=4)
            ctk.CTkLabel(
                box, text=m_val,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=stage["highlight_color"], anchor="w"
            ).pack(side="left", padx=6, pady=4)

        # Reset detail scroll position to top
        self.detail_scroll._parent_canvas.yview_moveto(0.0)

    # ── Slider & Stepper Handlers ──────────────────────────────
    def _on_slider_moved(self, value):
        idx = int(round(value)) - 1
        idx = max(0, min(7, idx))
        if idx != self.current_stage_idx:
            self._select_stage(idx, update_slider=False)

    def _prev_stage(self):
        new_idx = max(0, self.current_stage_idx - 1)
        self._select_stage(new_idx, update_slider=True)

    def _next_stage(self):
        new_idx = min(len(STAGES) - 1, self.current_stage_idx + 1)
        self._select_stage(new_idx, update_slider=True)

    def _on_canvas_click(self, event):
        canvas_y = self.flow_canvas.canvasy(event.y)
        canvas_x = self.flow_canvas.canvasx(event.x)
        for (x1, y1, x2, y2, idx) in self.card_coords:
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                self._select_stage(idx, update_slider=True)
                break

    def _on_canvas_motion(self, event):
        canvas_y = self.flow_canvas.canvasy(event.y)
        canvas_x = self.flow_canvas.canvasx(event.x)
        is_hover = False
        for (x1, y1, x2, y2, idx) in self.card_coords:
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                is_hover = True
                break
        self.flow_canvas.config(cursor="hand2" if is_hover else "")

    # ── Auto-Play Tour ─────────────────────────────────────────
    def _toggle_auto_play(self):
        if self.is_playing:
            self._stop_auto_play()
        else:
            self._start_auto_play()

    def _start_auto_play(self):
        self.is_playing = True
        self.btn_play.configure(text="Pause Tour", fg_color="#ef4444", hover_color="#dc2626")
        self._auto_play_step()

    def _stop_auto_play(self):
        self.is_playing = False
        self.btn_play.configure(text="Auto-Play Tour", fg_color="#10b981", hover_color="#059669")
        if self.play_timer:
            self.after_cancel(self.play_timer)
            self.play_timer = None

    def _auto_play_step(self):
        if not self.is_playing:
            return
        next_idx = (self.current_stage_idx + 1) % len(STAGES)
        self._select_stage(next_idx, update_slider=True)
        # Advance every 3.5 seconds
        self.play_timer = self.after(3500, self._auto_play_step)

    def _on_close(self):
        self._stop_auto_play()
        self.destroy()


# ──────────────────────────────────────────────────────────────
#  Standalone test execution
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Testing Visualizer")
    root.geometry("300x200")
    app = TrainingPipelineVisualizer(master=root)
    root.mainloop()
