---
title: Hair Progress Tracker & U-Net Pipeline Visualizer
emoji: 💇‍♂️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Clinical Scalp Hair Progression Monitoring using Deep U-Net & Biometric Alignment
---

# Hair Progress Tracker & Deep Learning Pipeline Visualizer
### Deep Learning Capstone Project | Clinical Scalp Hair Segmentation & Progression Tracking

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-blue)](https://huggingface.co/AlgomCorp/hair-segmentation-unet)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/U--Net-7.76M%20Params-success)](https://huggingface.co/AlgomCorp/hair-segmentation-unet)
[![Test Dice](https://img.shields.io/badge/Test%20Dice-95.19%25-brightgreen)](https://huggingface.co/AlgomCorp/hair-segmentation-unet)

---

## 🌟 Overview & Clinical Utility

The **Hair Progress Tracker** is a clinical-grade deep learning system designed for quantitative, objective monitoring of scalp hair progression (alopecia, hair loss treatments, hair transplant recovery, and regrowth therapy).

### Core Engineering Capabilities (v2.0):
1. **Accurate U-Net Semantic Segmentation**:
   * Symmetrical U-Net with channels `[32, 64, 128, 256]` and bottleneck `512` (~7.76M parameters).
   * Benchmarked on **2,930 holdout test images**: **Test Dice: 95.19%**, **Test IoU: 90.84%**, **Precision: 94.88%**, **Recall: 95.51%**.
2. **Test-Time Augmentation (TTA)**:
   * Dual-pass mirror-flip probability averaging eliminates boundary edge jitter and stabilizes hairline segmentation.
3. **Hierarchical Multi-Tier Registration**:
   * **Tier 1 (Facial Biometric Alignment)**: Rigid similarity transform mapping facial centers and pupil geometry.
   * **Tier 2 (AKAZE + RANSAC)**: Feature matching with affine determinant sanity checks ($0.4 \le \det(H) \le 2.5$) for crown/vertex views.
   * **Tier 3 (Reference Frame)**: Evaluates unaligned photos with true positive structural similarity ($0.0 \le \text{SSIM} \le 1.0$).
4. **Scale-Invariant Cranium Normalization**:
   * Computes hair density relative to the patient's anatomical head region ($\frac{N_{\text{hair}}}{N_{\text{head}}}$), guaranteeing invariance to camera zoom and distance shifts.
5. **Continuous Scalp Follicle Density Heatmap**:
   * Turbo colormap of continuous sigmoid probabilities identifies regions of sub-clinical thinning before complete follicular loss occurs.
6. **Dual Hairline Shift Contours**:
   * Simultaneous visualization of Baseline (Green) and Follow-up (Blue) hairlines with vertical pixel displacement tracking.
7. **Clinical Trichology Staging**:
   * Automatic classification on the Norwood-Hamilton Scale (Types I–VII) for male pattern hair loss.
8. **Interactive Split Curtain Slider**:
   * Scrubbable $0\% \to 100\%$ before/after wipe viewer.
9. **Mathematical Provenance & Pixel Audit Inspector**:
   * Live on-screen verification of raw pixel counts ($65,536\text{ px}$ frame grid), formulas, and tensor metadata proving 100% determinism.
10. **Medical-Grade 6-Panel Report Export**:
    * Generates a publication-grade summary report PNG.
