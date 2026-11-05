"""
evaluate.py
-----------
Evaluates the trained U-Net on the test set and generates:
  - A printed results table (Dice, IoU, Precision, Recall, Test Loss)
  - outputs/results/evaluation_report.csv
  - outputs/results/per_image_scores.csv  (score for every test image)
  - outputs/plots/sample_predictions/test_samples.png

Usage:
    python src/evaluate.py \
        --data_dir   data/processed \
        --splits_dir data/splits \
        --model      outputs/models/best_model.pth \
        --output_dir outputs
"""

import os
import sys
import csv
import argparse

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from dataset  import HairSegDataset, get_val_transforms
from model    import UNet
from loss     import BCEDiceLoss
from metrics  import compute_all_metrics
from utils    import save_sample_predictions, denormalize, overlay_mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on test set")
    parser.add_argument("--data_dir",   type=str, default="data/processed")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--model",      type=str, default="outputs/models/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers",type=int, default=0)
    parser.add_argument("--threshold",  type=float, default=0.5)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    results_dir = os.path.join(args.output_dir, "results")
    pred_dir    = os.path.join(args.output_dir, "plots", "sample_predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(pred_dir,    exist_ok=True)

    # -- Device ---------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  U-Net Evaluation ? Test Set")
    print(f"  Device : {device}")
    print(f"  Model  : {args.model}")
    print(f"{'='*60}\n")

    # -- Load model -----------------------------
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}\nTrain first with train.py")

    checkpoint = torch.load(args.model, map_location=device)
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved_epoch = checkpoint.get("epoch", "?")
    saved_dice  = checkpoint.get("val_dice", "?")
    print(f"  Loaded checkpoint from epoch {saved_epoch} | val Dice: {saved_dice}")

    # -- Test DataLoader -------------------------
    from torch.utils.data import DataLoader
    test_split = os.path.join(args.splits_dir, "test.txt")
    test_ds    = HairSegDataset(test_split, args.data_dir, get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    print(f"  Test samples: {len(test_ds):,} | Batches: {len(test_loader)}\n")

    # -- Per-image CSV ---------------------------
    per_image_path = os.path.join(results_dir, "per_image_scores.csv")
    criterion = BCEDiceLoss()

    agg = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    per_image_rows = []

    for images, masks, img_ids in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits  = model(images)
        loss    = criterion(logits, masks).item()
        metrics = compute_all_metrics(logits, masks, threshold=args.threshold)

        agg["loss"]      += loss
        for k in ["dice", "iou", "precision", "recall"]:
            agg[k] += metrics[k]

        # Per-image scores (one row per image in the batch)
        batch_preds = (torch.sigmoid(logits) > args.threshold).float()
        for i, img_id in enumerate(img_ids):
            m = compute_all_metrics(
                logits[i:i+1], masks[i:i+1], threshold=args.threshold)
            per_image_rows.append({
                "img_id": img_id,
                "dice": f"{m['dice']:.6f}",
                "iou":  f"{m['iou']:.6f}",
                "precision": f"{m['precision']:.6f}",
                "recall":    f"{m['recall']:.6f}",
            })

    n = len(test_loader)
    final = {k: v / n for k, v in agg.items()}

    # -- Save per-image CSV ----------------------
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img_id","dice","iou","precision","recall"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    # -- Summary report CSV ----------------------
    report_path = os.path.join(results_dir, "evaluation_report.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Dice Score",  f"{final['dice']:.4f}"])
        writer.writerow(["IoU",         f"{final['iou']:.4f}"])
        writer.writerow(["Precision",   f"{final['precision']:.4f}"])
        writer.writerow(["Recall",      f"{final['recall']:.4f}"])
        writer.writerow(["Test Loss",   f"{final['loss']:.4f}"])

    # -- Print results table ---------------------
    print(f"\n{'='*45}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Dice Score':<20} {final['dice']:>10.4f}")
    print(f"  {'IoU':<20} {final['iou']:>10.4f}")
    print(f"  {'Precision':<20} {final['precision']:>10.4f}")
    print(f"  {'Recall':<20} {final['recall']:>10.4f}")
    print(f"  {'Test Loss':<20} {final['loss']:>10.4f}")
    print(f"{'='*45}")
    print(f"\n  Reports saved:")
    print(f"    {report_path}")
    print(f"    {per_image_path}")

    # -- Save sample test predictions ------------
    save_sample_predictions(model, test_loader, device,
                            save_dir=pred_dir, epoch=0, n_samples=8)
    print(f"    {pred_dir}/epoch_000.png")

    # -- Find worst predictions (failure cases) --
    per_image_rows.sort(key=lambda x: float(x["dice"]))
    worst = per_image_rows[:20]
    worst_path = os.path.join(results_dir, "worst_predictions.csv")
    with open(worst_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img_id","dice","iou","precision","recall"])
        writer.writeheader()
        writer.writerows(worst)
    print(f"    {worst_path}  (20 worst predictions for failure analysis)")

    print(f"\n  Next step: python src/align_baseline.py  (or run the GUI app)")


if __name__ == "__main__":
    main()
