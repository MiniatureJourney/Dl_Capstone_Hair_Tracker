"""
train.py
--------
Full training script for U-Net binary hair segmentation.

Features:
  - GPU (CUDA) support for RTX 5060 (Blackwell, CUDA 12.8)
  - BCE + Dice combined loss
  - Adam optimizer with ReduceLROnPlateau scheduler
  - Early stopping (patience=10)
  - Model checkpointing (saves best val Dice)
  - Training logs saved to CSV
  - Loss and metric plots saved automatically

Usage:
    python src/train.py \
        --data_dir   data/processed \
        --splits_dir data/splits \
        --output_dir outputs \
        --epochs     50 \
        --batch_size 8 \
        --lr         1e-4
"""

import os
import sys
import argparse
import time
import csv

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from dataset import build_dataloaders
from model   import UNet, count_parameters
from loss    import BCEDiceLoss
from metrics import compute_all_metrics
from utils   import save_sample_predictions, plot_curves


# ----------------------------------------------
#  Argument parser
# ----------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for hair segmentation")
    parser.add_argument("--data_dir",   type=str,   default="data/processed")
    parser.add_argument("--splits_dir", type=str,   default="data/splits")
    parser.add_argument("--output_dir", type=str,   default="outputs")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--num_workers",type=int,   default=0,
                        help="DataLoader workers. Use 0 on Windows to avoid multiprocessing issues.")
    parser.add_argument("--img_size",   type=int,   default=256)
    return parser.parse_args()


# ----------------------------------------------
#  One training epoch
# ----------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ----------------------------------------------
#  One validation epoch
# ----------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    agg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits  = model(images)
        loss    = criterion(logits, masks)
        metrics = compute_all_metrics(logits, masks)

        total_loss += loss.item()
        for k in agg:
            agg[k] += metrics[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in agg.items()}


# ----------------------------------------------
#  Main training loop
# ----------------------------------------------

def main():
    args = parse_args()

    # -- Paths ----------------------------------
    model_dir = os.path.join(args.output_dir, "models")
    plot_dir  = os.path.join(args.output_dir, "plots")
    pred_dir  = os.path.join(args.output_dir, "plots", "sample_predictions")
    log_path  = os.path.join(args.output_dir, "training_log.csv")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir,  exist_ok=True)

    # -- Device ---------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  U-Net Hair Segmentation ? Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Patience    : {args.patience}")

    # -- DataLoaders ----------------------------
    print(f"\n  [DataLoaders]")
    loaders = build_dataloaders(
        processed_dir=args.data_dir,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.img_size,
    )

    # -- Model ----------------------------------
    model = UNet(in_channels=3, out_channels=1).to(device)
    print(f"\n  [Model] U-Net | Parameters: {count_parameters(model):,}")

    # -- Loss, Optimizer, Scheduler -------------
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5)

    # -- Log file header ------------------------
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "val_dice", "val_iou", "val_precision", "val_recall", "lr"])

    # -- Training state -------------------------
    best_dice       = 0.0
    no_improve_cnt  = 0
    history = {k: [] for k in
               ["train_loss", "val_loss", "val_dice", "val_iou",
                "val_precision", "val_recall"]}

    print(f"\n  {'-'*58}")
    print(f"  Starting training...")
    print(f"  {'-'*58}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch)

        # Validate
        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device)

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        # -- Print epoch summary -----------------
        print(
            f"  Epoch [{epoch:>3}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"LR: {cur_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # -- Log to CSV --------------------------
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_metrics['dice']:.6f}", f"{val_metrics['iou']:.6f}",
                f"{val_metrics['precision']:.6f}", f"{val_metrics['recall']:.6f}",
                f"{cur_lr:.2e}"
            ])

        # -- History -----------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in ["dice", "iou", "precision", "recall"]:
            history[f"val_{k}"].append(val_metrics[k])

        # -- LR scheduler step -------------------
        scheduler.step(val_loss)

        # -- Checkpoint: best model ---------------
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            no_improve_cnt = 0
            best_path = os.path.join(model_dir, "best_model.pth")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice":   best_dice,
                "val_iou":    val_metrics["iou"],
                "val_loss":   val_loss,
                "train_loss": train_loss,
                "args":       vars(args),
            }, best_path)
            print(f"  OK   Best model saved -> val Dice: {best_dice:.4f}")
        else:
            no_improve_cnt += 1

        # -- Periodic checkpoint (every 10 epochs) -
        if epoch % 10 == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_path)

        # -- Save sample predictions every 5 epochs -
        if epoch % 5 == 0:
            save_sample_predictions(
                model, loaders["val"], device,
                save_dir=pred_dir, epoch=epoch, n_samples=6
            )

        # -- Early stopping -----------------------
        if no_improve_cnt >= args.patience:
            print(f"\n  STOP  Early stopping triggered after {epoch} epochs")
            print(f"      No improvement in {args.patience} consecutive epochs.")
            break

    # -- Final summary --------------------------
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Val Dice : {best_dice:.4f}")
    print(f"  Best model    : outputs/models/best_model.pth")
    print(f"  Training log  : {log_path}")
    print(f"{'='*60}")

    # -- Plot curves ----------------------------
    plot_curves(history, save_dir=plot_dir)
    print(f"  Curves saved  : {plot_dir}/")
    print(f"\n  Next step: python src/evaluate.py")


if __name__ == "__main__":
    main()
