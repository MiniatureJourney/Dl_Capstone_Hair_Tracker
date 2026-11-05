"""
split_data.py
-------------
Splits the preprocessed CelebAMask-HQ data into
train / validation / test sets using a fixed random seed.

Split ratio: 80% train | 10% validation | 10% test (as per proposal)

Reads : data/processed/valid_ids.txt
Writes: data/splits/train.txt, val.txt, test.txt

Usage:
    python src/split_data.py \
        --processed_dir data/processed \
        --splits_dir    data/splits \
        --seed          42
"""

import os
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Val/Test split for hair segmentation")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--splits_dir",    type=str, default="data/splits")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--train_ratio",   type=float, default=0.80)
    parser.add_argument("--val_ratio",     type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.splits_dir, exist_ok=True)

    valid_ids_path = os.path.join(args.processed_dir, "valid_ids.txt")
    if not os.path.exists(valid_ids_path):
        raise FileNotFoundError(
            f"'{valid_ids_path}' not found.\n"
            "Run preprocess.py first to generate valid_ids.txt"
        )

    with open(valid_ids_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    total = len(ids)
    print(f"\n{'='*55}")
    print(f"  Dataset Split")
    print(f"  Total valid samples : {total:,}")
    print(f"  Seed                : {args.seed}")
    print(f"{'='*55}")

    random.seed(args.seed)
    random.shuffle(ids)

    n_train = int(total * args.train_ratio)
    n_val   = int(total * args.val_ratio)
    n_test  = total - n_train - n_val

    train_ids = ids[:n_train]
    val_ids   = ids[n_train : n_train + n_val]
    test_ids  = ids[n_train + n_val :]

    splits = {
        "train.txt": train_ids,
        "val.txt":   val_ids,
        "test.txt":  test_ids,
    }

    for filename, split_ids in splits.items():
        path = os.path.join(args.splits_dir, filename)
        with open(path, "w") as f:
            f.write("\n".join(split_ids))
        print(f"  OK   {filename:<12}  {len(split_ids):>6,} samples  ->  {path}")

    print(f"{'='*55}")
    print(f"\n  Train : {len(train_ids):,}  ({len(train_ids)/total*100:.1f}%)")
    print(f"  Val   : {len(val_ids):,}  ({len(val_ids)/total*100:.1f}%)")
    print(f"  Test  : {len(test_ids):,}  ({len(test_ids)/total*100:.1f}%)")
    print(f"\nNext step: python src/train.py")


if __name__ == "__main__":
    main()
