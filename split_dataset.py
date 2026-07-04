"""Split a labeled CSV into train, validation, and test sets for prompt tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified train/validation/test CSV files.")
    parser.add_argument("--input", required=True, help="Input CSV containing text and labels.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _allocate_counts(n_items: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.array(ratios, dtype=float) * n_items
    counts = np.floor(raw).astype(int)
    remainder = n_items - int(counts.sum())
    for idx in np.argsort(raw - counts)[::-1][:remainder]:
        counts[idx] += 1
    if n_items >= 3:
        for idx in range(3):
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def stratified_split(df: pd.DataFrame, label_col: str, ratios: tuple[float, float, float], seed: int):
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(label_col, sort=False):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000_000)))
        n_train, n_valid, n_test = _allocate_counts(len(shuffled), ratios)
        train_parts.append(shuffled.iloc[:n_train])
        valid_parts.append(shuffled.iloc[n_train : n_train + n_valid])
        test_parts.append(shuffled.iloc[n_train + n_valid : n_train + n_valid + n_test])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid_df = pd.concat(valid_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return train_df, valid_df, test_df


def main() -> int:
    args = parse_args()
    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("--train-ratio + --valid-ratio + --test-ratio must equal 1.0.")

    df = pd.read_csv(args.input, encoding=args.encoding)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, valid_df, test_df = stratified_split(df, args.label_col, ratios, args.seed)

    train_df.to_csv(output_dir / "train.csv", index=False, encoding=args.encoding)
    valid_df.to_csv(output_dir / "valid.csv", index=False, encoding=args.encoding)
    test_df.to_csv(output_dir / "test.csv", index=False, encoding=args.encoding)

    summary = {
        "total": len(df),
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
        "ratios": {"train": args.train_ratio, "valid": args.valid_ratio, "test": args.test_ratio},
        "label_distribution": {
            "train": train_df[args.label_col].value_counts().to_dict(),
            "valid": valid_df[args.label_col].value_counts().to_dict(),
            "test": test_df[args.label_col].value_counts().to_dict(),
        },
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
