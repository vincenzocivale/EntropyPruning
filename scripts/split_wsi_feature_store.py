#!/usr/bin/env python
"""Create reproducible train/val/test slide-id splits from a WSI feature store."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wsi import H5WSIFeatureStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible train/val/test slide-id splits."
    )

    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stratify-label",
        action="store_true",
        help="Stratify splits by scalar slide label.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite split files if they already exist.",
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative.")
    if args.train_ratio <= 0.0:
        raise ValueError("--train-ratio must be positive.")
    if sum(ratios) <= 0.0:
        raise ValueError("at least one split ratio must be positive.")

    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "--train-ratio + --val-ratio + --test-ratio must sum to 1.0; "
            f"got {total}."
        )

    output_files = [
        args.output_dir / "train.txt",
        args.output_dir / "val.txt",
        args.output_dir / "test.txt",
        args.output_dir / "split_summary.json",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "split output files already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing[:5])
        )


def _label_to_key(label: Any, *, slide_id: str) -> str:
    if label is None:
        raise ValueError(
            f"slide {slide_id} has no label; cannot use --stratify-label."
        )

    if isinstance(label, torch.Tensor):
        if label.numel() != 1:
            raise ValueError(
                f"slide {slide_id} has non-scalar tensor label; "
                "cannot use --stratify-label."
            )
        return str(label.detach().cpu().item())

    return str(label)


def _target_counts(n_items: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0

    raw = [n_items * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = n_items - sum(counts)

    fractional_order = sorted(
        range(3),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )

    for index in fractional_order[:remainder]:
        counts[index] += 1

    if n_items >= 3 and all(ratio > 0 for ratio in ratios):
        for index in range(3):
            if counts[index] == 0:
                donor = max(range(3), key=lambda donor_index: counts[donor_index])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1

    return counts[0], counts[1], counts[2]


def _split_ids(
    slide_ids: list[str],
    *,
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    shuffled = list(slide_ids)
    rng.shuffle(shuffled)

    n_train, n_val, n_test = _target_counts(len(shuffled), ratios)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val : n_train + n_val + n_test]

    return train_ids, val_ids, test_ids


def _make_unstratified_split(
    slide_ids: tuple[str, ...],
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    return _split_ids(list(slide_ids), ratios=ratios, rng=random.Random(seed))


def _make_stratified_split(
    store: H5WSIFeatureStore,
    slide_ids: tuple[str, ...],
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)

    for slide_id in slide_ids:
        bag = store.read(slide_id)
        grouped[_label_to_key(bag.label, slide_id=slide_id)].append(slide_id)

    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for label in sorted(grouped):
        group_train, group_val, group_test = _split_ids(
            grouped[label],
            ratios=ratios,
            rng=rng,
        )
        train_ids.extend(group_train)
        val_ids.extend(group_val)
        test_ids.extend(group_test)

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return train_ids, val_ids, test_ids


def _write_ids(path: Path, slide_ids: list[str]) -> None:
    path.write_text("".join(f"{slide_id}\n" for slide_id in slide_ids))


def _label_distribution(store: H5WSIFeatureStore, slide_ids: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for slide_id in slide_ids:
        bag = store.read(slide_id)
        label = bag.label
        if isinstance(label, torch.Tensor):
            if label.numel() == 1:
                key = str(label.detach().cpu().item())
            else:
                key = f"tensor_shape_{tuple(label.shape)}"
        else:
            key = "<missing>" if label is None else str(label)
        counts[key] += 1
    return dict(sorted(counts.items()))


def _assert_disjoint(train_ids: list[str], val_ids: list[str], test_ids: list[str]) -> None:
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlaps = {
        "train_val": sorted(train_set.intersection(val_set)),
        "train_test": sorted(train_set.intersection(test_set)),
        "val_test": sorted(val_set.intersection(test_set)),
    }
    overlaps = {key: value for key, value in overlaps.items() if value}
    if overlaps:
        raise AssertionError(f"split overlap detected: {overlaps}")


def main() -> int:
    args = parse_args()

    store = H5WSIFeatureStore(args.feature_store)
    slide_ids = store.slide_ids()
    if not slide_ids:
        raise ValueError(f"feature store contains no slides: {args.feature_store}")

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    if args.stratify_label:
        train_ids, val_ids, test_ids = _make_stratified_split(
            store,
            slide_ids,
            ratios=ratios,
            seed=args.seed,
        )
    else:
        train_ids, val_ids, test_ids = _make_unstratified_split(
            slide_ids,
            ratios=ratios,
            seed=args.seed,
        )

    _assert_disjoint(train_ids, val_ids, test_ids)

    all_split_ids = sorted(train_ids + val_ids + test_ids)
    if all_split_ids != sorted(slide_ids):
        raise AssertionError("split ids do not exactly cover feature store slide ids.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_ids(args.output_dir / "train.txt", train_ids)
    _write_ids(args.output_dir / "val.txt", val_ids)
    _write_ids(args.output_dir / "test.txt", test_ids)

    summary = {
        "feature_store": str(args.feature_store),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "stratify_label": args.stratify_label,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "n_slides": len(slide_ids),
        "splits": {
            "train": {
                "n_slides": len(train_ids),
                "label_distribution": _label_distribution(store, train_ids),
            },
            "val": {
                "n_slides": len(val_ids),
                "label_distribution": _label_distribution(store, val_ids),
            },
            "test": {
                "n_slides": len(test_ids),
                "label_distribution": _label_distribution(store, test_ids),
            },
        },
    }

    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
