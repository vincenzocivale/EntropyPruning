"""Evaluate PaPr pruning on EAF/Thunder FM classifiers without retraining."""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.data.thunder_loaders import build_thunder_loaders
from src.evaluation import benchmark_model, evaluate
from src.models import (
    PaPrPrunedClassifier,
    STRATEGIES,
    ThunderBackboneAdapter,
    build_classifier,
    build_papr_proposal,
)
from src.utils import get_device, set_seed


def _limit_loader(loader, max_samples):
    if max_samples is None:
        return loader
    if max_samples <= 0:
        raise ValueError(f"--max-samples must be positive, got {max_samples}")
    n = min(max_samples, len(loader.dataset))
    dataset = Subset(loader.dataset, range(n))
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.persistent_workers,
    )


def _default_classifier_ckpt(ckpt_root, dataset_name, model_name, adaptation):
    return ckpt_root / dataset_name / f"{model_name}_{adaptation}" / "best_model.pt"


def _load_phase1_classifier(
    model_name,
    dataset_name,
    base_data_folder,
    adaptation,
    classifier_ckpt,
    batch_size,
    num_workers,
    dropout,
    lora_r,
    lora_alpha,
    device,
):
    raw_backbone, transform, _ = get_model_from_name(model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    train_loader, val_loader, test_loader, class_names, n_classes = build_thunder_loaders(
        dataset_name,
        base_data_folder,
        transform,
        batch_size,
        num_workers,
        drop_last_train=False,
    )

    kwargs = {"dropout": dropout}
    if adaptation == "lora":
        kwargs.update(lora_r=lora_r, lora_alpha=lora_alpha)
    classifier = build_classifier(adaptation, raw_backbone, adapter, n_classes, **kwargs)
    state = torch.load(classifier_ckpt, map_location=device)
    missing, unexpected = classifier.load_state_dict(state, strict=False)
    classifier.to(device).eval()

    return {
        "classifier": classifier,
        "adapter": adapter,
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
        "class_names": class_names,
        "n_classes": n_classes,
        "missing": missing,
        "unexpected": unexpected,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only PaPr evaluation for Thunder FM classifiers. "
            "This follows PaPr's pipeline: frozen pretrained ConvNet proposal, "
            "one-step early token pruning, no fine-tuning."
        )
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--base-data-folder", type=str, required=True)
    parser.add_argument("--adaptation", type=str, default="lora", choices=STRATEGIES)
    parser.add_argument("--classifier-ckpt", type=str, default=None)
    parser.add_argument("--ckpt-root", type=str, default="checkpoints")
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.7, 0.5, 0.3])
    parser.add_argument("--proposal-model", type=str, default="mobileone_s0",
                        help=(
                            "Frozen proposal ConvNet. PaPr uses mobileone_s0 "
                            "by default; resnet18/resnet50 are supported."
                        ))
    parser.add_argument("--proposal-weights", type=str, default=None,
                        help="Optional local checkpoint for the proposal ConvNet.")
    parser.add_argument("--proposal-pretrained", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use pretrained proposal weights when no --proposal-weights is provided.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap for quick smoke evaluations.")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Also evaluate the unpruned Phase 1 classifier.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure latency/FLOPs in addition to metrics.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ckpt_root = Path(args.ckpt_root)
    classifier_ckpt = Path(args.classifier_ckpt) if args.classifier_ckpt else (
        _default_classifier_ckpt(
            ckpt_root, args.dataset_name, args.model_name, args.adaptation,
        )
    )
    if not classifier_ckpt.exists():
        raise FileNotFoundError(f"Phase 1 classifier checkpoint not found: {classifier_ckpt}")

    loaded = _load_phase1_classifier(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        base_data_folder=args.base_data_folder,
        adaptation=args.adaptation,
        classifier_ckpt=classifier_ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=device,
    )

    classifier = loaded["classifier"]
    adapter = loaded["adapter"]
    eval_loader = _limit_loader(loaded["loaders"][args.split], args.max_samples)

    print(f"Device: {device} | Model: {args.model_name} | Dataset: {args.dataset_name}")
    print(f"Phase 1 checkpoint: {classifier_ckpt}")
    print(
        f"Checkpoint load: missing={len(loaded['missing'])} "
        f"unexpected={len(loaded['unexpected'])}"
    )
    print(f"embed_dim={adapter.embed_dim} n_blocks={adapter.n_blocks} "
          f"n_patches={adapter.n_patches} prefix={adapter.num_prefix_tokens}")
    print(f"Eval split: {args.split} | samples: {len(eval_loader.dataset)}")
    print(f"Proposal: {args.proposal_model} pretrained={args.proposal_pretrained} "
          f"weights={args.proposal_weights}")

    proposal = build_papr_proposal(
        proposal_model=args.proposal_model,
        pretrained=args.proposal_pretrained,
        weights_path=args.proposal_weights,
    ).to(device)

    rows = []
    if args.include_baseline:
        metrics = evaluate(classifier, eval_loader, device, args.far_threshold)
        bench = (
            benchmark_model(classifier, eval_loader, device, label="baseline")
            if args.benchmark else {"ms_per_img": None, "gflops": None}
        )
        rows.append({
            "method": "baseline",
            "model": args.model_name,
            "dataset": args.dataset_name,
            "split": args.split,
            "adaptation": args.adaptation,
            "keep_ratio": 1.0,
            "proposal_model": "",
            "acc": float(metrics["acc"]),
            "f1_macro": float(metrics["f1_macro"]),
            "tar_at_far": float(metrics["tar_at_far"]),
            "threshold": float(metrics["threshold"]),
            "ms_per_img": bench["ms_per_img"],
            "gflops": bench["gflops"],
            "checkpoint": str(classifier_ckpt),
            "message": "unpruned",
        })
        print(f"[baseline] acc={metrics['acc']:.4f} f1={metrics['f1_macro']:.4f}")

    papr_model = PaPrPrunedClassifier(
        classifier=classifier,
        adapter=adapter,
        proposal=proposal,
        keep_ratio=args.keep_ratios[0],
    ).to(device)
    papr_model.eval()

    for keep_ratio in args.keep_ratios:
        papr_model.keep_ratio = keep_ratio
        metrics = evaluate(papr_model, eval_loader, device, args.far_threshold)
        bench = (
            benchmark_model(
                papr_model,
                eval_loader,
                device,
                label=f"PaPr keep={int(round(keep_ratio * 100))}%",
            )
            if args.benchmark else {"ms_per_img": None, "gflops": None}
        )
        row = {
            "method": "papr",
            "model": args.model_name,
            "dataset": args.dataset_name,
            "split": args.split,
            "adaptation": args.adaptation,
            "keep_ratio": keep_ratio,
            "proposal_model": args.proposal_model,
            "acc": float(metrics["acc"]),
            "f1_macro": float(metrics["f1_macro"]),
            "tar_at_far": float(metrics["tar_at_far"]),
            "threshold": float(metrics["threshold"]),
            "ms_per_img": bench["ms_per_img"],
            "gflops": bench["gflops"],
            "checkpoint": str(classifier_ckpt),
            "message": (
                f"prefix_preserved={adapter.num_prefix_tokens}; "
                f"spatial_tokens_kept={max(1, int(adapter.n_patches * keep_ratio))}"
            ),
        }
        rows.append(row)
        print(
            f"[PaPr keep={keep_ratio:.2f}] acc={metrics['acc']:.4f} "
            f"f1={metrics['f1_macro']:.4f}"
        )

    output_csv = Path(args.output_csv) if args.output_csv else Path(
        "results" / "papr" /
        f"{args.dataset_name}_{args.model_name}_{args.adaptation}_{args.split}.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "model", "dataset", "split", "adaptation", "keep_ratio",
        "proposal_model", "acc", "f1_macro", "tar_at_far", "threshold",
        "ms_per_img", "gflops", "checkpoint", "message",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()
