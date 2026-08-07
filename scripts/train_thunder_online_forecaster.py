#!/usr/bin/env python3
"""Train EAF online on THUNDER tiles with a frozen foundation-model teacher."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thunder.models.pretrained_models import get_model_from_name

from src.data.thunder_multi import (
    ThunderDatasetRegistry,
    build_multi_thunder_split_loader,
)
from src.models import AttentionForecaster, ThunderBackboneAdapter
from src.training.online_attention_distillation import (
    FrozenTimmAttentionTeacher,
    load_forecaster_checkpoint,
    spearman_correlation,
    topk_recall,
)
from src.utils import get_device, save_results, set_seed


TOPK_FRACTIONS = (0.10, 0.25, 0.50)


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _discover_datasets(base_data_folder: str) -> list[str]:
    splits_dir = Path(base_data_folder) / "data_splits"
    names = sorted(path.stem for path in splits_dir.glob("*.json"))
    if not names:
        raise FileNotFoundError(f"No THUNDER manifests found in {splits_dir}")
    return names


def _validate_names(names: list[str], available: list[str], label: str) -> None:
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label} THUNDER datasets: {unknown}")


def _checkpoint_payload(
    forecaster: AttentionForecaster,
    args: argparse.Namespace,
    train_datasets: list[str],
    eval_datasets: list[str],
    best_val_kl: float,
    epoch: int,
    embed_dim: int,
    target_layer: int,
) -> dict[str, Any]:
    return {
        "forecaster_state_dict": forecaster.state_dict(),
        "format_version": 1,
        "teacher": {
            "model_name": args.model_name,
            "source_layer": args.source_layer,
            "target_layer": target_layer,
            "target_normalization": args.target_normalization,
            "frozen": True,
        },
        "forecaster": {
            "embed_dim": embed_dim,
            "hidden": args.hidden,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        },
        "data": {
            "train_datasets": train_datasets,
            "eval_datasets": eval_datasets,
            "sampler": args.sampler,
            "labels_used_for_training": False,
        },
        "training": {
            "epoch": epoch,
            "best_selection_val_kl": best_val_kl,
            "seed": args.seed,
        },
    }


def _evaluate(
    forecaster: AttentionForecaster,
    teacher: FrozenTimmAttentionTeacher,
    loader,
    dataset_info: dict[int, dict],
    device: torch.device,
    amp: bool,
    description: str,
) -> dict[str, Any]:
    forecaster.eval()
    totals: dict[str, dict[str, float]] = {
        info["name"]: {
            "n": 0.0,
            "kl": 0.0,
            "rho": 0.0,
            **{f"recall_{fraction:.2f}": 0.0 for fraction in TOPK_FRACTIONS},
        }
        for info in dataset_info.values()
    }

    with torch.inference_mode():
        for images, _labels, dataset_idx in tqdm(loader, desc=description, leave=False):
            images = images.to(device, non_blocking=True)
            with _autocast(device, amp):
                embeddings, target = teacher(images)
                logits = forecaster(embeddings)
            logits_float = logits.float()
            target_float = target.float()
            sample_kl = F.kl_div(
                logits_float.log_softmax(-1),
                target_float,
                reduction="none",
            ).sum(-1)
            rho = spearman_correlation(logits_float, target_float)
            recalls = {
                fraction: topk_recall(logits_float, target_float, fraction)
                for fraction in TOPK_FRACTIONS
            }
            dataset_idx_device = dataset_idx.to(device, non_blocking=True)
            for local_idx in dataset_idx.unique().tolist():
                mask = dataset_idx_device == local_idx
                name = dataset_info[int(local_idx)]["name"]
                count = int(mask.sum().item())
                totals[name]["n"] += count
                totals[name]["kl"] += float(sample_kl[mask].sum().item())
                totals[name]["rho"] += float(rho[mask].sum())
                for fraction, values in recalls.items():
                    totals[name][f"recall_{fraction:.2f}"] += float(values[mask].sum())

    per_dataset: dict[str, dict[str, float]] = {}
    overall_n = sum(values["n"] for values in totals.values())
    if overall_n == 0:
        raise RuntimeError(f"No samples evaluated for {description}")
    overall_sums = {
        "kl": 0.0,
        "rho": 0.0,
        **{f"recall_{fraction:.2f}": 0.0 for fraction in TOPK_FRACTIONS},
    }
    for name, values in totals.items():
        n = values.pop("n")
        if n == 0:
            continue
        metrics = {key: value / n for key, value in values.items()}
        metrics["n"] = int(n)
        per_dataset[name] = metrics
        for key in overall_sums:
            overall_sums[key] += values[key]

    overall = {key: value / overall_n for key, value in overall_sums.items()}
    overall["n"] = int(overall_n)
    return {"overall": overall, "per_dataset": per_dataset}


def _train_one_epoch(
    forecaster: AttentionForecaster,
    teacher: FrozenTimmAttentionTeacher,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    max_steps: int | None,
    epoch: int,
) -> float:
    forecaster.train()
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=f"epoch {epoch:03d} train", leave=False)
    for step, (images, _labels, _dataset_idx) in enumerate(progress):
        if max_steps is not None and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        with _autocast(device, amp):
            embeddings, target = teacher(images)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = forecaster(embeddings)
            loss = F.kl_div(
                logits.log_softmax(-1), target, reduction="batchmean"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
        progress.set_postfix(kl=f"{loss_sum / max(sample_count, 1):.4f}")
    if sample_count == 0:
        raise RuntimeError("Training loader produced zero samples")
    return loss_sum / sample_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Online task-agnostic EAF training on THUNDER train splits. "
            "The tile encoder is frozen and labels are ignored."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-data-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="thunder_online")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--train-datasets", nargs="+", default=None)
    selection.add_argument("--holdout-plan", default=None)
    parser.add_argument("--n-holdout", type=int, default=0)
    parser.add_argument("--holdout-datasets", nargs="+", default=None)
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=None,
        help="Defaults to every downloaded THUNDER dataset.",
    )

    parser.add_argument("--source-layer", type=int, default=2)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument(
        "--target-normalization", choices=["patch", "none"], default="patch"
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sampler",
        choices=["dataset_balanced", "proportional"],
        default="dataset_balanced",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="External-only EAF checkpoint for evaluation or THUNDER continuation.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_only and not args.init_checkpoint:
        raise ValueError("--eval-only requires --init-checkpoint")
    if not args.eval_only and args.epochs <= 0:
        raise ValueError("--epochs must be positive during training")
    if args.n_holdout < 0:
        raise ValueError("--n-holdout must be non-negative")
    if args.train_datasets is not None and args.holdout_datasets is not None:
        raise ValueError("--holdout-datasets cannot be combined with --train-datasets")

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    available = _discover_datasets(args.base_data_folder)
    if args.train_datasets is not None:
        train_datasets = list(args.train_datasets)
        holdout_datasets: list[str] = []
    elif args.holdout_plan:
        registry = ThunderDatasetRegistry.from_plan(
            args.holdout_plan, args.base_data_folder
        )
        train_datasets = registry.train_datasets
        holdout_datasets = registry.holdout_datasets
    else:
        registry = ThunderDatasetRegistry(
            args.base_data_folder,
            n_holdout=args.n_holdout,
            holdout_datasets=args.holdout_datasets,
        )
        train_datasets = registry.train_datasets
        holdout_datasets = registry.holdout_datasets
    eval_datasets = list(args.eval_datasets or available)
    _validate_names(train_datasets, available, "training")
    _validate_names(eval_datasets, available, "evaluation")
    if not args.eval_only and not train_datasets:
        raise ValueError("No THUNDER datasets selected for training")

    data_plan = {
        "available_datasets": available,
        "train_datasets": train_datasets,
        "holdout_datasets": holdout_datasets,
        "eval_datasets": eval_datasets,
        "labels_used_for_training": False,
        "teacher_frozen": True,
    }
    (output_dir / "data_plan.json").write_text(json.dumps(data_plan, indent=2))

    raw_backbone, transform, _ = get_model_from_name(args.model_name, str(device))
    adapter = ThunderBackboneAdapter(raw_backbone)
    target_layer = (
        args.target_layer if args.target_layer is not None else adapter.n_blocks - 1
    )
    teacher = FrozenTimmAttentionTeacher(
        raw_backbone,
        adapter,
        source_layer=args.source_layer,
        target_layer=target_layer,
        target_normalization=args.target_normalization,
    ).to(device)
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    init_metadata = None
    if args.init_checkpoint:
        init_metadata = load_forecaster_checkpoint(forecaster, args.init_checkpoint)

    eval_val_loader, eval_val_info = build_multi_thunder_split_loader(
        eval_datasets,
        args.base_data_folder,
        transform,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler_mode="proportional",
        seed=args.seed,
    )
    test_loader, test_info = build_multi_thunder_split_loader(
        eval_datasets,
        args.base_data_folder,
        transform,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler_mode="proportional",
        seed=args.seed,
    )

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config={**vars(args), **data_plan, "target_layer_resolved": target_layer},
            tags=["eaf", "thunder", "online", "frozen_teacher"],
        )

    best_checkpoint = output_dir / "best_forecaster.pt"
    best_val_kl = float("inf")
    history: list[dict[str, float]] = []

    if not args.eval_only:
        train_loader, _ = build_multi_thunder_split_loader(
            train_datasets,
            args.base_data_folder,
            transform,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler_mode=args.sampler,
            drop_last=True,
            seed=args.seed,
        )
        selection_loader, selection_info = build_multi_thunder_split_loader(
            train_datasets,
            args.base_data_folder,
            transform,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler_mode="proportional",
            seed=args.seed,
        )
        optimizer = torch.optim.AdamW(
            forecaster.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1)
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(device.type == "cuda" and args.amp)
        )

        for epoch in range(1, args.epochs + 1):
            train_kl = _train_one_epoch(
                forecaster,
                teacher,
                train_loader,
                optimizer,
                scaler,
                device,
                args.amp,
                args.max_steps_per_epoch,
                epoch,
            )
            selection_metrics = _evaluate(
                forecaster,
                teacher,
                selection_loader,
                selection_info,
                device,
                args.amp,
                "selection val",
            )
            selection_kl = sum(
                metrics["kl"]
                for metrics in selection_metrics["per_dataset"].values()
            ) / len(selection_metrics["per_dataset"])
            selection_rho = sum(
                metrics["rho"]
                for metrics in selection_metrics["per_dataset"].values()
            ) / len(selection_metrics["per_dataset"])
            epoch_record = {
                "epoch": epoch,
                "train_kl": train_kl,
                "selection_val_macro_kl": selection_kl,
                "selection_val_macro_rho": selection_rho,
                "selection_val_micro_kl": selection_metrics["overall"]["kl"],
                "selection_val_micro_rho": selection_metrics["overall"]["rho"],
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(epoch_record)
            if wandb_run is not None:
                wandb_run.log(epoch_record)
            if selection_kl < best_val_kl:
                best_val_kl = selection_kl
                torch.save(
                    _checkpoint_payload(
                        forecaster,
                        args,
                        train_datasets,
                        eval_datasets,
                        best_val_kl,
                        epoch,
                        adapter.embed_dim,
                        target_layer,
                    ),
                    best_checkpoint,
                )
            scheduler.step()
            print(
                f"epoch={epoch:03d} train_kl={train_kl:.6f} "
                f"selection_val_macro_kl={selection_kl:.6f} "
                f"selection_val_macro_rho={selection_rho:.4f}"
            )
        load_forecaster_checkpoint(forecaster, best_checkpoint)
    else:
        best_checkpoint = Path(args.init_checkpoint)

    val_metrics = _evaluate(
        forecaster,
        teacher,
        eval_val_loader,
        eval_val_info,
        device,
        args.amp,
        "report val",
    )
    test_metrics = _evaluate(
        forecaster,
        teacher,
        test_loader,
        test_info,
        device,
        args.amp,
        "report test",
    )
    results = {
        "experiment_name": args.experiment_name,
        "mode": "eval_only" if args.eval_only else "train",
        "checkpoint": str(best_checkpoint),
        "init_checkpoint": args.init_checkpoint,
        "init_checkpoint_metadata": init_metadata,
        "teacher": {
            "model_name": args.model_name,
            "source_layer": args.source_layer,
            "target_layer": target_layer,
            "target_normalization": args.target_normalization,
            "frozen": True,
        },
        "data_plan": data_plan,
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
    }
    results_path = save_results(output_dir / "results.json", results)
    print(json.dumps({
        "results": str(results_path),
        "test_overall": test_metrics["overall"],
    }, indent=2))

    if wandb_run is not None:
        wandb_run.log({
            "test/kl": test_metrics["overall"]["kl"],
            "test/rho": test_metrics["overall"]["rho"],
            "test/recall_0.10": test_metrics["overall"]["recall_0.10"],
            "test/recall_0.50": test_metrics["overall"]["recall_0.50"],
        })
        wandb_run.finish()
    teacher.close()


if __name__ == "__main__":
    main()
