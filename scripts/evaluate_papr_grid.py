"""Run PaPr evaluation across multiple Thunder models and datasets."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _checkpoint_path(ckpt_root, dataset, model, adaptation):
    return ckpt_root / dataset / f"{model}_{adaptation}" / "best_model.pt"


def _read_rows(path, status="ok", message=""):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["status"] = status
        row["grid_message"] = message
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PaPr over a model/dataset grid and aggregate CSV results."
    )
    parser.add_argument("--model-names", nargs="+", required=True,
                        help="Thunder model names, e.g. uni hoptimus0.")
    parser.add_argument("--dataset-names", nargs="+", required=True,
                        help="Thunder dataset names, e.g. crc break_his.")
    parser.add_argument("--base-data-folder", required=True)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--adaptation", default="linear_probing",
                        choices=["linear_probing", "lora", "full", "bitfit"])
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.7, 0.5, 0.3])
    parser.add_argument("--proposal-model", default="mobileone_s0")
    parser.add_argument("--proposal-weights", default=None)
    parser.add_argument("--no-proposal-pretrained", action="store_true")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--far-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default="results/papr/grid.csv")
    parser.add_argument("--parts-dir", default="results/papr/grid_parts")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root)
    output_csv = Path(args.output_csv)
    parts_dir = Path(args.parts_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    script = Path(__file__).resolve().with_name("evaluate_papr.py")
    for dataset in args.dataset_names:
        for model in args.model_names:
            ckpt = _checkpoint_path(ckpt_root, dataset, model, args.adaptation)
            if not ckpt.exists():
                rows.append({
                    "method": "papr",
                    "model": model,
                    "dataset": dataset,
                    "split": args.split,
                    "adaptation": args.adaptation,
                    "keep_ratio": "",
                    "proposal_model": args.proposal_model,
                    "acc": "",
                    "f1_macro": "",
                    "tar_at_far": "",
                    "threshold": "",
                    "ms_per_img": "",
                    "gflops": "",
                    "checkpoint": str(ckpt),
                    "message": "",
                    "status": "checkpoint_missing",
                    "grid_message": f"Missing Phase 1 checkpoint: {ckpt}",
                })
                print(f"[SKIP] {dataset}/{model}: missing {ckpt}")
                continue

            part_csv = parts_dir / f"{dataset}_{model}_{args.adaptation}_{args.split}.csv"
            cmd = [
                sys.executable, str(script),
                "--model-name", model,
                "--dataset-name", dataset,
                "--base-data-folder", args.base_data_folder,
                "--ckpt-root", str(ckpt_root),
                "--adaptation", args.adaptation,
                "--proposal-model", args.proposal_model,
                "--split", args.split,
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--far-threshold", str(args.far_threshold),
                "--seed", str(args.seed),
                "--output-csv", str(part_csv),
                "--keep-ratios", *[str(r) for r in args.keep_ratios],
            ]
            if args.proposal_weights:
                cmd += ["--proposal-weights", args.proposal_weights]
            if args.no_proposal_pretrained:
                cmd.append("--no-proposal-pretrained")
            if args.include_baseline:
                cmd.append("--include-baseline")
            if args.benchmark:
                cmd.append("--benchmark")
            if args.max_samples is not None:
                cmd += ["--max-samples", str(args.max_samples)]

            print(f"[RUN] {dataset}/{model}")
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                rows.append({
                    "method": "papr",
                    "model": model,
                    "dataset": dataset,
                    "split": args.split,
                    "adaptation": args.adaptation,
                    "keep_ratio": "",
                    "proposal_model": args.proposal_model,
                    "acc": "",
                    "f1_macro": "",
                    "tar_at_far": "",
                    "threshold": "",
                    "ms_per_img": "",
                    "gflops": "",
                    "checkpoint": str(ckpt),
                    "message": "",
                    "status": "failed",
                    "grid_message": f"evaluate_papr.py exited with {proc.returncode}",
                })
                if args.fail_fast:
                    break
                continue

            rows.extend(_read_rows(part_csv))
        else:
            continue
        break

    fieldnames = [
        "method", "model", "dataset", "split", "adaptation", "keep_ratio",
        "proposal_model", "acc", "f1_macro", "tar_at_far", "threshold",
        "ms_per_img", "gflops", "checkpoint", "message", "status",
        "grid_message",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved aggregate CSV to: {output_csv}")


if __name__ == "__main__":
    main()
