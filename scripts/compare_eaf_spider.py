"""Compare per-dataset vs universal EAF forecasters.

Data sources
------------
--universal-json   One or more JSON files from train_forecaster_unsupervised.py.
                   Provides per-dataset rho_forecaster, rho_token_norm.
                   Multiple files produce one series each (e.g. h=256 vs h=512).

--per-dataset-dir  Directory produced by train_per_dataset_eaf.py.
                   Script discovers checkpoints matching
                   {dir}/{dataset}/forecaster_src{L:02d}_attn{T:02d}.pt,
                   re-evaluates them on the HDF5 test caches, and adds a series.

Output  (results/ablations/{model_name}/)
-------
    rho_spider.png    polar radar — all series + baseline
    rho_table.csv     numeric table  dataset × series

Usage
-----
# Universal JSON only
python scripts/compare_eaf_spider.py \\
    --model-name uni \\
    --cache-dir checkpoints/unsupervised \\
    --universal-json checkpoints/unsupervised/uni_forecaster/results_forecaster_uni_src02_attn23_universal.json

# Full comparison (per-dataset re-evaluation + universal JSON)
python scripts/compare_eaf_spider.py \\
    --model-name uni \\
    --cache-dir checkpoints/unsupervised \\
    --per-dataset-dir checkpoints/unsupervised/per_dataset \\
    --layers-source 2 \\
    --layer-target 23 \\
    --universal-json \\
        checkpoints/unsupervised/uni_forecaster/results_forecaster_uni_src02_attn23_universal.json \\
        checkpoints/unsupervised/uni_forecaster_h512/results_forecaster_uni_src02_attn23_universal.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import AttentionForecaster
from src.data import MultiH5ForecastDataset


# ──────────────────────────────────────────────────────────────────────────────
# Palette  (color, linestyle, linewidth, fill-alpha)
# ──────────────────────────────────────────────────────────────────────────────
_BASELINE = dict(color="#9E9E9E", ls="--", lw=1.4, alpha=0.10, label="token-norm baseline")
_SERIES_STYLES = [
    dict(color="#1565C0", ls="-",  lw=2.2, alpha=0.15),   # blue   — per-dataset
    dict(color="#E65100", ls="-",  lw=2.2, alpha=0.15),   # orange — universal #1
    dict(color="#2E7D32", ls="-",  lw=2.2, alpha=0.15),   # green  — universal #2
    dict(color="#6A1B9A", ls="-",  lw=2.2, alpha=0.15),   # purple — universal #3
    dict(color="#00838F", ls="-",  lw=2.2, alpha=0.15),   # teal   — universal #4
]


# ──────────────────────────────────────────────────────────────────────────────
# Spearman
# ──────────────────────────────────────────────────────────────────────────────
def _spearman(y_pred, y_true):
    def _rank(t):
        return t.argsort(dim=-1).argsort(dim=-1).float()
    rp = _rank(y_pred) - _rank(y_pred).mean(-1, keepdim=True)
    rt = _rank(y_true) - _rank(y_true).mean(-1, keepdim=True)
    return (rp * rt).sum(-1) / (
        torch.sqrt((rp**2).sum(-1) * (rt**2).sum(-1)) + 1e-8
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────
def _load_universal_json(path):
    """Return (rho_forecaster, rho_baseline, label) dicts from a results JSON."""
    with open(path) as f:
        obj = json.load(f)
    per_ds = obj.get("per_dataset", {})
    rho_fc = {k: v["rho_forecaster"] for k, v in per_ds.items()}
    rho_bl = {k: v["rho_token_norm"]  for k, v in per_ds.items()}

    layers = obj.get("layers_source", obj.get("layer_source", "?"))
    src = "+".join(map(str, layers)) if isinstance(layers, list) else str(layers)
    hidden = obj.get("hidden", "?")
    label = f"universal  src={src}  h={hidden}"
    return rho_fc, rho_bl, label


@torch.no_grad()
def _eval_per_dataset_ckpts(
    ckpt_dir, cache_dir, model_name, layers_source, layer_target,
    n_heads, batch_size, num_workers, device,
):
    """
    Re-evaluate every per-dataset checkpoint found in ckpt_dir.
    Returns {dataset: rho} for datasets whose checkpoint exists.
    """
    ckpt_dir  = Path(ckpt_dir)
    cache_dir = Path(cache_dir)

    caches = {
        p.stem.replace(f"_{model_name}_attn_features", ""): p
        for p in sorted(cache_dir.glob(f"*_{model_name}_attn_features.h5"))
    }
    if not caches:
        print(f"[warn] No HDF5 caches found in {cache_dir} for model {model_name}")
        return {}

    results = {}
    for dataset, h5_path in caches.items():
        ckpt = ckpt_dir / dataset / f"forecaster_src{layers_source[0]:02d}_attn{layer_target:02d}.pt"
        if not ckpt.exists():
            print(f"  [{dataset}] checkpoint not found: {ckpt} — skip")
            continue

        state = torch.load(ckpt, map_location=device, weights_only=True)
        embed_dim = state["input_proj.weight"].shape[1]
        hidden    = state["input_proj.weight"].shape[0]
        n_layers  = sum(1 for k in state
                        if k.startswith("self_attn.") and k.endswith(".norm1.weight"))

        forecaster = AttentionForecaster(
            embed_dim=embed_dim, hidden=hidden,
            n_heads=n_heads, n_layers=max(n_layers, 1), dropout=0.0,
        ).to(device)
        forecaster.load_state_dict(state)
        forecaster.eval()

        ds = MultiH5ForecastDataset({dataset: h5_path}, "test", layers_source, layer_target)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0))

        rhos = []
        for emb, target, _, _ in loader:
            emb, target = emb.to(device), target.to(device)
            rhos.append(_spearman(forecaster(emb), target).cpu())
        results[dataset] = torch.cat(rhos).mean().item()
        print(f"  [{dataset}] per-dataset rho = {results[dataset]:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Spider plot
# ──────────────────────────────────────────────────────────────────────────────
def _spider(series, baseline, datasets, title, out_path):
    """Polar radar chart."""
    N      = len(datasets)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ap     = np.append(angles, angles[0])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    if baseline:
        bv = np.array([baseline.get(d, np.nan) for d in datasets])
        bv_p = np.append(bv, bv[0])
        ax.plot(ap, bv_p, color=_BASELINE["color"], ls=_BASELINE["ls"],
                lw=_BASELINE["lw"], label=_BASELINE["label"])

    for (label, rho_dict), style in zip(series, _SERIES_STYLES):
        v  = np.array([rho_dict.get(d, np.nan) for d in datasets])
        vp = np.append(v, v[0])
        prev = None
        for i in range(len(vp)):
            if np.isnan(vp[i]):
                prev = None; continue
            if prev is not None:
                ax.plot(ap[prev:i+1], vp[prev:i+1],
                        color=style["color"], ls=style["ls"], lw=style["lw"])
            prev = i
        if not np.isnan(v).any():
            ax.fill(ap, vp, alpha=style["alpha"], color=style["color"])
        ax.plot([], [], color=style["color"], ls=style["ls"], lw=style["lw"], label=label)

    all_vals = [v for _, d in series for v in d.values() if not np.isnan(v)]
    if baseline:
        all_vals += [v for v in baseline.values() if not np.isnan(v)]
    lo = max(0.0, min(all_vals) - 0.05) if all_vals else 0.0
    ax.set_ylim(lo, 1.0)
    ax.set_yticks(np.round(np.linspace(lo, 1.0, 5), 2))
    ax.set_rlabel_position(20)
    ax.set_xticks(angles)
    ax.set_xticklabels(datasets, size=9)
    ax.set_title(title, size=12, pad=24, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.18), fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  spider  → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CSV table
# ──────────────────────────────────────────────────────────────────────────────
def _save_csv(series, baseline, datasets, out_path):
    cols = ["dataset"] + [label for label, _ in series]
    if baseline:
        cols += ["token_norm_baseline"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ds in datasets:
            row = [ds]
            for _, rho_dict in series:
                row.append(round(rho_dict.get(ds, float("nan")), 6))
            if baseline:
                row.append(round(baseline.get(ds, float("nan")), 6))
            w.writerow(row)

        avg_row = ["avg"]
        for _, rho_dict in series:
            vals = [v for d in datasets for v in [rho_dict.get(d)] if v is not None]
            avg_row.append(round(float(np.nanmean(vals)), 6) if vals else "")
        if baseline:
            bl_vals = [baseline.get(d) for d in datasets if baseline.get(d) is not None]
            avg_row.append(round(float(np.nanmean(bl_vals)), 6) if bl_vals else "")
        w.writerow(avg_row)

    print(f"  table   → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Stdout summary
# ──────────────────────────────────────────────────────────────────────────────
def _print_summary(series, baseline, datasets):
    cols = [label for label, _ in series]
    if baseline:
        cols += ["baseline"]
    w = max(len(c) for c in cols + ["dataset"]) + 2
    header = f"{'dataset':>{max(25, w)}}" + "".join(f"{c:>{w}}" for c in cols)
    sep    = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for ds in datasets:
        row = f"{ds:>{max(25, w)}}"
        for _, rho in series:
            v = rho.get(ds, float("nan"))
            row += f"{v:>{w}.4f}"
        if baseline:
            bl = baseline.get(ds, float("nan"))
            row += f"{bl:>{w}.4f}"
        print(row)
    print(sep)
    avg_row = f"{'avg':>{max(25, w)}}"
    for _, rho in series:
        vals = [v for v in [rho.get(d) for d in datasets] if v is not None]
        avg_row += f"{np.nanmean(vals):>{w}.4f}" if vals else f"{'—':>{w}}"
    if baseline:
        bl_vals = [baseline.get(d) for d in datasets if baseline.get(d) is not None]
        avg_row += f"{np.nanmean(bl_vals):>{w}.4f}" if bl_vals else f"{'—':>{w}}"
    print(avg_row)
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Compare per-dataset and universal EAF forecasters — spider plot + CSV"
    )
    ap.add_argument("--model-name",         type=str, required=True)
    ap.add_argument("--cache-dir",          type=str, default="checkpoints/unsupervised",
                    help="Directory with *_{model}_attn_features.h5 caches.")
    ap.add_argument("--universal-json",     type=str, nargs="*", default=None,
                    help="JSON file(s) from train_forecaster_unsupervised.py.")
    ap.add_argument("--per-dataset-dir",    type=str, default=None,
                    help="Directory of per-dataset checkpoints "
                         "({dir}/{dataset}/forecaster_src{L:02d}_attn{T:02d}.pt). "
                         "Checkpoints are re-evaluated on the HDF5 test caches.")
    ap.add_argument("--layers-source",      type=int, nargs="+", default=[2],
                    help="Source layer(s) — must match the per-dataset checkpoints.")
    ap.add_argument("--layer-target",       type=int, default=23,
                    help="Target layer — must match the per-dataset checkpoints.")
    ap.add_argument("--forecaster-n-heads", type=int, default=4)
    ap.add_argument("--batch-size",         type=int, default=256)
    ap.add_argument("--num-workers",        type=int, default=4)
    ap.add_argument("--datasets",           type=str, nargs="+", default=None,
                    help="Explicit dataset list and order. "
                         "Default: union of all sources, sorted.")
    ap.add_argument("--no-baseline",        action="store_true",
                    help="Omit the token-norm baseline line.")
    ap.add_argument("--out-dir",            type=str, default="results/ablations",
                    help="Output directory (default: results/ablations).")
    args = ap.parse_args()

    if not args.universal_json and not args.per_dataset_dir:
        ap.error("Provide at least one of --universal-json or --per-dataset-dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model_name}")

    # ── collect series ────────────────────────────────────────────────────────
    series   = []   # [(label, {dataset: rho}), ...]
    baseline = {}

    if args.per_dataset_dir:
        print("\n[per-dataset] Re-evaluating checkpoints …")
        pd_rho = _eval_per_dataset_ckpts(
            args.per_dataset_dir, args.cache_dir, args.model_name,
            args.layers_source, args.layer_target,
            args.forecaster_n_heads, args.batch_size, args.num_workers, device,
        )
        if pd_rho:
            series.append(("per-dataset EAF", pd_rho))
        else:
            print("  [warn] No per-dataset checkpoints found — skipping series.")

    if args.universal_json:
        for jp in args.universal_json:
            jp = Path(jp)
            if not jp.exists():
                print(f"[warn] JSON not found: {jp} — skip")
                continue
            rho_fc, rho_bl, label = _load_universal_json(jp)
            series.append((label, rho_fc))
            if not baseline:
                baseline = rho_bl

    if not series:
        print("No data available. Exiting.")
        return

    if args.no_baseline:
        baseline = {}

    # ── dataset list ─────────────────────────────────────────────────────────
    if args.datasets:
        datasets = args.datasets
    else:
        all_ds = set()
        for _, d in series:
            all_ds.update(d.keys())
        if baseline:
            all_ds.update(baseline.keys())
        datasets = sorted(all_ds)

    print(f"\nDatasets ({len(datasets)}): {datasets}")

    _print_summary(series, baseline, datasets)

    # ── output paths ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    src_tag = "+".join(map(str, args.layers_source))
    stem    = f"src{src_tag}"

    def _avg(d):
        v = [x for x in d.values() if not np.isnan(x)]
        return np.mean(v) if v else float("nan")

    avg_parts = [f"{lbl[:18]}={_avg(rho):.3f}" for lbl, rho in series]
    if baseline and not args.no_baseline:
        avg_parts.append(f"baseline={_avg(baseline):.3f}")
    subtitle = "  ".join(avg_parts)
    title    = f"EAF Spearman ρ — {args.model_name}  (src={src_tag})\n{subtitle}"

    print("\nSaving outputs …")
    _spider(series, baseline, datasets, title, out_dir / "rho_spider.png")
    _save_csv(series, baseline, datasets, out_dir / "rho_table.csv")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
