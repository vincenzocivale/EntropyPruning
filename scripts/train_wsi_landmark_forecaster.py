#!/usr/bin/env python
"""Train WSIDenseForecaster to distill TITAN's real final-layer cross-tile
attention from the tile encoder's final embeddings.

Both inputs and targets are fully precomputed (Tile-EAF + WSI-EAF compact
caches) -- no tile-encoder or WSI-FM forward pass runs here, only the small
forecaster itself. See `src/models/wsi/dense_forecaster.py` for why this uses
standard dense self-attention (mirroring TITAN's own proven-stable block
design at smaller scale) rather than the earlier landmark-bottleneck
architecture, and the WSI-EAF attention-signal investigation notes for why
final-layer embeddings (not early-layer) and learned weights (not raw content
similarity) are both required ingredients.

Loss/metric design mirrors `scripts/train_wsi_tile_eaf_online.py` (KL +
rank-alignment, Spearman rho, top-k recall) for consistency across the two
EAF forecaster training scripts in this repo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.wsi.wsi_forecaster_dataset import (
    WSIForecasterDataset,
    WSIForecasterManifestConfig,
    assign_splits,
    build_attention_manifest_csv,
    build_manifest,
    write_manifest_csv,
)
from src.models.wsi.dense_forecaster import WSIDenseForecaster, WSIDenseForecasterALiBi
from src.utils import set_seed


def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rank = prediction.argsort(-1).argsort(-1).float()
    target_rank = target.argsort(-1).argsort(-1).float()
    pred_rank -= pred_rank.mean(-1, keepdim=True)
    target_rank -= target_rank.mean(-1, keepdim=True)
    denominator = torch.sqrt(pred_rank.square().sum(-1) * target_rank.square().sum(-1)).clamp_min(1e-8)
    return (pred_rank * target_rank).sum(-1) / denominator


def _topk_recall(prediction: torch.Tensor, target: torch.Tensor, ratio: float) -> torch.Tensor:
    count = max(1, int(round(prediction.shape[-1] * ratio)))
    pred_indices = prediction.topk(count, dim=-1).indices
    target_indices = target.topk(count, dim=-1).indices
    pred_mask = torch.zeros_like(prediction, dtype=torch.bool)
    pred_mask.scatter_(1, pred_indices, True)
    return pred_mask.gather(1, target_indices).float().mean(dim=-1)


def _rank_alignment_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_score = target.clamp_min(1e-8).log()
    logits_centered = logits - logits.mean(dim=-1, keepdim=True)
    target_centered = target_score - target_score.mean(dim=-1, keepdim=True)
    return 1.0 - F.cosine_similarity(logits_centered, target_centered, dim=-1).mean()


def _autocast(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _run_epoch(
    *,
    model: WSIDenseForecaster | WSIDenseForecasterALiBi,
    use_alibi: bool,
    content_blind: bool,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_dtype: str,
    kl_weight: float,
    rank_loss_weight: float,
    keep_ratio_metric: float,
    slides_per_step: int,
    train: bool,
    log_every: int,
    global_step: int,
    use_wandb: bool,
) -> tuple[dict[str, float], int]:
    model.train(train)
    sums = {key: 0.0 for key in ("loss", "kl", "rank", "rho", "topk_recall")}
    total = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for step, (tile_embeddings, coords, target, slide_id) in enumerate(iterator):
        tile_embeddings = tile_embeddings.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if content_blind:
            # Control experiment: every tile gets the exact same (zero) input, so the
            # model has zero per-tile content to distinguish tiles by. Any two tiles at
            # the same pairwise distances from everything else are then genuinely
            # interchangeable -- the *only* thing that can break that symmetry through
            # attention is the additive ALiBi bias term (position-only). This isolates
            # how much of the real model's rho comes from the geometric prior TITAN and
            # this forecaster share by construction, vs. from actually reading tile
            # content -- see the WSI-EAF ALiBi-prior discussion in this conversation.
            tile_embeddings = torch.zeros_like(tile_embeddings)
        try:
            with torch.set_grad_enabled(train), _autocast(device, amp_dtype):
                if use_alibi:
                    logits = model(tile_embeddings, coords.to(device, non_blocking=True))
                else:
                    logits = model(tile_embeddings)
            # KL/rank-alignment over up to ~20k tiles is a large-vocabulary softmax/log-domain
            # reduction -- kept in fp32 on general principle (same reason LLM training always
            # upcasts logits to fp32 before cross-entropy), even though the grad_norm-spike
            # instability actually traced to the earlier landmark architecture's unbounded
            # learned-token growth, not to this precision choice (see
            # docs/wsi_eaf_landmark_forecaster notes / dense_forecaster.py docstring).
            logits = logits.float()
            target = target.float()
            kl = F.kl_div(logits.log_softmax(dim=-1), target, reduction="batchmean")
            rank_loss = _rank_alignment_loss(logits, target)
            loss = kl_weight * kl + rank_loss_weight * rank_loss

            if train:
                (loss / slides_per_step).backward()
        except torch.cuda.OutOfMemoryError as exc:
            # ALiBi's per-head [N,N] bias (N = tile count, up to ~20k on the largest HISTAI
            # slides) is a real, unavoidable memory spike distinct from TITAN's own OOMs
            # during cache building -- skip this one slide rather than losing the whole
            # epoch/run to it. `optimizer.zero_grad` discards any partial gradients this
            # slide may have accumulated before failing.
            if train:
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(
                f"[oom-skip] slide={slide_id} n_tiles={tile_embeddings.shape[1]} "
                f"train={train}: {exc}",
                flush=True,
            )
            continue

        if train:
            if (step + 1) % slides_per_step == 0 or step + 1 == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if use_wandb and global_step % log_every == 0:
                    wandb.log(
                        {
                            "step/loss": float(loss.detach()),
                            "step/kl": float(kl.detach()),
                            "step/rank": float(rank_loss.detach()),
                            "step/grad_norm": float(grad_norm),
                            "trainer/global_step": global_step,
                        },
                        step=global_step,
                    )

        with torch.no_grad():
            rho = _spearman(logits.float(), target.float()).mean()
            recall = _topk_recall(logits.float(), target.float(), keep_ratio_metric).mean()
        sums["loss"] += float(loss.detach())
        sums["kl"] += float(kl.detach())
        sums["rank"] += float(rank_loss.detach())
        sums["rho"] += float(rho.detach())
        sums["topk_recall"] += float(recall.detach())
        total += 1

    elapsed = max(time.perf_counter() - start, 1e-6)
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    metrics["slides_per_second"] = total / elapsed
    return metrics, global_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-eaf-root", type=Path, required=True)
    parser.add_argument("--wsi-eaf-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", default=None)
    parser.add_argument("--attention-key", default="attention/global_to_tiles_mass_share")
    parser.add_argument("--target-layer", type=int, default=-1, help="-1 = TITAN's real final layer")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=17, help="matches the correlational-analysis splits")

    parser.add_argument(
        "--input-source", choices=("tile_embeddings", "titan_hidden"), default="tile_embeddings",
        help=(
            "tile_embeddings: the tile encoder's context-free final embeddings (original "
            "behavior). titan_hidden: TITAN's own intermediate hidden state at "
            "--titan-hidden-layer (auxiliary/hidden_layer_{k:03d} in the wsi_eaf output "
            "file, see wsi_eaf_infer_wsi_fm.py --titan-hidden-layer); requires --architecture "
            "dense_alibi (a plain forecaster has no way to use TITAN's own ALiBi-contextualized "
            "features correctly without the matching spatial bias)."
        ),
    )
    parser.add_argument(
        "--titan-hidden-layer", type=int, default=None,
        help="0-based TITAN vision-encoder block index to read as the input bag; required "
        "when --input-source titan_hidden.",
    )
    parser.add_argument(
        "--architecture", choices=("dense", "dense_alibi"), default="dense",
        help="dense: original WSIDenseForecaster (no positional information). dense_alibi: "
        "WSIDenseForecasterALiBi, which adds TITAN-matching ALiBi spatial bias from real tile "
        "coordinates -- required for --input-source titan_hidden, optional (but recommended) "
        "for tile_embeddings too since the target attention is itself ALiBi-biased regardless "
        "of what feeds the forecaster.",
    )
    parser.add_argument(
        "--content-blind", action="store_true",
        help="Control experiment: zero out tile_embeddings before every forward pass, so the "
        "only thing the model (and its ALiBi bias) can use is tile geometry, never content. "
        "Compare this run's val_rho against a normal run on the same layer/split to see how "
        "much of the reported rho is explained by the ALiBi spatial prior TITAN and this "
        "forecaster share by construction, rather than by genuinely reading tile content.",
    )

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1.0, help="0 trains on rank_loss alone")
    parser.add_argument("--rank-loss-weight", type=float, default=0.3)
    parser.add_argument("--keep-ratio-metric", type=float, default=0.2, help="retention fraction for the top-k recall metric")
    parser.add_argument("--slides-per-step", type=int, default=16, help="gradient-accumulation width")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <output-dir>/latest_wsi_landmark_forecaster.pt (model+optimizer+scheduler+epoch), "
        "if present. The background jobs in this environment have been killed by external interruptions "
        "(disk pressure, session restarts) more than once; without this, every interruption threw away all "
        "prior training progress and restarted from a fresh random init.",
    )
    args = parser.parse_args()

    if args.input_source == "titan_hidden":
        if args.titan_hidden_layer is None:
            raise SystemExit("--titan-hidden-layer is required when --input-source titan_hidden")
        # NOTE: --architecture dense (no ALiBi) is a legitimate choice here, unlike for
        # --input-source tile_embeddings. TITAN's own hidden state at --titan-hidden-layer
        # was already produced by real ALiBi-biased self-attention in TITAN's own earlier
        # blocks, so tile position is already mixed into the *content* of this input --
        # a plain (no-bias) forecaster reading it is not starting from context-free,
        # position-blind features the way it would with raw tile-encoder embeddings.
        # Whether our own extra ALiBi bias still helps on top of that is an empirical
        # question, not a hard requirement -- see the content-blind control experiment.
    if args.content_blind and args.architecture != "dense_alibi":
        raise SystemExit(
            "--content-blind only isolates a meaningful signal against --architecture dense_alibi "
            "(plain dense with zeroed input has nothing left to distinguish tiles by at all)"
        )
    use_alibi = args.architecture == "dense_alibi"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_config = WSIForecasterManifestConfig(
        tile_eaf_root=args.tile_eaf_root,
        wsi_eaf_root=args.wsi_eaf_root,
        attention_key=args.attention_key,
        target_layer=args.target_layer,
        cohorts=tuple(args.cohorts) if args.cohorts else None,
        hidden_layer=args.titan_hidden_layer if args.input_source == "titan_hidden" else None,
    )
    table = build_manifest(manifest_config)
    table = assign_splits(
        table,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    write_manifest_csv(table, args.output_dir / "manifest.csv")
    attention_manifest_path = args.output_dir / "attention_manifest.csv"
    build_attention_manifest_csv(table, attention_manifest_path)
    print(
        "slides: "
        + ", ".join(f"{split}={count}" for split, count in table["split"].value_counts().sort_index().items())
    )

    def make_loader(split: str, *, shuffle: bool) -> DataLoader:
        dataset = WSIForecasterDataset(
            table, attention_manifest_path=attention_manifest_path, config=manifest_config, split=split
        )
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("validation", shuffle=False)

    embed_dim = train_loader.dataset[0][0].shape[-1]
    model_cls = WSIDenseForecasterALiBi if use_alibi else WSIDenseForecaster
    model = model_cls(
        embed_dim=embed_dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Matches scripts/train_wsi_tile_eaf_online.py's convention: a smoothly decaying
    # LR reduces how large a late-training update can be, which is cheap insurance
    # against the kind of training-time instability the landmark architecture hit.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_rho = float("-inf")
    global_step = 0
    latest_ckpt_path = args.output_dir / "latest_wsi_landmark_forecaster.pt"
    best_ckpt_path = args.output_dir / "best_wsi_landmark_forecaster.pt"
    if args.resume and latest_ckpt_path.exists():
        payload = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        best_val_rho = payload["best_val_rho"]
        print(f"resumed from {latest_ckpt_path}: epoch={payload['epoch']} val_rho={payload['val_rho']:.4f}")
    elif args.resume and best_ckpt_path.exists():
        # Pre-dates this checkpoint format (no optimizer/scheduler/epoch state, only weights)
        # -- e.g. a run interrupted before --resume support existed. Warm-start the model from
        # it rather than silently discarding real training progress; optimizer/scheduler/epoch
        # restart from scratch since that state was never saved.
        payload = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        best_val_rho = payload["val_rho"]
        print(f"warm-started weights from {best_ckpt_path} (epoch={payload['epoch']} val_rho={payload['val_rho']:.4f}); optimizer/schedule restart fresh (no saved state in this older checkpoint format)")

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step = _run_epoch(
            model=model,
            use_alibi=use_alibi,
            content_blind=args.content_blind,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_dtype=args.amp_dtype,
            kl_weight=args.kl_weight,
            rank_loss_weight=args.rank_loss_weight,
            keep_ratio_metric=args.keep_ratio_metric,
            slides_per_step=args.slides_per_step,
            train=True,
            log_every=args.log_every,
            global_step=global_step,
            use_wandb=use_wandb,
        )
        val_metrics, _ = _run_epoch(
            model=model,
            use_alibi=use_alibi,
            content_blind=args.content_blind,
            loader=val_loader,
            optimizer=None,
            device=device,
            amp_dtype=args.amp_dtype,
            kl_weight=args.kl_weight,
            rank_loss_weight=args.rank_loss_weight,
            keep_ratio_metric=args.keep_ratio_metric,
            slides_per_step=args.slides_per_step,
            train=False,
            log_every=args.log_every,
            global_step=global_step,
            use_wandb=False,
        )
        scheduler.step()
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_rho={train_metrics['rho']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_rho={val_metrics['rho']:.4f} "
            f"val_topk_recall={val_metrics['topk_recall']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if use_wandb:
            wandb.log(
                {**{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}, "epoch": epoch},
                step=global_step,
            )
        if val_metrics["rho"] > best_val_rho:
            best_val_rho = val_metrics["rho"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_rho": best_val_rho},
                args.output_dir / "best_wsi_landmark_forecaster.pt",
            )
        # Full-state checkpoint every epoch (overwritten each time) so --resume can pick up
        # after an external interruption without losing model/optimizer/scheduler progress.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step,
                "val_rho": val_metrics["rho"],
                "best_val_rho": best_val_rho,
            },
            latest_ckpt_path,
        )

    print(f"best val_rho={best_val_rho:.4f}, checkpoint at {args.output_dir / 'best_wsi_landmark_forecaster.pt'}")
    if use_wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
