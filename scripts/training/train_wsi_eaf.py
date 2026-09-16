#!/usr/bin/env python
"""Train WSIDenseForecaster to distill TITAN's real final-layer cross-tile
attention from the tile encoder's final embeddings.

Both inputs and targets are fully precomputed (Tile-EAF + WSI-EAF compact
caches) -- no tile-encoder or WSI-FM forward pass runs here, only the small
forecaster itself. See `src/models/wsi/dense_forecaster.py` for why this uses
standard dense self-attention (mirroring TITAN's own proven-stable block
design at smaller scale) rather than the earlier landmark-bottleneck
architecture, and the Loss/metric design mirrors `scripts/training/train_tile_eaf.py` (KL +
rank-alignment, Spearman rho, top-k recall) for consistency across the two
EAF forecaster training scripts in this repo.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.wsi.wsi_forecaster_dataset import (
    WSIForecasterDataset,
    WSIForecasterManifestConfig,
    assign_splits,
    build_attention_manifest_csv,
    build_manifest,
    write_manifest_csv,
)
from src.models.wsi.dense_forecaster import WSIDenseForecaster, WSIDenseForecasterALiBi
from src.utils import default_checkpoint_root, set_seed, wsi_encoder_pair_dir_name
from src.wsi_pipeline.experiment_results import publish_run_summary


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
    parser.add_argument("--tile-input-root", type=Path, required=True, help="Tile embeddings used by the WSI student; final runs use the distilled Tile-EAF-pruned encoder cache")
    parser.add_argument("--source-wsi-root", type=Path, required=True, help="WSI cache built from --tile-input-root; intermediate hidden states are read here")
    parser.add_argument("--teacher-wsi-root", type=Path, required=True, help="Frozen full WSI teacher cache built from the unpruned tile encoder; attention targets are read here")
    parser.add_argument(
        "--tile-encoder", required=True,
        help="Tile encoder that produced --tile-eaf-root's cache, e.g. conch_v15 -- used for "
        "the default run-name/checkpoint-dir naming (<tile-encoder>__<wsi-encoder>_src<NN>), "
        "same convention as scripts/training/train_tile_eaf.py.",
    )
    parser.add_argument(
        "--wsi-encoder", default="titan",
        help="WSI-FM whose attention this forecaster targets -- naming only, does not select code path.",
    )
    parser.add_argument(
        "--tile-input-variant", default="base",
        help="Tag identifying the tile-encoder input this run was trained on: 'base' (frozen "
        "tile-encoder cache) or the tile-EAF pruned run name (e.g. from `eaf.py cache tile "
        "--pruned-adapter-ckpt`'s output-dir suffix) if --tile-eaf-root points at a pruned-input "
        "cache. Folded into the run name so a pruned-input run is never confused with a base one.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to $EAF_WSI_ROOT/checkpoints/wsi_eaf/<tile-encoder>__<wsi-encoder>/<run-name>/",
    )
    parser.add_argument("--cohorts", nargs="+", default=None)
    parser.add_argument(
        "--exclude-cohort", nargs="*", default=["HISTAI-mixed", "HISTAI-skin-b2"],
        help=(
            "HISTAI subsets to exclude (default: the two largest/slowest-to-download "
            "subsets, HISTAI-mixed and HISTAI-skin-b2, same default as "
            "train_tile_eaf.py -- pass --exclude-cohort with no values to "
            "include everything)"
        ),
    )
    parser.add_argument("--attention-key", default="attention/global_to_tiles_mass_share")
    parser.add_argument("--target-layer", type=int, default=-1, help="-1 = TITAN's real final layer")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=17, help="case-disjoint HISTAI split seed")

    parser.add_argument(
        "--input-source", choices=("tile_embeddings", "titan_hidden"), default="tile_embeddings",
        help=(
            "tile_embeddings: the tile encoder's context-free final embeddings (original "
            "behavior). titan_hidden: TITAN's own intermediate hidden state at "
            "--titan-hidden-layer (auxiliary/hidden_layer_{k:03d} in the wsi_eaf output "
            "file, see cache_wsi_teacher.py --titan-hidden-layer); requires --architecture "
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
    parser.add_argument(
        "--slides-per-epoch", type=int, default=0,
        help=(
            "Cap training to a random subset of this many train-split slides per epoch "
            "(0 = every train slide, the previous behavior). Unlike Tile-EAF, WSI-EAF has no "
            "--tiles-per-wsi-style knob: one 'epoch' here is a full pass over EVERY train "
            "slide at its FULL tile bag (no tile subsampling), so with a large corpus one "
            "epoch can take on the order of a day -- one validation/early-stopping check a "
            "day is too coarse to catch a plateau promptly (observed: val_rho already "
            ">=0.97 within the first few epochs on past runs). A different random subset is "
            "drawn each epoch (seeded from --seed + epoch, so it's reproducible across a "
            "--resume) so the full train split is still covered over enough epochs."
        ),
    )
    parser.add_argument(
        "--val-slides", type=int, default=0,
        help="Cap the per-epoch validation pass to a random subset of this many validation-split "
        "slides (0 = every validation slide). Sampled once at startup (fixed across epochs, so "
        "early-stopping/best-checkpoint decisions compare against a stable set) -- keep this "
        "well below --slides-per-epoch or validation dominates epoch wall-clock time.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=6,
        help="Stop after this many consecutive epochs without a val_rho improvement of at least "
        "--early-stopping-min-delta (0 disables early stopping). Same default/semantics as "
        "train_tile_eaf.py.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    parser.add_argument("--wandb-project", default="EAF-WSI-level")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <output-dir>/latest_<run-name>.pt (model+optimizer+scheduler+epoch), "
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

    # Run/checkpoint naming mirrors scripts/training/train_tile_eaf.py: one
    # subdirectory per run under $EAF_WSI_ROOT/checkpoints/wsi_eaf/<pair>/, named
    # deterministically from the source layer (or "srcfinal" for the original
    # context-free tile_embeddings input) and the tile-input variant.
    pair = wsi_encoder_pair_dir_name(args.tile_encoder, args.wsi_encoder)
    layer_tag = f"src{args.titan_hidden_layer:02d}" if args.input_source == "titan_hidden" else "srcfinal"
    variant_tag = "" if args.tile_input_variant == "base" else f"__{args.tile_input_variant}"
    run_name = args.run_name or f"{pair}_{layer_tag}{variant_tag}"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (default_checkpoint_root("wsi_eaf") / pair / run_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_config = WSIForecasterManifestConfig(
        tile_input_root=args.tile_input_root,
        source_wsi_root=args.source_wsi_root,
        teacher_wsi_root=args.teacher_wsi_root,
        attention_key=args.attention_key,
        target_layer=args.target_layer,
        cohorts=tuple(args.cohorts) if args.cohorts else None,
        exclude_cohorts=tuple(args.exclude_cohort) if args.exclude_cohort else None,
        hidden_layer=args.titan_hidden_layer if args.input_source == "titan_hidden" else None,
    )
    table = build_manifest(manifest_config)
    table = assign_splits(
        table,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    write_manifest_csv(table, output_dir / "manifest.csv")
    attention_manifest_path = output_dir / "attention_manifest.csv"
    build_attention_manifest_csv(table, attention_manifest_path)
    print(
        "slides: "
        + ", ".join(f"{split}={count}" for split, count in table["split"].value_counts().sort_index().items())
    )

    def make_dataset(split: str) -> WSIForecasterDataset:
        return WSIForecasterDataset(
            table, attention_manifest_path=attention_manifest_path, config=manifest_config, split=split
        )

    train_dataset = make_dataset("train")
    val_dataset = make_dataset("validation")
    full_val_len = len(val_dataset)

    # Val subset is fixed once at startup (not resampled per epoch) so early-stopping/
    # best-checkpoint decisions always compare against the same slides.
    if 0 < args.val_slides < full_val_len:
        val_indices = random.Random(args.seed).sample(range(full_val_len), args.val_slides)
        val_dataset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Train: --slides-per-epoch=0 (default) keeps the original single-loader-reused-
    # every-epoch behavior (DataLoader(shuffle=True) reshuffles on every fresh __iter__
    # regardless). >0 draws a fresh random subset each epoch instead -- see the flag's
    # help text for why (one epoch over the full split can take on the order of a day).
    subsample_train = 0 < args.slides_per_epoch < len(train_dataset)
    static_train_loader = None
    if not subsample_train:
        static_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    def epoch_train_loader(epoch: int) -> DataLoader:
        if static_train_loader is not None:
            return static_train_loader
        indices = random.Random(args.seed + epoch).sample(range(len(train_dataset)), args.slides_per_epoch)
        return DataLoader(Subset(train_dataset, indices), batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print(
        f"epoch length: train={args.slides_per_epoch if subsample_train else len(train_dataset)}/"
        f"{len(train_dataset)} slides, val={len(val_dataset)}/{full_val_len} slides"
    )

    embed_dim = train_dataset[0][0].shape[-1]
    model_cls = WSIDenseForecasterALiBi if use_alibi else WSIDenseForecaster
    model = model_cls(
        embed_dim=embed_dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Matches scripts/training/train_tile_eaf.py's convention: a smoothly decaying
    # LR reduces how large a late-training update can be, which is cheap insurance
    # against the kind of training-time instability the landmark architecture hit.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_rho = float("-inf")
    global_step = 0
    epochs_without_improvement = 0
    latest_ckpt_path = output_dir / f"latest_{run_name}.pt"
    best_ckpt_path = output_dir / f"best_{run_name}.pt"
    if args.resume and latest_ckpt_path.exists():
        payload = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        best_val_rho = payload["best_val_rho"]
        epochs_without_improvement = payload.get("epochs_without_improvement", 0)
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
        wandb.init(project=args.wandb_project, name=run_name, job_type="wsi_eaf", config=vars(args))

    last_epoch_run = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        last_epoch_run = epoch
        train_metrics, global_step = _run_epoch(
            model=model,
            use_alibi=use_alibi,
            content_blind=args.content_blind,
            loader=epoch_train_loader(epoch),
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
        improved = val_metrics["rho"] > best_val_rho + args.early_stopping_min_delta
        if improved:
            best_val_rho = val_metrics["rho"]
            epochs_without_improvement = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "run_name": run_name, "epoch": epoch, "val_rho": best_val_rho},
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
        if use_wandb:
            wandb.log({"early_stopping/epochs_without_improvement": epochs_without_improvement}, step=global_step)
        # Full-state checkpoint every epoch (overwritten each time) so --resume can pick up
        # after an external interruption without losing model/optimizer/scheduler progress.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "run_name": run_name,
                "epoch": epoch,
                "global_step": global_step,
                "val_rho": val_metrics["rho"],
                "best_val_rho": best_val_rho,
                "epochs_without_improvement": epochs_without_improvement,
            },
            latest_ckpt_path,
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"({epochs_without_improvement} epochs without a val_rho improvement >= {args.early_stopping_min_delta})"
            )
            break

    print(f"best val_rho={best_val_rho:.4f}, checkpoint at {best_ckpt_path}")
    summary = {
        "run_name": run_name,
        "best_val_rho": best_val_rho,
        "checkpoint": str(best_ckpt_path),
        "epochs_completed": last_epoch_run - start_epoch + 1,
        "stopped_early": last_epoch_run + 1 < args.epochs,
        "slides_per_epoch": args.slides_per_epoch or len(train_dataset),
        "val_slides": len(val_dataset),
        "tile_encoder": args.tile_encoder,
        "wsi_encoder": args.wsi_encoder,
        "tile_input_variant": args.tile_input_variant,
        "input_source": args.input_source,
        "titan_hidden_layer": args.titan_hidden_layer,
        "architecture": args.architecture,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    (output_dir / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    publish_run_summary(family="wsi_eaf", stage="forecaster", run_name=run_name, args=args, summary=summary)
    if use_wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
