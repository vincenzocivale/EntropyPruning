# Online WSI training for tile EAF

This pipeline trains one task-agnostic EAF module per THUNDER tile encoder and,
optionally, adapts that encoder to its own pruning pattern. It reads raw tiles
from WSI files and generates all teacher signals on the fly.

## Design goals

1. **No feature cache.** Source tokens, final-layer attention, and full-teacher
   embeddings are never written to disk.
2. **Broad WSI coverage without enumerating every tile.** An epoch sees a large
   rotating subset of slides and a bounded number of tiles per slide.
3. **Case-disjoint validation.** Existing manifest splits are respected;
   otherwise a stable hash of `case_id` creates the split in memory.
4. **Task independence.** Neither stage uses downstream labels or a
   dataset-specific classification head.
5. **Efficient I/O and GPU use.** Batches group several tiles from a small
   number of WSI files, DataLoader workers keep LRU OpenSlide/coordinate caches and
   configurable decoded-tile caches,
   and training uses mixed precision, pinned memory, persistent workers, TF32,
   and gradient accumulation.

## Sampling policy

The default training budget is resolved as:

```text
min(number of training WSI,
    max(512, ceil(0.50 * number of training WSI)))
```

For a training split of roughly 1,300 WSI, this schedules about 650 WSI per
epoch rather than all slides. Tile-EAF samples 24 tiles per scheduled WSI,
which is approximately 15,600 tiles per epoch. Pruning-aware adaptation samples
16 tiles per WSI, approximately 10,400 tiles per epoch.

The sampler:

- assigns cohort weights proportional to `sqrt(number of slides)` by default
  (`--cohort-balance-power 0.5`), reducing dominance by large cohorts without
  making every cohort artificially equal;
- keeps one deterministic permutation per cohort and rotates through it across
  epochs, so repeated epochs cover the corpus instead of redrawing a narrow iid
  subset;
- samples tile coordinates without replacement whenever a WSI has enough
  tissue tiles;
- uses four WSI per batch by default, amortizing OpenSlide seeks while preserving
  cross-slide diversity;
- keeps validation fixed by always evaluating sampler epoch zero.

Override the budget with `--train-wsis-per-epoch N`, or change the automatic
fraction with `--train-wsi-fraction`. Increasing tiles per WSI improves local
morphological coverage but increases correlation and I/O per slide; increasing
WSI per epoch improves cohort and patient coverage.

## Stage 1: train tile EAF online

```bash
export EAF_ROOT=/data2/home/vcivale/projects/imaging/EAF
export EAF_WSI_ROOT=/data2/home/vcivale/projects/imaging/data/WSI
export DATASET_ROOT=$EAF_WSI_ROOT/datasets/pretraining/eaf_wsi_pretrain_strict_v1
# ^ HISTAI+GTEx+HEST union, TCGA excluded by policy (see CLAUDE.md "WSI EAF
#   Training Data Policy"). Build it first with:
#   python scripts/eaf.py data build-strict --data-root "$EAF_WSI_ROOT"
#   The old tcga_eaf_multicohort_v1 / eaf_multisource_clean_v1 corpora (TCGA-
#   containing) were deleted 2026-08-07; do not point EAF training at TCGA.

cd "$EAF_ROOT"

python scripts/train_wsi_tile_eaf_online.py \
  --model-name <thunder-model-name> \
  --manifest "$DATASET_ROOT/manifests/slides.csv" \
  --data-root "$EAF_WSI_ROOT" \
  --source-layer 2 \
  --tile-size-at-target-mag <encoder-tile-size> \
  --keep-ratio-metric 0.10 \
  --train-wsi-fraction 0.50 \
  --tiles-per-wsi 24 \
  --batch-size 32 \
  --slides-per-batch 4 \
  --num-workers 8 \
  --openslide-cache-mib 256 \
  --amp-dtype bf16 \
  --early-stopping-patience 6 \
  --wandb-project eaf-tile-online
  # --output-dir defaults to checkpoints/tile_eaf/<thunder-model-name>
  # (tile-encoder-dependent); pass --output-dir to override.
```

Use `--teacher-checkpoint` only when the THUNDER encoder must be initialized
from a specific task-independent checkpoint. The teacher remains frozen.

The canonical coordinates describe 512 px windows at 20x. Set
`--tile-size-at-target-mag` to the physical tile size expected by the selected
encoder (for example, the size prescribed by its THUNDER/TRIDENT recipe). If it
is smaller than 512, the training loader jitters that crop inside the parent
window and validation uses a deterministic center crop. Omitting the option
uses the full canonical 512 px field of view.

For each tile, the forward pass captures:

```text
source block output patch tokens -> AttentionForecaster
final block CLS-to-patch attention -> normalized teacher target
```

The loss combines KL divergence with a rank-alignment term. Validation reports
KL, Spearman correlation, and top-k recall at the requested pruning ratio.
Early stopping monitors validation KL.

Only the best forecaster checkpoint and a JSON run summary are saved.

## Stage 2: adapt the pruned tile encoder online

```bash
python scripts/finetune_wsi_tile_encoder_pruned_online.py \
  --model-name <same-thunder-model-name> \
  --manifest "$DATASET_ROOT/manifests/slides.csv" \
  --data-root "$EAF_WSI_ROOT" \
  --forecaster-ckpt checkpoints/tile_eaf/<thunder-model-name>/best_<run>.pt \
  --prune-layer 2 \
  --keep-ratio 0.10 \
  --tile-size-at-target-mag <encoder-tile-size> \
  --train-wsi-fraction 0.50 \
  --tiles-per-wsi 16 \
  --batch-size 16 \
  --slides-per-batch 4 \
  --grad-accum 2 \
  --num-workers 8 \
  --openslide-cache-mib 256 \
  --amp-dtype bf16 \
  --gradient-checkpointing \
  --early-stopping-patience 5 \
  --wandb-project eaf-pruned-tile-online
  # --output-dir defaults to checkpoints/pruned_finetuned/<thunder-model-name>
  # (tile-encoder-dependent); pass --output-dir to override.
```

This stage is not supervised classification fine-tuning. It minimizes the
representation discrepancy between:

```text
frozen full base encoder(tile)
LoRA-adapted EAF-pruned encoder(tile)
```

The objective combines cosine distance, normalized MSE, and within-batch
pairwise-similarity preservation. The full teacher and pruned student share one
backbone allocation: LoRA is temporarily disabled for the full target pass,
then enabled for the pruned pass. This substantially reduces VRAM relative to
holding two foundation models.

Only trainable LoRA tensors are saved. Downstream heads remain separate and can
be trained later for evaluation without changing the task-agnostic EAF stage.

## OpenSlide and DataLoader tuning

Both training stages reuse persistent DataLoader workers and worker-local LRU caches
for WSI handles and coordinate arrays. ``--openslide-cache-mib`` additionally sets the
decoded-tile cache on every OpenSlide handle. This extraction optimization also applies
directly to training because training rereads raw pixels every epoch. Capacity is per
handle, and each worker may retain up to ``--slide-cache-size`` handles. A conservative
upper host-RAM budget is therefore:

```text
num_workers * slide_cache_size * openslide_cache_mib
```

Actual residency is normally lower, but monitor it on long runs. Start with 4 workers,
``--slide-cache-size 2`` and 256 MiB, then measure end-to-end training tiles/s. The
extraction result of batch 128 must not be copied blindly to training: early-layer
teacher inference, forecaster backward, optimizer state, augmentation and validation
have a different GPU-memory and CPU profile. Tune its batch size separately and use
gradient accumulation when a larger effective batch is desired.

Increasing workers cannot parallelize construction of one batch: PyTorch assigns the
batch index list to one worker. The balanced sampler already limits each batch to a
small set of WSI, but random coordinates have weaker spatial locality than sequential
cache extraction. If data wait remains material, the next optimization to evaluate is
batched spatial reordering inside ``Dataset.__getitems__`` while restoring the original
sampler order before collation.

For scale, sampling 24 random coordinates from each of 100 HISTAI slides produced a
23.3% theoretical decoded-block reuse rate. Once the samples were split into the three
8-tile-per-slide rounds that may be handled by different workers, within-round reuse
was only 8.6%. This supports a smaller training default (256 MiB) than sequential cache
extraction (512 MiB), and means the option should be benchmarked rather than assumed to
reproduce extraction's throughput gain.

## Weights & Biases and early stopping

Both commands log:

- per-step loss, gradient norm, and tile throughput;
- epoch train/validation metrics;
- effective WSI and tile budget;
- learning rate;
- best validation objective and patience counter;
- peak allocated CUDA memory.

Modes:

```text
--wandb-mode online    # default
--wandb-mode offline
--wandb-mode disabled
```

The best checkpoint is updated only when validation improves by at least
`--early-stopping-min-delta`. Training stops after
`--early-stopping-patience` non-improving epochs. Set patience to zero to
disable early stopping.

## Compatibility entry points

The historical commands now delegate to the online implementations:

```text
scripts/train_forecaster.py
scripts/train_multi_thunder_forecaster.py
  -> scripts/train_wsi_tile_eaf_online.py

scripts/finetune_pruned.py
scripts/finetune_multi_thunder_pruned.py
  -> scripts/finetune_wsi_tile_encoder_pruned_online.py
```

Their old cache-oriented and supervised CLI arguments are intentionally no
longer accepted. This prevents accidental recreation of multi-gigabyte HDF5
feature stores or task-specific pruned encoders.

## Recommended smoke test

Before a full run, create a manifest containing 8-16 WSI from at least two
cohorts and run one epoch with W&B disabled:

```bash
python scripts/train_wsi_tile_eaf_online.py \
  --model-name <thunder-model-name> \
  --manifest /path/to/smoke_slides.csv \
  --data-root "$EAF_WSI_ROOT" \
  --train-wsis-per-epoch 8 \
  --tiles-per-wsi 4 \
  --val-wsis 4 \
  --val-tiles-per-wsi 4 \
  --epochs 1 \
  --num-workers 2 \
  --wandb-mode disabled
```

Confirm that no `.h5`, `.npy`, or tile-image cache appears in the output
directory and that only one checkpoint plus one JSON summary are created.
