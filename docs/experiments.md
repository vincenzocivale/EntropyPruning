# Experiment registry and artifact policy

Every scientific run is identified by:

```text
experiment_id / variant_id / seed
```

`experiment_id` and `variant_id` must be declared in
`configs/experiments/registry.toml` **before** launch.

Timestamps, UUIDs, SLURM/job IDs and W&B IDs are metadata only. They are never
experiment names and never determine output paths.

## Canonical paths

Example:

```text
tile_source_ablation_conch15 / src02 / seed_42
```

maps to:

```text
$EAF_WSI_ROOT/
├── checkpoints/tile_eaf/forecaster/
│   └── tile_source_ablation_conch15/src02/seed_42/
│       ├── best.pt
│       └── latest.pt
├── results/tile_eaf/forecaster/
│   └── tile_source_ablation_conch15/src02/seed_42/
│       ├── run.json
│       ├── summary.json
│       └── history.json
├── logs/tile_eaf/forecaster/
│   └── tile_source_ablation_conch15/src02/seed_42/
└── caches/experiments/
    └── tile_source_ablation_conch15/src02/seed_42/
```

Checkpoint filenames do not repeat the run name because the directory is unique.

The same pattern applies to `wsi_eaf`, with two distinct stages that are easy to
confuse:

```text
checkpoints/wsi_eaf/forecaster/<experiment_id>/<variant_id>/seed_<seed>/best.pt
checkpoints/wsi_eaf/distillation/<experiment_id>/<variant_id>/seed_<seed>/best.pt
```

`forecaster/` holds the frozen WSI-EAF network that scores which TITAN tiles to
keep (step 6, `train_wsi_eaf.py`) -- it is not itself a usable pruned FM.
`distillation/` holds the LoRA weights that actually implement pruning inside
TITAN's own forward pass (step 7, `distill_wsi_titan.py`), loaded together with
its frozen forecaster via `PrunedLoRATitanEncoder`
(`src/models/wsi/pruned_titan.py`). Evaluation and downstream use always target
a `distillation/` checkpoint, never a `forecaster/` one directly.

## Registry workflow

Before a paper run:

1. add/update the experiment in the registry;
2. choose a descriptive `experiment_id`;
3. enumerate allowed `variant_id` values;
4. keep `status = "blocked"` while a scientific choice is unresolved;
5. set `status = "ready"` only when inputs/configuration are frozen;
6. commit the registry change;
7. launch with `--experiment-id` and `--variant-id`.

Inspect without launching:

```bash
python scripts/experiments.py list
python scripts/experiments.py show tile_source_ablation_conch15
python scripts/experiments.py paths \
  --experiment-id tile_source_ablation_conch15 \
  --variant-id src02 \
  --seed 42
```

## Ablation policy

Source-layer, keep-ratio and comparable tuning ablations are performed only on:

```text
CONCH v1.5 + TITAN
```

Other foundation models inherit the frozen configuration directly.

## Unlabeled training data

The canonical paper corpus is `histai_core_v1`, permanently excluding:

```text
HISTAI-mixed
HISTAI-skin-b2
```

The exclusion must be encoded in the manifest, not remembered only via runtime flags.

## Reusable vs experiment-derived caches

Reusable full teacher caches stay dataset/model-centric. Existing validated caches
remain where they are and must not be duplicated merely to match a new layout:

```text
caches/tile_eaf/...
caches/wsi_eaf/...
```

Experiment-derived caches belong under:

```text
caches/experiments/<experiment_id>/<variant_id>/seed_<seed>/
```

Reserved subdirectories:

```text
tile_pruned/
wsi_source/
```

Do not build a pruned tile cache for every source-layer candidate. Select/freeze the
tile configuration first, then build one canonical pruned tile cache. Likewise, one
multi-hidden TITAN source cache can support the WSI source-layer sweep.

## What to save

Training:

```text
best.pt
latest.pt                 # only if resume is supported
run.json
summary.json
history.json              # if epoch history exists
run.log
```

Evaluation:

```text
run.json
summary.json
results.csv
run.log
```

`run.json` is written at startup and records registry hash, git revision, dirty state,
full config and canonical paths.

`summary.json` records final metrics plus provenance of manifests, tile input cache,
WSI source cache, WSI teacher cache, labels and input checkpoints.

The external spatial biology evaluation additionally freezes `protocol_sha256`
in its registry entry. Its per-spot predictions, signature audit, spatial metrics
and report follow the contracts in [spatial_biology.md](spatial_biology.md).

## What not to save

Do not permanently save per-epoch checkpoints, Tile-EAF `early_tokens`, duplicate full
teacher caches under every experiment, arbitrary one-off downstream embeddings, or
result files inside the git working tree.

W&B is optional telemetry, not the canonical scientific record.
