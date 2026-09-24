# Repository Guidelines

## Scope

This repository supports one experimental chain only:

`HISTAI -> Tile-EAF -> tile embedding distillation -> WSI-EAF -> WSI embedding distillation -> THUNDER/TCGA-CPTAC evaluation`.

The registered HEST immune/stromal niche experiment is an additional external
evaluation of this same frozen chain, never an EAF pretraining extension.

Do not reintroduce supervised THUNDER pretraining, TCGA/HEST/GTEx EAF pretraining,
morphology coarsening, signal-discovery branches, or competing pruning baselines.

## Scientific invariants

- EAF and both distillation stages use unlabeled HISTAI data only.
- Downstream labels are used only by evaluation heads.
- Tile distillation targets the same frozen unpruned tile encoder.
- WSI student inputs come from the distilled/pruned tile encoder.
- WSI targets come from the full tile + full WSI teacher pipeline.
- Never collapse `source_wsi_root` and `teacher_wsi_root` in the final combined run.
- Patient/case grouping must remain disjoint across train/validation/test.

## Code structure

Reusable logic belongs under `src/`; `scripts/` contains thin entry points.
Supported training entry points are `train_tile_eaf.py`, `distill_tile_encoder.py`,
`train_wsi_eaf.py`, and `distill_wsi_titan.py`. Supported evaluation entry points
are `evaluate_tile_thunder.py`, `evaluate_wsi_downstream.py`, and
`evaluate_spatial_biology.py` (external evaluation only; see `docs/spatial_biology.md`).

Runtime artifacts never belong in git. Use `$EAF_WSI_ROOT` for datasets, caches,
checkpoints, results and logs.

## Tests

Run `pytest` from the repository root. Optional heavy dependencies must be guarded
with `pytest.importorskip(...)` or imported lazily.

## Experiment identity and storage

Before launching any scientific run, inspect `configs/experiments/registry.toml`.
Never invent a run ID, timestamp-based experiment name, UUID or arbitrary output path.
The canonical identity is `experiment_id / variant_id / seed`. Blocked entries must not
be bypassed.

Ablation sweeps are performed on CONCH v1.5 + TITAN only. Other foundation models
inherit the frozen configuration unless a separate experiment is explicitly approved.

Reusable full teacher caches stay dataset/model-centric. Student/pruned-derived caches
belong under `caches/experiments/<experiment>/<variant>/seed_<seed>/`.

See `docs/experiments.md`.
