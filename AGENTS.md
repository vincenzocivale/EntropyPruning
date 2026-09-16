# Repository Guidelines

## Scope

This repository supports one experimental chain only:

`HISTAI -> Tile-EAF -> tile embedding distillation -> WSI-EAF -> WSI embedding distillation -> THUNDER/EAGLE evaluation`.

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
are `evaluate_tile_thunder.py` and `evaluate_wsi_eagle.py`.

Runtime artifacts never belong in git. Use `$EAF_WSI_ROOT` for datasets, caches,
checkpoints, results and logs.

## Tests

Run `pytest` from the repository root. Optional heavy dependencies must be guarded
with `pytest.importorskip(...)` or imported lazily.
