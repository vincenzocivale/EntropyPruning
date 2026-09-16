# EAF

This repository now contains one paper-driven pipeline only: train EAF without
downstream labels on HISTAI, distill pruned tile/WSI encoders to reproduce their
unpruned embeddings, and evaluate frozen representations on THUNDER and the public
WSI tasks used by EAGLE.

## Scientific pipeline

```text
HISTAI WSI (unlabeled)
  |
  +-- full tile teacher cache
  |     |
  |     +-- train Tile-EAF: early patch tokens -> final teacher attention
  |     |
  |     +-- distill pruned tile encoder -> full tile embedding
  |             |
  |             +-- pruned tile-input cache
  |
  +-- full TITAN teacher cache ------------------------------+
  |                                                         |
  +-- TITAN source cache built from pruned tile inputs      |
        |                                                    |
        +-- train WSI-EAF: intermediate TITAN state          |
        |                 -> FULL teacher attention           |
        |                                                    |
        +-- distill pruned TITAN + pruned tile inputs -------+
                           -> FULL teacher slide embedding
```

The separation between `source_wsi_root` and `teacher_wsi_root` is an invariant:
WSI-EAF may observe hidden states produced from pruned tile inputs, but its target
remains the full pipeline. This prevents accidental pruned-to-pruned distillation.

## Supported entry points

Data/cache operations:

```bash
python scripts/eaf.py data plan-histai --data-root "$EAF_WSI_ROOT"
python scripts/eaf.py data download-histai --data-root "$EAF_WSI_ROOT"
python scripts/eaf.py cache tile --help
```

Training:

```text
scripts/training/train_tile_eaf.py
scripts/training/distill_tile_encoder.py
scripts/training/train_wsi_eaf.py
scripts/training/distill_wsi_titan.py
```

Teacher/source cache creation:

```text
scripts/features/cache_wsi_teacher.py
```

Evaluation:

```text
scripts/evaluation/evaluate_tile_thunder.py
scripts/evaluation/evaluate_wsi_eagle.py
```

Read `docs/pipeline.md` for the exact cache contracts and launch order, and `docs/experiments.md` for experiment naming, storage and provenance.

## Scope

Historical pruning baselines, supervised THUNDER pretraining, TCGA/HEST/GTEx EAF
pretraining, morphology coarsening, attention-signal discovery, and experimental
side pipelines were intentionally removed. The optional "HISTAI + biological"
experiment must be added as a new, explicit corpus extension rather than reviving
those historical pipelines.
