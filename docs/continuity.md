# Project status and handoff

This page records the verified state on 16 September 2026. It is a handoff for
work on this repository; the live source of truth remains the read-only audit.
Run it before selecting work:

```bash
export EAF_WSI_ROOT=/data2/home/vcivale/data/WSI
python scripts/eaf.py experiments audit --data-root "$EAF_WSI_ROOT"
```

The command rewrites `$EAF_WSI_ROOT/results/experiment_catalog/catalog.json`.
It scans the runtime root and the repository's legacy `checkpoints/` and
`logs/` paths, without loading tensors or changing data. Do not commit its
output.

## Working rules

- `$EAF_WSI_ROOT` is `/data2/home/vcivale/data/WSI` on the current host.
  Set it explicitly; do not infer it from the repository path.
- New derived arrays are `.npyd` directories. HDF5 remains read-only input
  compatibility. Use `eaf.py cache convert-numpy` to migrate an existing
  derived HDF5 artifact after validation.
- A frozen teacher runs only while building a cache. Tile-EAF training obtains
  the configured early representation online; it must not invoke the full
  frozen teacher for every epoch.
- Keep raw WSI in `sources/`, keep pretraining and downstream manifests
  separate, and do not move current TCGA or HEST assets.
- Each new run must write a `summary.json` under `results/` with code revision,
  data and split, cache, model revision, layer, keep ratio, seed, checkpoint,
  metrics and timings.

## Verified inventory

The audit on the date above found 26 runs: 7 complete, 18 partial and 1 marked
non-comparable. It also found 36 cache metadata/manifest files, 29 downstream
label files, five profiling files and 27 rows in the legacy WSI baseline CSV.
A `complete` run has a best checkpoint and machine-readable summary. It does
not establish downstream quality by itself.

| Area | Evidence | Status |
| --- | --- | --- |
| Tile CONCH v1.5 cache | HISTAI subsets; cache manifests | present |
| Tile UNI2-h cache | HEST/THUNDER manifest | present only |
| Tile CONCH forecaster | `conch_v15_src00`, `titan_src01` | complete |
| Tile CONCH distillation | five 10%/20% pruned runs | complete |
| WSI TITAN cache | HISTAI and six TCGA cohorts | present |
| WSI TITAN forecaster/distillation | legacy checkpoints | partial |
| WSI downstream labels | BRCA, COAD, LUAD, LUSC, READ, STAD | 29 task files |
| WSI evaluation | baseline result tables | partial; no complete EAF comparison |

The completed Tile-EAF runs are useful starting points, not a model selection:

| Run | Verified metric |
| --- | --- |
| `conch_v15_src00` forecaster | best validation KL 0.0986; maximum validation rho 0.7948 |
| `titan_src01` forecaster | best validation KL 0.0932; maximum validation rho 0.8053 |
| `src00`, `src01`, `src02` distillation | five complete 10%/20% variants; best validation loss ranges 0.0283–0.0671 |

The directory named `hidden_layer2_final_CONTAMINATED_bak_20260821` is marked
non-comparable. Do not report it or use it to choose settings.

## Gaps and order of work

| Model / family | Next required evidence |
| --- | --- |
| CONCH v1.5 Tile-EAF | paired downstream evaluation and end-to-end latency/quality at each selected keep ratio |
| UNI2-h Tile-EAF | forecaster, distillation, evaluation, quality and cost after its existing cache is validated |
| Virchow2, H-Optimus-1, Prov-GigaPath Tile-EAF | cache first, then the same forecaster/distillation/evaluation path |
| TITAN WSI-EAF | complete summaries, paired EAF versus baseline evaluation, quality and cost |

Proceed in this order:

1. Regenerate the audit and validate every cache used by the next run.
2. Profile one representative CONCH run before changing it. The profile records
   data wait, teacher, forecaster, backward and metrics time; change one factor
   and profile the same data and parameters again.
3. Finish paired Tile-EAF evaluation for the completed CONCH settings, including
   end-to-end inference latency and retained-tile ratio.
4. Finish paired TITAN WSI evaluation on the existing 29 label tasks, recording
   skipped tasks and patient-level splits.
5. Only then build and benchmark the missing encoder caches, one encoder at a
   time. Do not launch a multi-model sweep without a measured baseline.

## Entry points

`python scripts/eaf.py ...` is the operational CLI. Specialized commands are
organized under `scripts/data`, `scripts/features`, `scripts/training`,
`scripts/evaluation` and `scripts/analysis`. The former HISTAI queue
orchestrator, single-purpose GDC commands, one-off inventory/materialization
scripts, shell smoke wrappers and the obsolete UNI classifier have been
removed. Compose the supported commands above instead of restoring wrappers.
