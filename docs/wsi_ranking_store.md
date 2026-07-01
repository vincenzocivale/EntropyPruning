# WSI Ranking Store

`WSIRankingStore` is the rank-only output format for the WSI EAF pipeline.
It stores one file per slide in an output directory, without materializing a
new pruned feature HDF5.

Implemented in [src/data/wsi/ranking_store.py](/data2/home/vcivale/projects/imaging/EAF/src/data/wsi/ranking_store.py).

## Purpose

Use this mode when you want the EAF forecaster to score and rank tiles from an
early feature store, but you do not want to write a second feature store with
pruned tile embeddings.

Each per-slide ranking file contains:

- `slide_id`
- `coords` if available in the input feature store
- `scores` of shape `[N]`
- `ranks` of shape `[N]`, aligned to original tile order
- `selected_indices` for each keep ratio, ordered by descending score
- `original_order_indices` for each keep ratio, sorted by original tile order
- `metadata_json`

Tile features are never stored in the ranking output.

## CLI

Generate rankings with:

```bash
python scripts/rank_wsi_tiles_from_feature_store.py \
  --input-feature-store data/features_layer2.h5 \
  --forecaster-checkpoint checkpoints/exp001_importance_forecaster/best_wsi_tile_importance_forecaster.pt \
  --output-dir rankings/exp001 \
  --keep-ratios 0.05 0.10 0.25 \
  --device cuda
```

Optional arguments:

```text
--slide-ids-file
--batch-size
--num-workers
--output-format npz|parquet
--overwrite
```

Default output format is `npz`. `parquet` is supported when the runtime has
`pandas` and a parquet engine available.

## Ranking Conventions

- `scores[i]` is the forecaster score for tile `i` in the original tile order.
- `ranks[i]` is the 1-based rank of tile `i`, where `1` is the highest score.
- `selected_indices[ratio]` contains the top tiles for that keep ratio,
  ordered by descending score.
- `original_order_indices[ratio]` contains the same tile indices, but sorted
  ascending by original tile order.

This lets downstream code choose either:

- score order for top-k inspection and visualization
- original tile order for deterministic pruning/materialization later

## Metadata

Each slide file stores JSON metadata with at least:

- ranking timestamp
- forecaster checkpoint path
- input feature store path
- keep ratios
- feature dimension
- tile count
- output format
- model type

Source bag metadata from the input store is preserved and extended with these
ranking-specific fields.

## Compatibility

This ranking store is additive. It does not modify or replace
`scripts/create_pruned_wsi_feature_store.py`, and the existing pruning pipeline
continues to materialize HDF5 feature stores exactly as before.
