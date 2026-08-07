# WSI attention–embedding audit

## Scope

The WSI codebase now has one purpose: measure how a frozen WSI model's existing
tile-attention signal relates to tile-level embedding geometry. It does not:

- train an attention forecaster;
- train ABMIL;
- create pruned feature stores;
- evaluate classification, survival, retrieval, or other downstream tasks;
- treat attention as proof of causal tile utility.

Those stages must be designed after the label-free audit identifies a stable,
interpretable signal.

## Retained data contract

`WSIBag` and HDF5 schema version 1 are intentionally retained. Existing stores
remain readable, including:

- `tile_features [N, D]`;
- optional `coords [N, 2|4]`;
- optional embedded `attention [N]`;
- legacy `label` and metadata fields.

Labels are loaded only for backward compatibility and are never consumed by the
audit.

Attention may be supplied in three ways:

1. embedded in the feature HDF5 store;
2. in a separate legacy HDF5 target store;
3. through a CSV manifest pointing to `.h5`, `.hdf5`, `.pt`, `.pth`, `.npy`, or
   `.npz` artifacts.

### Attention manifest

Minimum schema:

```csv
slide_id,attention_path
TCGA-A1-0001,attention/TCGA-A1-0001.npz
```

Legacy `target_path` is accepted in place of `attention_path`. Optional columns:

```text
coords_path
attention_key
coords_key
tile_axis
tile_slice_start
reduction
```

Relative paths are resolved against the manifest directory. When a native
attention tensor contains layers, heads, or query axes, the tile axis must be
unique or provided explicitly. Non-tile axes are reduced only after optional
axis selections.

## Basic commands

Embedded attention:

```bash
python scripts/analyze_wsi_attention_embeddings.py \
  --feature-store data/wsi_features.h5 \
  --output-dir results/wsi_attention_audit/embedded
```

Separate legacy target store:

```bash
python scripts/analyze_wsi_attention_embeddings.py \
  --feature-store data/early_tile_features.h5 \
  --attention-store data/wsi_attention_targets.h5 \
  --alignment coords \
  --output-dir results/wsi_attention_audit/legacy_targets
```

Multi-layer, multi-head tensor `[layers, heads, tiles]`, selecting the final
layer and averaging heads:

```bash
python scripts/analyze_wsi_attention_embeddings.py \
  --feature-store data/early_tile_features.h5 \
  --attention-manifest manifests/wsi_attention.csv \
  --tile-axis 2 \
  --attention-select 0=-1 \
  --attention-reduction mean \
  --attention-normalization auto \
  --output-dir results/wsi_attention_audit/final_layer
```

Enable an approximate local-redundancy descriptor:

```bash
python scripts/analyze_wsi_attention_embeddings.py \
  --feature-store data/early_tile_features.h5 \
  --attention-manifest manifests/wsi_attention.csv \
  --knn-k 16 \
  --knn-reference-size 4096 \
  --output-dir results/wsi_attention_audit/with_knn
```

## Outputs

```text
<output-dir>/
├── config.json
├── aggregate.json
├── slide_metrics.csv
├── errors.csv
├── slide_details/<slide_id>.json
└── tiles/<slide_id>.npz
```

Per-tile files contain raw attention, normalized attention probability,
coordinates when available, and embedding descriptors. Slide summaries include:

- Pearson and Spearman association;
- linear and quadratic explained variance;
- top-attention overlap with high/low descriptor tails;
- attention-weighted descriptor means;
- descriptor-decile attention mass;
- attention entropy, effective tile count, and top-fraction mass.

The initial descriptors are:

- embedding norm;
- cosine similarity/distance to the slide centroid;
- Euclidean distance to the slide centroid;
- diagonal-Mahalanobis distance to the slide centroid;
- spatial distance to the slide center;
- optional mean cosine distance to approximate nearest neighbours.

## Interpretation rule

Do not infer that attention is causally important from a correlation. The audit
answers narrower questions such as whether high attention preferentially falls
on representative, atypical, spatially peripheral, or locally rare tiles. A
later phase must test intervention-based tile removal and downstream
preservation separately.

### CLS/query attention example

For a tensor `[layers, heads, query_tokens, key_tokens]` with CLS at index 0,
select the final layer and CLS query, then remove CLS from the key-token axis:

```bash
python scripts/analyze_wsi_attention_embeddings.py \
  --feature-store data/early_tile_features.h5 \
  --attention-manifest manifests/native_attention.csv \
  --tile-axis 3 \
  --tile-slice-start 1 \
  --attention-select 0=-1 \
  --attention-select 2=0 \
  --attention-reduction mean \
  --output-dir results/wsi_attention_audit/cls_to_tiles
```

Axis selection is applied before reduction. The command fails when the tile
axis is ambiguous or when a longer token axis is provided without an explicit
tile slice.

## Minimal pipeline wrapper

`scripts/run_wsi_attention_embedding_audit.sh` (an env-var wrapper with no test
coverage of its own) was removed in the offline-EAF refactor; call
`scripts/analyze_wsi_attention_embeddings.py` directly instead — it is the part
that is actually tested (`tests/scripts/test_analyze_wsi_attention_embeddings.py`).
If the feature store does not exist yet, import it first with
`scripts/import_trident_feature_store.py --manifest ... --output-feature-store ...`.

```bash
python scripts/analyze_wsi_attention_embeddings.py \
  --feature-store data/wsi/early_features.h5 \
  --output-dir results/wsi_attention_audit/gigapath_final \
  --alignment coords \
  --attention-normalization auto \
  --attention-manifest data/wsi/manifests/native_attention.csv \
  --tile-axis 3 --tile-slice-start 1 \
  --attention-select 0=-1 --attention-select 2=0
```

This does not call Patho-Bench, train a model, read labels, or create a pruned
store.
