# Validating morphology-preserving WSI coarsening

## Scope

This experiment leaves the EAF tile-pruning method unchanged. EAF still reduces
patch tokens inside each tile and emits one standard tile embedding. The new
module acts only between tile feature extraction and the WSI encoder:

```text
WSI tiles -> tile encoder + EAF -> tile embeddings + coordinates
          -> morphology/topology coarsener -> shorter tile sequence
          -> ABMIL / TITAN / another WSI encoder
```

The scientific hypothesis is that a WSI contains two distinct forms of
redundancy:

1. **intra-tile redundancy**, already addressed by EAF;
2. **inter-tile redundancy**, caused by large spatially contiguous areas with
   similar morphology.

The proposed coarsener removes repeated tile embeddings while protecting
spatially connected morphological regions, transition boundaries,
heterogeneous regions, and small rare regions.

## What the patch adds

- `src/data/wsi/morphology_coarsening.py`
  - `random`: retention-matched negative control;
  - `spatial_fps`: spatial-coverage baseline;
  - `morphology_fps`: morphology-only diversity baseline;
  - `morphology_topology`: proposed morphology- and topology-preserving method.
- `scripts/create_wsi_morphology_coarsened_store.py`
  - materializes a standard HDF5 WSI feature store containing only real selected
    tiles;
  - preserves coordinates, labels, attention values, and original tile order;
  - writes per-slide and aggregate diagnostics.
- `scripts/evaluate_wsi_coarsened_store_agreement.py`
  - evaluates a frozen ABMIL checkpoint on the full and coarsened feature stores;
  - reports prediction, logit, and bag-embedding agreement;
  - optionally reports retained teacher-attention mass when full attention and
    coordinates are available.

No WSI foundation model, tile encoder, or EAF component is modified.

## Proposed strategy

For `morphology_topology`, the script:

1. builds a spatial graph from tile coordinates;
2. computes cosine similarity between neighboring tile embeddings;
3. forms connected morphology regions by merging sufficiently similar spatial
   neighbors;
4. assigns a fixed global budget across regions using:
   - sublinear region size;
   - within-region heterogeneity;
   - boundary complexity;
   - region rarity;
5. selects real tile embeddings within each region using a prototype, boundary
   protection, and diversity sampling.

If the number of connected regions is no larger than the budget, every region
receives at least one representative. If regions outnumber the budget, regions
are prioritized by the same morphology-aware allocation score.

The optional WSI attention stored in the input file is used only for evaluation.
It never affects selection.

## Minimal smoke test

```bash
python scripts/create_synthetic_wsi_feature_store.py \
  --output /tmp/wsi_full.h5 \
  --n-slides 8 \
  --feature-dim 64 \
  --min-tiles 64 \
  --max-tiles 128 \
  --overwrite

python scripts/create_wsi_morphology_coarsened_store.py \
  --input-feature-store /tmp/wsi_full.h5 \
  --output-feature-store /tmp/wsi_morphology_25.h5 \
  --strategy morphology_topology \
  --keep-ratio 0.25 \
  --overwrite
```

The synthetic store is only a software smoke test. Its random features do not
model spatially coherent histology and must not be used to judge the scientific
hypothesis.

## First real benchmark

Use the same EAF-derived tile feature store for every method. Run three retention
levels and all controls:

```bash
INPUT=data/wsi/tcga1000_antibench/full_eaf_features.h5
OUT=results/wsi_morphology_coarsening
mkdir -p "$OUT/stores" "$OUT/reports"

for ratio in 0.50 0.25 0.10; do
  tag=${ratio/./p}
  for strategy in random spatial_fps morphology_fps morphology_topology; do
    python scripts/create_wsi_morphology_coarsened_store.py \
      --input-feature-store "$INPUT" \
      --output-feature-store "$OUT/stores/${strategy}_${tag}.h5" \
      --strategy "$strategy" \
      --keep-ratio "$ratio" \
      --seed 17 \
      --report-csv "$OUT/reports/${strategy}_${tag}_slides.csv" \
      --summary-json "$OUT/reports/${strategy}_${tag}_summary.json" \
      --overwrite
  done
done
```

The default `--merge-quantile 0.25` merges the upper 75% of neighboring
similarities within each slide. This is deliberately a starting point, not a
fixed methodological claim. Sweep at least:

```text
merge_quantile in {0.10, 0.25, 0.50}
connectivity in {4, 8}
rarity_weight in {0, 1, 2, 4}
boundary_weight in {0, 1, 2}
```

Tune only on the training/validation slides. Lock the configuration before the
held-out WSI evaluation.

## Frozen-ABMIL agreement

Train the full-store ABMIL checkpoint once using the existing script, then reuse
that checkpoint for every coarsened store:

```bash
python scripts/evaluate_wsi_coarsened_store_agreement.py \
  --full-feature-store "$INPUT" \
  --coarsened-feature-store "$OUT/stores/morphology_topology_0p25.h5" \
  --abmil-checkpoint results/wsi_abmil/checkpoints/best.pt \
  --output-csv "$OUT/reports/morphology_topology_0p25_agreement.csv" \
  --per-slide-csv "$OUT/reports/morphology_topology_0p25_agreement_slides.csv" \
  --overwrite
```

This tests whether the same frozen WSI model tolerates the shorter sequence. A
second experiment should retrain or fine-tune the downstream head on each
coarsened training store to distinguish representation preservation from
adaptation to the new sequence distribution.

## TITAN or another WSI FM

The generated store uses the repository's canonical `WSIBag` representation:

```text
tile_features [K, D]
coords        [K, 2] or [K, 4]
```

It can therefore replace the full tile feature store in an existing TITAN or WSI
FM inference pipeline. Use the same frozen WSI checkpoint for the primary
agreement experiment. Report:

- slide-embedding cosine similarity;
- prediction agreement and probability KL divergence;
- downstream AUROC / macro-F1;
- measured latency, memory, and throughput;
- retained tile ratio and effective WSI encoder speed-up.

Do not infer WSI encoder speed-up from tile ratio alone. Measure it because the
actual gain depends on the WSI architecture and implementation.

## Diagnostics produced by the selector

The per-slide CSV includes:

- `projected_feature_coverage`: mean maximum cosine similarity from every full
  tile to a selected tile; higher is better;
- `mean_normalized_spatial_distance`: mean distance from every tile to the
  nearest selected coordinate after slide-wise normalization; lower is better;
- `region_coverage`: fraction of discovered connected regions represented;
- `significant_region_coverage`: coverage after applying `min_region_size`;
- `rare_region_coverage`: coverage of the most morphologically isolated regions;
- `boundary_top_recall`: recall of tiles with the strongest local morphology
  transitions;
- attention-mass metrics, when teacher attention is available.

These diagnostics are secondary. The primary criterion is preservation of the
full WSI model's embedding and downstream prediction at lower measured compute.

## Decision rules

The biological prior is supported only if `morphology_topology`, at the same
retention, improves over both:

- `morphology_fps`, which controls for feature diversity without spatial tissue
  structure;
- `spatial_fps`, which controls for spatial coverage without morphology.

The strongest expected evidence is:

1. better frozen-model slide embedding and prediction agreement;
2. better worst-group downstream performance at 10-25% retention;
3. higher rare-region and boundary coverage;
4. a measurable WSI encoder latency reduction.

If `morphology_topology` does not beat `morphology_fps`, the spatial/topological
histology prior has not added value. If it preserves diagnostics but not WSI
predictions, the region construction or budget allocation is not aligned with
the downstream representation and should not be promoted as a paper
contribution.

## Required ablations for a paper claim

- proposed method without rarity weighting;
- proposed method without boundary weighting;
- morphology-only versus morphology + spatial connectivity;
- fixed one-per-region versus adaptive regional budget;
- EAF tile features versus full tile-encoder features;
- at least two WSI aggregators or WSI FMs;
- held-out tumour/dataset evaluation;
- multiple seeds for stochastic baselines.

The primary paper comparison should present the full hierarchical pipeline:

```text
full tile encoder + full WSI encoder
EAF tile encoder + full WSI encoder
EAF tile encoder + generic WSI tile reduction
EAF tile encoder + morphology/topology coarsening
```
