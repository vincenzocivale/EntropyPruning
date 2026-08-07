# Pre-TITAN signal discovery for tile attention

This experiment asks a narrow question:

> Is final TITAN tile attention associated with a signal that is already available before the WSI encoder runs?

TITAN attention is used only as an offline target. The candidate signals are computed from CONCH tile embeddings and coordinates. No TITAN hidden state, query, key, early attention, rollout, or slide token is used as an input.

## Implemented hypotheses

The pipeline evaluates four increasingly structured signals.

1. **Cross-case kNN retrieval**: a tile receives the average final-attention percentile of morphologically similar training tiles from other cases.
2. **Linear attention propensity**: a case-disjoint linear probe separates the top-attention tail from a conservative low-attention background.
3. **Positive/negative semantic prototypes**: each TITAN head group is represented by multiple high-attention and low-attention prototypes. The score is a log-sum-exp similarity likelihood ratio.
4. **Context-conditioned linear propensity**: the linear signal is augmented with cheap, pre-TITAN WSI descriptors: slide-prototype affinity, prototype prevalence, assignment ambiguity, embedding norm, and normalized coordinates. Optional spatial-neighbor residuals can be enabled.

The experiment is deliberately diagnostic. It does not train a new EAF forecaster and does not claim that attention is causal importance.

## Leakage controls

- Train, validation, and test splits are assigned by `case_id`.
- All slides from one case stay in the same split.
- Splits are deterministic and stratified by project.
- TITAN head groups are fitted from the **training split only**.
- kNN retrieval for validation and test therefore cannot retrieve a tile from the same patient.
- Model selection uses validation only. The selected method is then reported on test.

## Head-aware target

The 12 TITAN heads are not averaged into one target. The pipeline computes the median train-set head-rank correlation matrix and applies agglomerative clustering. The default is six groups, motivated by the measured effective head rank.

For group `g`, the target is the normalized mean attention of its member heads:

```text
attention_group[g, tile]
```

Within each WSI it is converted to a percentile. Training labels are:

- positive: top 10%;
- negative: bottom 50%;
- ambiguous: ignored by the binary probes.

The thresholds are configurable.

## Feature views

At least two matched views should be compared:

- `conch_final`: upper bound available after the complete tile encoder;
- `conch_layer2`: deployable early signal for pruning before completing CONCH and before running TITAN.

A feature view is declared with four arguments:

```text
--feature-view NAME MANIFEST FEATURE_SET_ID FEATURE_KEY
```

For separate legacy HDF5 stores, the dataset is normally exposed as `features` and read using `FEATURE_KEY=final`, even when the file semantically contains layer-2 embeddings.

For combined `eaf.wsi.tile_features.v2` files, use `FEATURE_KEY=final` or `FEATURE_KEY=layer2`.

## Primary run

```bash
python -u scripts/wsi_run_attention_signal_discovery.py \
  --attention-manifest \
    artifacts/wsi_attention/titan_real_maps/manifest.csv \
  --attention-key global_to_tiles_mass_share \
  --target-layer -1 \
  --metadata \
    data/wsi/tcga1000_antibench/labels/tcga_project.csv \
  --feature-view \
    conch_final \
    data/wsi/tcga1000_antibench/registry/artifacts.csv \
    conch_v15_final_d768 \
    final \
  --feature-view \
    conch_layer2 \
    data/wsi/tcga1000_antibench/registry/artifacts.csv \
    conch_v15_layer2_d1024 \
    final \
  --output-dir \
    results/wsi_attention_signal_discovery/titan_final_heads \
  --head-groups 6 \
  --positive-fraction 0.10 \
  --negative-fraction 0.50 \
  --retention 0.30 0.40 0.50 0.60 \
  --max-train-tiles 100000 \
  --max-tiles-per-slide 2048 \
  --projection-dim 256 \
  --knn-k 32 \
  --knn-backend auto \
  --prototype-count 16 \
  --slide-prototype-count 16 \
  --bootstrap-replicates 500 \
  --seed 17
```

`faiss` is used automatically when installed; otherwise scikit-learn is used. Install `faiss-cpu` or the appropriate GPU build for faster kNN retrieval.

## Nohup launcher

```bash
mkdir -p logs/wsi_attention_signal

nohup env PYTHONUNBUFFERED=1 \
  python -u scripts/wsi_run_attention_signal_discovery.py \
  ...same arguments... \
  > logs/wsi_attention_signal/titan_signal_discovery.log 2>&1 &

echo $! > logs/wsi_attention_signal/titan_signal_discovery.pid
```

Long phases use progress bars. `--resume` skips feature views that already contain `complete.json` and reuses an existing train-fitted `head_groups.json`.

## Confirmation run without random projection

The primary command uses a deterministic 256-dimensional random projection to reduce memory and make kNN practical. If a signal is found, repeat the winning methods without projection:

```bash
python -u scripts/wsi_run_attention_signal_discovery.py \
  ... \
  --methods linear prototype context_linear \
  --projection-dim 0 \
  --output-dir results/wsi_attention_signal_discovery/titan_final_heads_full_dim
```

This separates a negative biological result from information lost by dimensionality reduction.

## Optional spatial-context ablation

The attention audit found only weak spatial autocorrelation, so spatial features are disabled by default. To test whether they add complementary information:

```bash
--spatial-neighbors 8
```

This adds mean embedding similarity to the eight nearest coordinate neighbors and its residual. It requires a spatial nearest-neighbor index for each slide and is therefore slower.

## Outputs

```text
results/.../
├── analysis_manifest.csv
├── split_counts.csv
├── config.json
├── environment.json
├── head_groups.json
├── median_head_rank_correlation.csv
├── median_head_rank_correlation.npy
├── selection.json
├── selected_view_comparison.csv
├── summary.json
└── views/
    ├── conch_final/
    │   ├── training_sample_manifest.csv
    │   ├── training_data_summary.json
    │   ├── per_slide_group_metrics.csv
    │   ├── per_slide_coverage_metrics.csv
    │   ├── aggregate_group_metrics.csv
    │   ├── aggregate_coverage_metrics.csv
    │   ├── errors.csv
    │   ├── complete.json
    │   └── models/
    └── conch_layer2/
```

The kNN training matrix is not serialized by default, avoiding a large duplicate artifact. Linear models, random projection, and semantic prototypes are saved.

## Metrics

Per head group and WSI:

- Spearman correlation;
- AUPRC for the final top-attention tail;
- recall of final top-attention tiles at each retention;
- final attention mass retained;
- NDCG at each retention;
- lift over random retention.

For the multi-group pruning set, tile scores are converted to within-slide percentiles and combined by their maximum. The pipeline reports:

- mean group recall;
- worst-group recall;
- mean group attention mass;
- worst-group attention mass.

Worst-group metrics are essential: a selector must not achieve good average recall by dropping one specialized TITAN head family.

## Decision rules

Interpret the views in this order.

1. **CONCH final succeeds, layer 2 succeeds**: a useful signal is already available early; proceed to an efficient selector.
2. **CONCH final succeeds, layer 2 fails**: the signal emerges in later CONCH blocks; test intermediate tile-encoder layers.
3. **Only context-conditioned models succeed**: attention propensity depends on WSI composition rather than tile identity alone.
4. **kNN succeeds but linear/prototypes fail**: the signal is local and nonlinear in the embedding space.
5. **All methods fail on CONCH final**: final TITAN attention is not stably recoverable from individual pre-TITAN tile representations; do not repeat a larger EAF regression without changing the objective.

A practical initial criterion is whether, on case-disjoint test data at 50% retention, the method achieves substantial positive lift in both mean and worst-group top-attention recall. The exact operational threshold should be chosen only after the oracle full-vs-pruned TITAN fidelity curve is available.
