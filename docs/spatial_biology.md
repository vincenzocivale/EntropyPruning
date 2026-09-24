# Does EAF preserve immune and stromal niches?

This is an **external evaluation** of frozen Tile-EAF and WSI-EAF, not a new
training corpus or a biological discovery algorithm. EAF and both distillation
stages continue to use unlabeled HISTAI only. Measured spatial transcriptomics
(ST), rather than teacher attention, supplies the biological reference.

The implementation answers three complementary questions:

1. Can frozen spot embeddings predict measured immune/stromal program scores?
2. Are spatial gradients and within-niche prediction accuracy preserved?
3. Do the WSI tiles retained by EAF cover independently annotated niches, and do
   slide embeddings preserve section-level program statistics/niche proportions?

Program scores are expression proxies, **not** deconvolved cell fractions or
evidence of a causal mechanism. Coverage is not an attribution experiment. ST
usually covers only part of a section; conclusions apply to that observed area.

## Inputs and preparation

Use Python >=3.11, numpy, pandas, scipy and scikit-learn. Plots require matplotlib;
H5AD requires anndata. Encoder export additionally uses the repository's torch,
PEFT, THUNDER/TITAN dependencies and authorized model weights. The evaluator
itself does not download data or model weights. Tests guard optional dependencies.

Prepare a selected breast cancer cohort using the official
[HEST library/tutorials](https://github.com/mahmoodlab/HEST). Freeze release/sample
IDs, QC exclusions, physical crop size, resolution and patient mapping. Use
registered H&E/ST coordinates, not array row/column coordinates. Select signatures
and primary endpoints before inspecting test performance. Import curated GMT gene
sets with their original references/version, e.g. immune-response and stromal
programs; no unvalidated marker panel is silently bundled as biological truth.

Place all data outside git under `$EAF_WSI_ROOT`. The paths in the example config
are templates, not declarations that these datasets already exist.

For the registered HEST breast subset, after accepting the gated dataset terms
on Hugging Face, download only SPA51--SPA80 and materialize the aligned inputs:

```bash
python scripts/evaluation/prepare_hest_biology.py \
  --root "$EAF_WSI_ROOT/datasets/hest_biology_hest1k_breast_v1"
```

The command writes `spots.csv`, `expression.npz`, and `split.json`; it verifies
ST/patch barcode alignment and assigns complete patients to deterministic,
patient-disjoint splits.

### Spot table

`spots.csv` requires:

```csv
spot_id,slide_id,patient_id,cohort,split,x,y,niche,patch_path
study_s1_AAAC,study_s1,study_p1,study,train,1024,1536,stromal,patches/study_s1_AAAC.png
```

- IDs must be globally unique across cohorts. Prefix HEST barcodes with section ID.
- `x,y` are spot **centres** in level-0 image pixels; x is horizontal, y vertical.
- Explicit `train`, `validation`, `test` partitions must all be present. A patient
  and all related sections belong to only one partition. No random spot splits.
- `niche` is optional, independently supplied from ST/annotations; empty entries
  are excluded from niche-stratified analyses. Complete annotations are required
  if slide-level niche proportions are evaluated. `__all__` is reserved.
- `patch_path` is needed only for tile export, absolute or relative to the CSV.
  The HEST preparation script keeps native patch stores without a second image
  copy and writes `patch_store` plus zero-based `patch_index`; the exporter reads
  the `img` dataset from those HDF5 stores. PNG/JPEG/TIFF `patch_path` remains
  supported for non-HEST cohorts.
- Identity checks catch declared overlap, not undisclosed overlap in proprietary
  foundation-model pretraining. Record that uncertainty for each comparator.

### Expression and signatures

Use one cohort-level H5AD (`obs_names = spot_id`, `var_names = gene symbol`) or NPZ:

```python
np.savez_compressed(path, spot_id=np.asarray(ids, dtype=str),
                    genes=np.asarray(gene_symbols, dtype=str), expression=matrix)
```

IDs and genes must be unique; pickle/object arrays are not accepted. Expression
must cover exactly the spot table, in any order. `normalization="counts"` applies
per-spot counts-per-10,000 and log1p. Alternatively declare `log1p_cp10k` explicitly
for already normalized nonnegative data. Set `expression_layer` for a H5AD layer.
Raw counts must include the original assayed gene universe before normalization,
not just signature genes. Zero-library spots must be QC-filtered in all inputs.
The current implementation materializes the expression matrix in RAM; prepare a
bounded cohort, not the entire HEST atlas in one invocation.

GMT is tab-delimited: `program_name<TAB>reference/version<TAB>GENE1<TAB>GENE2...`.
Gene means/standard deviations are fitted on training spots only. A program is
the mean of its standardized usable genes. Missing or constant training genes
are audited; default usable-gene coverage is >=80%, otherwise the program is
skipped. Failure of all programs stops evaluation. Scores are not thresholded
to infer niches. Human/mouse symbols are not mapped automatically.

### Frozen embedding comparators

NPZ files contain `spot_id` strings and `embeddings` of shape `[spots, dimensions]`.
Every model must cover exactly the same spots; rows are aligned by ID. Models
may have different embedding dimensions. For WSI, supply `slide_id, embeddings`
covering every section in the spot table. Additional methods are configured as
`[[methods]]` in `configs/evaluation/spatial_biology.example.toml`.

Use the same physical crops, tissue masks, QC and splits. Architecture-required
input resizing/normalization may differ and must be recorded. The shared head
is StandardScaler -> PCA -> ridge. Transforms fit training data only. PCA dimension
is common to the compared embedding models, capped at 256, smallest feature
dimension and training rows minus one. Ridge alpha is selected separately for
each model/program by patient-balanced validation MSE. Training sample weights
also balance patients. The selected model is not refitted using test data.

### External ST methods from the literature

Two distinct comparisons are supported:

| Comparator | Input | Interpretation |
| --- | --- | --- |
| Full CONCH vs EAF; UNI or other frozen FMs | Spot embeddings | Representation quality under the same head |
| [BLEEP](https://github.com/bowang-lab/BLEEP), [TRIPLEX](https://github.com/NEXGEM/TRIPLEX) | Test-only expression predictions | Complete supervised ST system vs EAF + head |

Run external ST methods in their official environments on the **same patient
split**, then adapt their output to NPZ with `spot_id`, `genes`, `expression`.
The values must be predicted **log1p(CP10k)** expression on the original reference
gene universe's scale. Do not renormalize a limited predicted gene panel to 10k.
If the method predicts standardized targets, invert its training-fitted transform
first. If that conversion is unavailable, the predictions are not comparable and
must not be imported as this format. All genes used in included signatures must
be predicted. The evaluator applies its frozen training-fitted program transform.

Declare `predictions` instead of `embeddings`, `split_sha256` (SHA256 of spots.csv),
`training_patients`, `validation_patients`, and `test_expression_used=false`.
These assertions and patient membership are checked, but cannot prove how a
third-party checkpoint was trained. Audit its pretraining and reference retrieval
bank too: BLEEP must not retrieve expression from held-out patients. Methods
requiring measured test RNA address a different task and are rejected here.

For every method record `citation` (including revision/checkpoint), `supervision`
and `pretraining_overlap`. Their declarations appear in the report. The code does
not train or reproduce BLEEP/TRIPLEX internally, and does not compare published
leaderboard numbers obtained with different splits. Pruning baselines remain out
of scope. External methods have no tile-retention metric unless an actual,
comparable selection grid is provided; do not manufacture a selection from saliency.

## Export from the trained EAF models

First freeze the cohort, preprocessing, signatures, split and checkpoints. Inspect
`configs/experiments/registry.toml`: the existing
`hest_biological_conch15_titan/final` remains blocked until these choices are made.
Update blockers and mark ready only after review, then commit the declaration.
The implementation never bypasses a blocked experiment.

```bash
python scripts/features/export_spatial_eaf.py tile \
  --manifest "$EAF_WSI_ROOT/datasets/hest_biology/spots.csv" \
  --pruned-checkpoint /absolute/path/to/distilled_tile_best.pt \
  --experiment-id hest_biological_conch15_titan --variant-id final --seed 42
```

This runs the same frozen encoder with LoRA/pruning disabled for full and enabled
for EAF. It writes `spatial/tile/eaf.npz` below the registered derived-cache
directory and full features to `caches/spatial/hest/conch_v15/tile/full.npz`.
Existing full features are verified and reused, never overwritten or duplicated
under each experiment. Backbone weights are never updated. Override `--forecaster-checkpoint`
only when relocating the tile forecaster referenced by the checkpoint.

For WSI export prepare a CSV with `slide_id,source_path,teacher_path`. The source
is the existing cache with `coords,tile_embeddings` produced by the **distilled
tile encoder** on the common WSI grid. The teacher is the full tile + full TITAN
output cache from `cache_wsi_teacher.py`. This grid can differ from the spot-centred
crops used for tile prediction. Geometry must be registered to the same image.

```bash
python scripts/features/export_spatial_eaf.py wsi \
  --manifest "$EAF_WSI_ROOT/datasets/hest_biology/wsi_caches.csv" \
  --pruned-checkpoint /absolute/path/to/distilled_wsi_best.pt \
  --experiment-id hest_biological_conch15_titan --variant-id final --seed 42
```

This exports `spatial/wsi/eaf.npz` and `selection.csv`, with full features at
`caches/spatial/hest/conch_v15_titan/wsi/full.npz`. TITAN diagnostic
inference records the actual pruning decision and resolves the live model's
grid order back to original input tiles. Colliding grid cells, zero tile vectors
or unsupported preprocessing fail rather than producing misleading maps.
Checkpoint loading requires trusted local checkpoints. Export refuses existing
output directories; inspect and archive partial outputs before retrying.

Manual selection import uses `slide_id,x,y,width,height,kept`, with level-0 tile
**top-left** coordinates, positive dimensions and `kept` equal to 0 or 1 for every
input tile. All selection methods must share the full original grid. A spot is
covered when its centre lies in a retained rectangle; overlaps count once and
rectangles are half-open. Full reference coverage is derived from the grid if
not supplied. This measures WSI tile retention, not tile-encoder patch retention.

## Validate, freeze and run

Copy the example TOML to a versioned protocol config, set real paths and include
the desired competitors. Uncomment WSI inputs once ready. Validation is read-only
and may run while the scientific experiment is still blocked:

```bash
python scripts/evaluation/evaluate_spatial_biology.py validate --config /path/to/protocol.toml
sha256sum /path/to/protocol.toml
```

Record that digest as `protocol_sha256` in the experiment entry before evaluation,
and commit the frozen config/declaration. The config hash locks the method list
and analysis settings; file hashes of all loaded inputs are additionally saved.
Regenerating input files at the same path is a new scientific input and requires
review even though the config hash alone cannot detect it before reading.

```bash
python scripts/evaluation/evaluate_spatial_biology.py evaluate \
  --config /path/to/protocol.toml \
  --experiment-id hest_biological_conch15_titan --variant-id final --seed 42
```

Use `--no-plots` on minimal CPU environments. Completed evaluations cannot be
overwritten. No scientific run is launched by merely adding this code.

## Results and interpretation

Canonical results live under
`$EAF_WSI_ROOT/results/wsi_eaf/evaluation/hest_biological_conch15_titan/final/seed_42/`:

- `results.csv`: per-section program Pearson/Spearman, RMSE, edge-difference RMSE,
  absolute Moran's I error, with whole-section and annotated-niche strata.
- `coverage.csv`: eligible/retained spot counts, coverage, relative coverage and
  complete niche loss. Uncovered spots outside the original grid are reported,
  not counted as pruning losses.
- `patient_summary.csv`: section-averaged patient metrics and paired deltas to
  the reference with 95% patient-bootstrap intervals (default 2,000 resamples).
- `program_targets.npz`, `predictions_*.npz`, `program_transform.npz`: held-out
  targets/predictions and frozen score transformation.
- Optional `slide_targets.npz`, `slide_predictions_*.npz`: predictions of program
  mean/standard deviation and proportions of training-observed niche labels;
  patient summaries report absolute error. These describe the ST-observed area,
  not an assertion about unmeasured parts of the WSI.
- `protocol.json`, `run.json`, `summary.json`: input hashes, signature exclusions,
  chosen alphas, configuration, git/registry/checkpoint provenance and skips.
- `report.md`, `figures/*.png` and `*.pdf`: measured/full/EAF/competitor maps on
  shared colour scales. Up to six sections are selected by sorted ID, not quality.

Spatial metrics use a symmetric six-nearest-neighbour graph per section/stratum,
dropping edges longer than twice the median nearest-neighbour distance. Moran's
I is a binary undirected-edge statistic; similar Moran's I alone does not imply
correctly located biology. Correlations with constant vectors or fewer than three
spots, and graph metrics without edges, are explicitly undefined. Fewer than two
paired test patients yields no confidence interval. No spot-level significance
tests, equivalence claims or automatic biological-discovery claims are produced.

Choose primary programs/endpoints in advance; the many exploratory metrics are
not multiplicity-corrected hypothesis tests. Report failures and low-prevalence
niches as well as averages. Replicate in an independent cohort before making
general biological claims. Preserve patient grouping when combining cohorts.

The ridge/PCA approach is inspired by HEST, but this program/niche protocol is
**not a reproduction of the official 50-HVG HEST benchmark**. Encoder export
timings include the first batch and omit I/O; they are diagnostic only, not a
fair end-to-end efficiency comparison with external cache producers.

## Preliminary findings (exploratory, tile-level, 2026-09-20)

These runs used `hest_biological_conch15_titan_tile_exploratory` (spot-level,
tile only, no UNI/WSI) plus ad hoc scripts under `scratchpad/`, not the frozen
`hest_biological_conch15_titan/final` protocol. They are recorded here as
evidence, not as the committed scientific result; `final` is still blocked
(WSI-EAF checkpoint missing, UNI comparator missing).

**Tile-level full-vs-EAF (`src02_keep15`), spot-level program-score regression.**
On the frozen 30-section/10-patient breast subset, single-split Pearson was
statistically unusable (2 test patients): full and EAF both hovered near zero
with sign flipping across the fixed split. A leave-one-patient-out-style 5-fold
patient CV on the same 10 patients showed the same instability. Extending the
prepared cohort to the full same-platform/same-organ breast-IDC block available
in HEST (`SPA51`-`SPA154`, 104 sections, 31 patients, `--first 51 --last 154`)
stabilized the estimate: mean test Pearson across all 31 patients was
0.062 (full) vs 0.067 (eaf) for `immune_activation`, and 0.112 (full) vs 0.109
(eaf) for `stromal_ecm` — both programs now clearly positive on average, and
full vs EAF differ by far less than the between-patient standard deviation
(~0.10-0.14). `prepare_hest_biology.py` gained `--ids`, `--cohort` and
`--append` to support this and further cohort extensions without overwriting
existing spots.

**Cross-cancer extension.** Kidney SCCRCC (Visium, `INT1`-`INT24`, 24
patients, dataset "Tertiary lymphoid structures... renal cell cancer") was
appended via `--cohort hest_kidney_rcc_visium --append`, giving 55 patients
across two organs/platforms (breast=ST, kidney=Visium; the two are perfectly
confounded in this design and cannot be separated). One zero-library spot
(`SPA132:018x020`) was dropped from `spots.csv`/`expression.npz` and the
matching row from both embedding caches as a one-off QC fix; `read_expression`
already refuses zero-library spots by contract, so future cohort appends must
carry the same check.

**Gene-identifier bug found and fixed (`prepare_hest_biology.py`).** The
first kidney append silently produced all-zero expression for every
signature gene on every kidney spot (confirmed: `sum()==0` across all 43
Ensembl IDs, 73,813 spots), which made every kidney-patient Pearson/Spearman
`NaN` — invisible in the aggregate because pandas `.mean()` skips `NaN`, so
the "55-patient" summary printed at the time was silently just the 31 breast
patients under a wrong label. Root cause: the breast ST cohort's `st/*.h5ad`
uses Ensembl IDs directly as `var_names`, but the kidney Visium cohort's
`st/*.h5ad` uses gene *symbols* as `var_names` with the Ensembl ID in
`var["gene_ids"]` — `prepare_hest_biology.py` used `var_names` unconditionally
for every cohort, so kidney's symbol-keyed columns never matched any
Ensembl-ID-keyed signature gene during the union/reindex step. Fixed by
preferring `var["gene_ids"]` when present (`current_genes = a.var["gene_ids"]
if "gene_ids" in a.var else a.var_names`), with a duplicate-identifier guard.
Re-running `prepare_hest_biology.py --first 51 --last 154` (fresh) then
`--ids INT1..INT24 --cohort hest_kidney_rcc_visium --append` after the fix
gave real, non-degenerate kidney expression (13.2% nonzero fraction across
signature genes, max count 40) and a genuine 55-patient result:

| cohort | target | full | eaf |
|---|---|---|---|
| pan-cancer (55) | immune_activation | 0.062 | 0.058 |
| pan-cancer (55) | stromal_ecm | 0.113 | 0.103 |
| breast (31) | immune_activation | 0.064 | 0.058 |
| breast (31) | stromal_ecm | 0.133 | 0.121 |
| kidney (24) | immune_activation | 0.060 | 0.058 |
| kidney (24) | stromal_ecm | 0.087 | 0.081 |

Both organs independently show positive mean Pearson for both programs, and
full vs EAF stay within 0.004-0.012 of each other in every row — the
conclusion (full and EAF are statistically indistinguishable, well inside
between-patient noise) is unchanged from the pre-fix breast-only estimate,
but is now genuinely supported by a second independent organ rather than by
a bug that happened to exclude it. Any further cohort append must be
spot-checked the same way (`expression[cohort_mask][:, signature_gene_cols]`
should be a mix of zero and positive counts, never uniformly zero) before
trusting downstream regression numbers — a silently-all-`NaN` cohort does
not raise an error anywhere in this pipeline.

**Patient/batch confound check.** Regressing out each test patient's own mean
before computing Pearson ("within-patient" residual correlation) showed most of
the raw signal in the top-50-HVG regression (scanpy `flavor="seurat"`,
per-fold) was inter-patient level, not real intra-section spatial biology: e.g.
pooled r=0.16 dropping to r=-0.05 for one gene. A small number of genes
(notably `KDM4D`/ENSG00000186280) retained genuine within-patient signal
(full 0.248, eaf 0.235) — full and EAF tracked each other closely on both the
confounded and the deconfounded metric. `evaluate_spatial_biology_patient_cv.py`
implements the patient-rotation CV; the within-patient demeaning check was run
ad hoc and is not yet a script.

**Single-cell swap-oracle pilot (tile-level token pruning).** Motivated by the
question "does token-level pruning discard rare functional states within a
tile", one Xenium breast section (`TENX191`, 280-gene targeted panel, not part
of the ST/Visium spot cohorts above) was used to build genuine single-cell
pathway vectors from `transcripts/*.parquet` (`cell_id`, `he_x`/`he_y` already
H&E-registered) and `xenium_seg/*_xenium_cell_seg.parquet`, filtered to
`is_gene=True` and `cell_id != "UNASSIGNED"`, QC'd at >=10 transcripts/cell
(202k/208k cells kept). HEST's `st/*.h5ad` for Xenium is a hexbin aggregate,
**not** single-cell — the per-cell data lives only in `transcripts/` and
`xenium_seg/`. The narrow 541-var/280-gene panel covers only 6/23 and 4/22 of
the curated breast immune/stromal marker symbols above; a panel-specific
signature was substituted (21 immune genes: CD3E, CD8A, TRAC, CD68, CD163,
MS4A1, BANK1, IL7R, IL2RG, CCL5, CXCL10, CTLA4, TIGIT, CCR7, ITGAX, LYZ, CD27,
CD83, TNFRSF13C, CYTIP, BIRC3; ~19 stromal/vascular genes: FBLN1, PDGFRB,
TAGLN, MYH11, MYLK, RGS5, COL4A1, COL17A1, LAMA2/3/4, LAMB1/3, LAMC2,
MMP2/9/11/12/14, TIMP1, SFRP1/4, HSPG2, VWF, PECAM1, CLEC14A, PLVAP, FLT1).

Cells were mapped to the live EAF-Tile ViT token grid: CONCH v1.5/TITAN resizes
the 224x224 HEST tile crop to 448x448 (`Resize(448)+CenterCrop(448)`), patch16
gives a 28x28=784-token grid (confirmed at runtime: block output
`[1, 785, 1024]`), so each token covers an 8x8 px box in the original 224px
crop; a cell's tile-local pixel position divided by 8 gives its token index.
Real per-token EAF forecaster scores (`src02_keep15`, keep=117/784) were
extracted via a forward hook on `raw_backbone.blocks[prune_layer]` (the exact
tensor scored by `PrunedLoRAEncoder._pruning_hook`).

A facility-location coverage objective `U_bio(S) = mean_c[w_c * max_{s in S}
sim(p_c, rep_s)]` (`sim = exp(-euclidean/sqrt(dim))`, `w_c` inverse local
density = rarity) was computed per tile, comparing EAF's real top-117 token
selection against 1-swap-at-a-time greedy alternatives that trade a kept token
for a discarded one when it improves `U_bio` beyond a gain threshold at
bounded forecaster-score cost. Run at full scale (974 tiles with >=30 mapped
cells) at three threshold settings, with two independent state representations
(2D program score, 15D whitened PCA over the full 280-gene panel — results
were near-identical, ruling out "representation too coarse" as the
explanation):

| setting | gain/cost thresholds | tiles w/ >=1 swap | mean swaps | mean % tokens changed | mean U_bio gain |
|---|---|---|---|---|---|
| strict | 0.02 / 0.02 | 1.5% (0.9% rich) | 0.02 | 0.01% | 0.0006 |
| medium | 0.01 / 0.05 | 10.8% (7.9% rich) | 0.12 | 0.10% | 0.0040 |
| loose  | 0.005 / 0.15 | 53.7% (49.9% rich) | 1.02 | 0.86% | 0.0228 |

Even under the loosest setting, fewer than 1% of kept tokens per tile benefit
from a swap on average — far below the 10-20% "operating zone" that would
justify a bio-guided token-selection fine-tune at the tile level. **Read
together with the pan-cancer spot-level result above (full vs EAF
indistinguishable, well within between-patient noise, across two organs and
55 patients), this is convergent evidence, from an independent single-cell
data source, that EAF-Tile's existing pruning is not discarding rare
biological states it would need to be corrected for** — not proof of zero
loss, but no signal of loss found by any of the three independent
tile-level tests run (spot-level program regression, per-gene/HVG regression,
single-cell facility-location swap oracle). The corollary is that a
biology-guided EAF fine-tune, if pursued, has a much better-motivated target
at the **WSI level** (whole-tile retention across a slide, where a rare niche
can disappear entirely if its tile is dropped) than at the tile-token level —
but that path is blocked on training a base WSI-EAF checkpoint first
(`checkpoints/wsi_eaf/` is currently empty; see the blocked
`wsi_source_ablation_conch15_titan` / `hest_biological_conch15_titan`
registry entries).

**Two real bugs fixed in `src/evaluation/spatial/extract.py`** while running
the above (both affect any HEST-style manifest using `patch_store`/
`patch_index` instead of `patch_path`, i.e. every cohort prepared by
`prepare_hest_biology.py`): (1) `main()` unconditionally resolved
`spots.patch_path`, crashing before reaching `export_tile_pair`'s
`patch_store` fallback; (2) `load_image()`'s `"patch_store" in row` checked
row *values*, not field names, on an `itertuples` row, so it always fell
through to the (absent) `patch_path` branch. Fixed to `if "patch_path" in
spots:` (column-conditional resolution) and `getattr(row, "patch_store",
None)` respectively; both are file-format bugs, not scientific-logic changes,
and were required before any `patch_store`-based export could run at all.
