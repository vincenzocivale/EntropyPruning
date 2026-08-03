# WSI preprocessing, multi-layer tile embeddings and attention audit

This pipeline is label-free until downstream evaluation is explicitly added. It is designed to:

1. run resumable TRIDENT tissue segmentation and coordinate extraction;
2. extract the second transformer-block CLS embedding and final tile embedding in the same forward pass;
3. store both embeddings once, in one per-slide HDF5 file;
4. extract WSI embeddings and native tile attention where the public model API exposes it;
5. test whether WSI attention favors majority/dense regions of tile-embedding space.

## Storage contract

New tile-feature files use `eaf.wsi.tile_features.v2`:

```text
coords                    [N,2] int32
embeddings/layer2         [N,D2] float16 by default
embeddings/final          [N,Df] float16 by default
```

Coordinates are not duplicated between feature sets. Writes are atomic and a file is reusable only when `complete=true`.

New WSI-output files use `eaf.wsi.fm_output.v1`:

```text
slide_embedding           [D]
coords                    [N,2]
attention/logits          [N]       # FEATHER
attention/probs           [N]       # FEATHER pooling weights
```

The default compression is LZF because it is fast and lossless. Floating outputs are stored as FP16 unless `--storage-dtype float32` is requested.

## Supported models

### Tile encoders

- `conch_v15`: primary adapter. It loads CONCH v1.5 through `MahmoodLab/TITAN`, hooks transformer block index 1, and obtains the final 768-d output in the same forward pass.
- `timm`: generic ViT adapter for compatible timm/Hugging Face models such as UNI/UNI2, Virchow, GigaPath tile encoders and H-optimus. Model-specific image normalization and resolution must be configured correctly.

### WSI encoders

- `feather`: primary attention model. Saves native gated-ABMIL logits, normalized pooling weights, and the slide embedding.
- `titan`: saves the official slide embedding from CONCH v1.5 features and coordinates. The public API does not expose a canonical native tile-attention tensor; the output records this limitation rather than fabricating one.
- `gigapath`: saves the official slide embedding from native 1536-d GigaPath tile features. Its public API does not expose a canonical attention matrix.

The first hypothesis test must therefore use FEATHER. TITAN and GigaPath are included for later representation comparisons and intervention-based attribution.

## Installation

```bash
pip install -r requirements-wsi-pipeline.txt
```

Install TRIDENT from its official repository for preprocessing. Request access to gated models and set:

```bash
export HF_TOKEN=...
```

## 1. Preprocess missing WSI

For the existing TCGA1000 run, reuse the 1,056 coordinate files. Run this only for the 424 missing slides or for a new dataset.

```bash
python scripts/wsi_preprocess.py \
  --trident-repo /path/to/TRIDENT \
  --wsi-dir data/wsi/tcga1000_antibench/source/wsis \
  --job-dir data/wsi/tcga1000_antibench/processed/trident/20x_512px_stride512 \
  --stages seg coords \
  --segmenter hest \
  --mag 20 \
  --patch-size 512 \
  --overlap 0 \
  --gpus 0
```

TRIDENT provides progress bars for slide-level preprocessing; the wrapper provides a stage-level progress bar and preserves terminal output.

## 2. Extract layer-2 and final CONCH embeddings

`slides.csv` must contain `slide_id,wsi_path`. The coordinate registry must contain `slide_id,path`.

Start with five slides:

```bash
python scripts/wsi_extract_tile_embeddings.py \
  --slides data/wsi/tcga1000_antibench/registry/slides.csv \
  --coords-registry data/wsi/tcga1000_antibench/registry/coords.csv \
  --output-dir data/wsi/tcga1000_antibench/processed/tile_features/conch_v15_multilayer_v2 \
  --encoder conch_v15 \
  --device cuda \
  --batch-size 64 \
  --num-workers 4 \
  --storage-dtype float16 \
  --compression lzf \
  --max-slides 5
```

The script shows progress per slide and per patch batch. Completed files are skipped on rerun.

For the current 1,056-slide analysis, the already validated `conch_v15_final_d768` files can be used directly. Re-extraction is needed only if layer-2 embeddings must be regenerated with documented provenance.

## 3. Extract FEATHER attention and WSI embeddings

Build a feature manifest with `slide_id,path` referencing the validated CONCH final HDF5 files, then run:

```bash
python scripts/wsi_extract_fm_outputs.py \
  --feature-manifest data/wsi/tcga1000_antibench/registry/conch_v15_final_manifest.csv \
  --output-dir artifacts/wsi_attention/feather24k_conch_v15 \
  --model feather \
  --feature-key final \
  --device cuda \
  --storage-dtype float16 \
  --max-slides 5
```

Then scale to the discovery set and finally all available slides by removing `--max-slides`.

TITAN embeddings can be extracted from the same features:

```bash
python scripts/wsi_extract_fm_outputs.py \
  --feature-manifest data/wsi/tcga1000_antibench/registry/conch_v15_final_manifest.csv \
  --output-dir artifacts/wsi_embeddings/titan_conch_v15 \
  --model titan \
  --feature-key final \
  --patch-size-level0 512 \
  --device cuda
```

TITAN output has no `attention/probs`; it cannot be passed to the majority-attention script.

## 4. Validate or refute the majority hypothesis

The primary hypothesis is:

> Within each WSI, tiles in denser regions of CONCH embedding space receive greater FEATHER attention.

Run:

```bash
python scripts/wsi_analyze_majority_attention.py \
  --feature-manifest data/wsi/tcga1000_antibench/registry/conch_v15_final_manifest.csv \
  --wsi-output-manifest artifacts/wsi_attention/feather24k_conch_v15/manifest.csv \
  --metadata data/wsi/tcga1000_antibench/labels/tcga_project.csv \
  --output-dir results/wsi_attention_majority/feather24k_conch_v15 \
  --feature-key final \
  --attention-key probs \
  --knn 16 32 64 \
  --clusters 4 8 16 \
  --n-permutations 200 \
  --discovery-fraction 0.125 \
  --seed 17
```

Outputs:

```text
per_slide.csv
per_cluster.csv
errors.csv
aggregate.json
```

The confirmation verdict is positive only when both conditions hold:

- the lower 95% case-bootstrap bound of the median attention–kNN-density Spearman correlation is above zero;
- the lower 95% bound of dominant-cluster enrichment is above one.

High total attention mass in the dominant cluster alone is not evidence because a large cluster receives high mass even under uniform attention.

## Computational choices

- One tile-encoder forward produces both layer-2 and final embeddings.
- Per-slide HDF5 avoids monolithic rewrites and enables resume.
- FP16 halves storage; analysis converts to FP32.
- Full self-attention matrices are rejected above a safety threshold. Prefer per-tile attention or CLS-to-tile projections.
- Nearest-neighbor density is computed within each slide and never by concatenating all WSI tiles.
- Clustering fits on at most 20,000 sampled tiles per slide, then predicts labels for all tiles.
- Bootstrap resamples cases, not individual tiles.

## Important interpretation

A positive FEATHER result shows that its pretrained instance scorer prefers morphologies that are common/dense within slides. ABMIL scores each tile before pooling, so this does not yet prove that the model contextually detects which population is the majority. That stronger claim requires a contextual WSI encoder and prevalence-intervention experiments.

## Real TITAN self-attention extraction

TITAN's documented API returns a slide embedding but the live vision encoder is a
ViT and computes self-attention internally. The pipeline instruments the gated
model during the official `encode_slide_from_patch_features` forward and captures
the **post-softmax attention actually used by the model**.

Two implementations are supported:

1. explicit ViT attention (`Tensor.softmax`, `torch.softmax`, or `F.softmax`);
2. fused PyTorch SDPA, where probabilities are reconstructed exactly from the
   live query/key tensors and the same scale/mask before the original fused
   operator is called.

Extraction requires `model.eval()`. In strict mode, non-zero attention dropout is
rejected. If no supported attention path is observed, extraction fails and prints
the candidate module tree; it never substitutes a geometric proxy.

### Efficient default

The default avoids storing quadratic matrices and writes:

```text
attention/global_to_tokens                 [L,H,T]
attention/received_by_tokens               [L,H,T]
attention/rollout_global_to_tokens         [T]
```

When token-to-tile mapping can be validated, it additionally writes:

```text
attention/global_to_tiles_broadcast        [L,H,N]
attention/global_to_tiles_mass_share       [L,H,N]
attention/received_by_tiles_broadcast      [L,H,N]
attention/rollout_global_to_tiles_*        [N]
auxiliary/tile_to_token                     [N]
auxiliary/tiles_per_token                   [T]
```

The `mass_share` representation divides a token's mass among the input tiles
mapped to that token. The `broadcast` representation is useful for ranking, but
does not conserve total mass when multiple tiles form one TITAN token.

### Smoke test on one slide

```bash
export HF_TOKEN=hf_...

python scripts/wsi_extract_fm_outputs.py \
  --feature-manifest data/wsi/tcga1000_antibench/registry/artifacts.csv \
  --artifact-type tile_features \
  --feature-set-id conch_v15_final_d768 \
  --model titan \
  --feature-key final \
  --patch-size-level0 512 \
  --output-dir artifacts/wsi_attention/titan_real \
  --max-slides 1
```

Inspect the HDF5 attributes `attention_capture_backends`,
`attention_module_names`, `attention_tokens`, and `tile_token_mapping` before
scaling to the full cohort.

### Full matrix for selected slides

The complete matrix is opt-in because storage is O(H*T^2) for each saved layer:

```bash
python scripts/wsi_extract_fm_outputs.py \
  --feature-manifest data/wsi/tcga1000_antibench/registry/artifacts.csv \
  --artifact-type tile_features \
  --feature-set-id conch_v15_final_d768 \
  --model titan \
  --feature-key final \
  --output-dir artifacts/wsi_attention/titan_real_full_sample \
  --titan-attention-mode global_to_tokens \
  --titan-attention-mode received \
  --titan-attention-mode rollout \
  --titan-attention-mode full \
  --titan-full-layer -1 \
  --titan-max-full-attention-tokens 2048 \
  --max-slides 10
```

This writes `attention/full_layer_XXX [H,T,T]` for the selected layer. Raise the
token limit only after estimating disk and host-memory requirements.

### Interpretation

The full matrices are defined over TITAN vision tokens. Depending on the live
checkpoint configuration, a token may correspond to one CONCH tile or to a
spatial block of the dense feature grid. Tile-level derivatives are emitted only
when the inferred mapping is validated against the captured token count. When
mapping cannot be validated, the real token matrix is still saved but tile-level
analysis is intentionally disabled for that slide.

Runtime monkey-patching is scoped to one model forward and is restored on exit.
Do not call the same adapter concurrently from multiple Python threads; use one
process per GPU/worker.
