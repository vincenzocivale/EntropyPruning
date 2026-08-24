# Offline EAF teacher-cache pipeline

## Why offline

EAF should not repeatedly execute a frozen foundation model during every training epoch.
A frozen teacher is run once, its supervision is versioned, and all subsequent EAF runs
read the cache. This makes large HISTAI/GTEx training feasible and guarantees that two EAF
variants trained from the same cache see identical teacher targets.

## Tile EAF

**Compact permanent cache (CACHE_SCHEMA_VERSION v2, 2026-08-07).** Cache creation stores
three arrays per slide, one HDF5 file per WSI:

```text
coords            int32   [N, 2]           # level-0 (x, y) of each tile's top-left corner
final_attention   float16 [N, T]           # CLS-to-patch, head-averaged, L1-renormalized
tile_embeddings   float16 [N, Dout]        # Dout=768 for CONCH v1.5 (pooled, not per-token)
```

`early_tokens` `float16 [N, T, D]` (T=784, D=1024 for CONCH v1.5) is **deliberately not
persisted**. It was measured at >99.8% of on-disk bytes (~1.53 MiB/tile vs. ~3.27
KiB/tile for everything else combined — a real 2-slide HISTAI-hematologic smoke test
cache dropped from 265/352 MiB to 550/729 KiB per slide once removed), which at
full-corpus scale is a many-TB-to-low-PB difference for a component EAF Tile training
can instead recompute for a few hundred milliseconds per batch. Storing it was the v1
design (see below); **EAF Tile training now recomputes it ONLINE** via
`HookedViTTileTeacherAdapter.extract_early(images, early_layer)` — a cheap early-exit
partial forward through blocks `0..early_layer` only (measured ~8.7x faster than
`extract_final`'s full 24-block forward on CPU; a bigger margin is expected on GPU),
raised as an exception from inside the target block's own `forward_hook` so no later
block, the final norm, or the pooling head ever executes. See
`src/wsi_pipeline/compact_cache_dataset.py`'s `CompactTileTargetDataset`, which pairs
the cached `final_attention` (target) with tile pixels re-read from the source WSI
(image), and lets the training loop call `extract_early` on the collated batch.

**Consequence for the cold-archive/raw-release policy** (see "Cold archive" below): EAF
Tile training is *not* pixel-free anymore. It needs raw WSI (or, in the future, the cold
archive) access at every training step for `extract_early`, even though it never runs
the full frozen teacher. Releasing raw for a corpus EAF Tile is actively training
against — without first pointing training at an alternative pixel source — would break
it; this is a real, load-bearing dependency introduced by moving `early_tokens` out of
the permanent cache, not a slip to overlook.

**Encoder support (2026-08-24).** `--encoder conch_v15` uses TITAN's `return_conch()`
accessor; any other name (`uni2h`, `virchow2`, `hoptimus1`, `provgigapath`, ...) is
loaded through THUNDER's model registry
(`HookedViTTileTeacherAdapter.from_thunder_model`), the same path
`scripts/train_wsi_tile_eaf_online.py`/`finetune_wsi_tile_encoder_pruned_online.py`
already use for `--model-name`. The cache never depends on `early_layer`/source layer
(`cache_id` excludes it, see `cache_contracts.py`), so one cache per encoder covers
every `--source-layer` sweep without rebuilding — see
`docs/tile_eaf_experiment_roadmap.md` for the current experiment plan and which
corpus/datasets back each stage.

`HookedViTTileTeacherAdapter.extract_final(images)` is the offline, cache-building
counterpart: one full forward, only `final_attention` + `tile_embeddings`, no
`early_tokens` hook installed at all (so building the cache doesn't even pay the small
extra cost of capturing/transferring an array it will discard). The combined
`extract(images, early_layer)` (both quantities from one full forward) still exists
purely for numeric-equivalence testing and ad-hoc inspection — no production code path
uses it.

**`early_layer` semantics (pinned, not ambiguous):** 0-based transformer block index;
the value is that block's **output** — after both its attention and MLP residual
branches, i.e. the hidden state as handed to `block[early_layer + 1]` — with the
CLS/register prefix stripped. This is captured via a plain `register_forward_hook` on
the block itself. It is deliberately **not** the block's attention-submodule *input*
(`norm1(block_input)`, which is effectively the *previous* block's output after
normalization) — an earlier version of this adapter captured that instead for the same
`early_layer` value, silently disagreeing with the online teacher below. Every cache's
`TileCacheSpec.early_layer_semantics` field carries this exact prose so a reader never
has to guess what a given cache's `early_layer=2` means.

**`final_attention` semantics (pinned):** CLS-to-patch self-attention (post-softmax) at
the model's **last** transformer block, averaged over heads, then L1-renormalized over
the patch axis so each row sums to 1.0 (`TileCacheSpec.attention_reduction ==
"cls_mean_heads_l1norm"`). Renormalization matters: softmax attention including the
CLS/register prefix does not sum to 1 once those prefix columns are dropped from
storage, so skipping it leaves a teacher distribution the online trainer was never
actually fit against.

Both of the above are pinned to match `src.models.online_tile_eaf.OnlineAttentionTeacher`
— the teacher actually driving the live online Tile-EAF trainer
(`scripts/train_wsi_tile_eaf_online.py`) — bit for bit, verified numerically (fp16
tolerance) against the real, gated CONCH v1.5 checkpoint (`MahmoodLab/TITAN`'s
`return_conch()`), not merely "a" reasonable definition. See
`src/wsi_pipeline/model_adapters.py::HookedViTTileTeacherAdapter` for the implementation
and its extended docstring for the full derivation.

The tile adapter implements `TileTeacherAdapter.extract(images, early_layer)` and returns
`TileTeacherOutput`; `HookedViTTileTeacherAdapter` is the concrete implementation, and
`python scripts/eaf.py cache tile` is the canonical, sole CLI entry point that drives it
end to end (dataset adapter → TRIDENT coords → `HookedViTTileTeacherAdapter` → atomic
per-slide HDF5 via `TileCacheWriter`). The older `scripts/wsi_extract_tile_embeddings.py`
+ `ConchV15MultiLayerEncoder`/`TimmViTMultiLayerEncoder` (which extracted layer-2 + final
embeddings only, with no attention target) has been removed — it also never actually ran
against the real CONCH v1.5 checkpoint (its `conch.encode_image(...)` call does not exist
on that model; the real API is `conch(images)`) and used the wrong input resolution (512
instead of CONCH v1.5's actual 448, from `titan.return_conch()`'s own transform). Block
discovery reuses `src.wsi_pipeline.tile_encoders.hooks.find_transformer_blocks`, so both
plain timm ViTs (`HookedViTTileTeacherAdapter.from_timm(...)`, e.g. UNI) and CONCH v1.5
(`HookedViTTileTeacherAdapter.from_conch()`, resolving `conch.trunk.blocks`, 24 blocks,
`num_prefix_tokens=1`) are supported as long as the attention module exposes the standard
`qkv`/`scale`/`attn_drop`/`proj`/`proj_drop` interface (optionally `q_norm`/`k_norm`).
Prefix-token count is auto-resolved (`resolve_num_prefix_tokens`), not hardcoded to 1.

`tile_embeddings` is the encoder's own final pooled output exactly as the model produces
it (`conch(images)` for CONCH v1.5 — its 768-d attentional-pooler + LayerNorm output, not
a raw CLS token) and is shared, unmodified, with the WSI-EAF cache stage below — it is
never treated as Tile-EAF-exclusive.

### Resume, integrity and atomicity

`TileCacheWriter` writes to a sibling `.tmp` file; the real path only appears via an
atomic `os.replace` when the writer closes *without* an exception, and only after the
HDF5 `complete` attribute is set `True`. A crash, OOM, or Ctrl-C mid-slide therefore never
leaves a corrupt/partial file at the real cache path. `tile_cache_status(path,
coords_path=..., spec=...)` (in `src/wsi_pipeline/cache_io.py`) is the resume check every
caller (the CLI, the HISTAI orchestrator) uses before (re)building a slide: it rebuilds
from scratch — never silently trusts — whenever the cache is missing, not `complete`,
corrupt/unreadable, was built under a different `TileCacheSpec.cache_id` (encoder/layer/
attention/dtype changed), or its tile count disagrees with the TRIDENT coords file's row
count. `validate_cache`/`n_coords_in_registry` retry briefly on a known transient HDF5
"unable to lock file" race (observed right after a DataLoader with worker processes tears
down) rather than treating a momentary lock contention as corruption.

### Storage cost (measured, not estimated)

**Compact cache (current, v2):** per tile, CONCH v1.5, fp16, lzf: `final_attention` 1.53
KiB, `tile_embeddings` 1.5 KiB, `coords` 8 B — **~3.27 KiB/tile total.** A 172-tile and a
229-tile HISTAI-hematologic smoke-test slide measured 3,271 and 3,261 bytes/tile on disk.
Projected: **~3.3 GB per 1M tiles, ~33 GB per 10M tiles** — full HISTAI+GTEx+HEST at
tile-level is now a routine amount of storage, not a capacity-planning decision.

**v1 (superseded), for context on why the array was dropped:** with `early_tokens`
[N,784,1024] included, per-tile cost was ~1.53 MiB (99.8% of it `early_tokens`; lzf
achieves ~0% reduction on dense float activations), projecting to ~1.53 TB/1M tiles and
~15.3 TB/10M tiles — a real many-TB-to-low-PB commitment at full-corpus scale that this
redesign (online `extract_early` recomputation instead of caching, ~470-500x smaller on
disk) removes.

## WSI EAF

A WSI teacher consumes frozen tile embeddings and coordinates. The default cache is:

```text
coords            int32   [N, 2]
tile_embeddings   float16 [N, D]
tile_scores        float16 [N]
wsi_embedding      float16 [Dw]
```

The canonical `tile_scores` vector is deliberately model-agnostic. TITAN can derive it
from attention; an ABMIL teacher may expose its attention directly; another WSI FM can
implement a documented importance projection. Full attention tensors are optional and are
not retained in the main training corpus unless `store_raw_attention=True`.

A WSI adapter implements `WSITeacherAdapter.extract(tile_embeddings, coords)` and returns
`WSITeacherOutput`.

Concrete implementations in `src/wsi_pipeline/model_adapters.py`, each wrapping the
existing `src/wsi_pipeline/wsi_models/` adapter rather than reimplementing it:

- `TitanWSITeacherAdapter` — `tile_scores` defaults to `global_to_tiles_mass_share`
  (TITAN's global/CLS attention to its own vision tokens, mass-normalized and mapped
  down to one value per input tile; see `titan_attention.py`), with a 1:1 token/tile
  fallback to `global_to_tokens` when tile-to-token mapping is unavailable.
- `FeatherWSITeacherAdapter` — FEATHER's gated-ABMIL instance-attention probabilities
  are already one canonical value per tile; used directly.
- `GigaPathWSITeacherAdapter` — Prov-GigaPath exposes no native attention; `tile_scores`
  falls back to a documented, deterministic cosine similarity between each tile
  embedding and the resulting slide embedding (an explicit importance projection, never
  presented as native attention).

## Cache identity

Cache identity includes model name/revision, magnification, patch size, dtype and
attention policy. Two caches with different teacher semantics must never share a cache
directory. `TileCacheSpec.cache_id` and `WSICacheSpec.cache_id` provide stable IDs.
`TileCacheSpec.early_layer` is the one field deliberately **excluded** from
`cache_id`: since v2 it no longer determines any stored byte (see "Tile EAF" above) — it
is purely a training-time parameter for `extract_early`, recorded in every cache's
metadata as documentation, but changing it must never force an unrelated, expensive
full-forward rebuild of `final_attention`/`tile_embeddings`.

## Large-corpus workflow

### Profile and tune tile-cache extraction

Use real WSI reads for tuning; the synthetic ``--batch-size auto`` probe only checks
VRAM fit. The tuner publishes no cache during its benchmark and records phase timings
for the subsequent run:

```bash
python scripts/eaf.py cache tile \
  --data-root "$EAF_WSI_ROOT" \
  --manifest "$SOURCE_DATASET/manifests/slides.csv" \
  --output-dir "$EAF_WSI_ROOT/caches/tile_eaf/<dataset>/conch_v15/<cache-id>" \
  --encoder conch_v15 --dataset <dataset> \
  --autotune --batch-size-candidates 32 64 96 \
  --worker-candidates 4 8 16 --prefetch-candidates 2 4 \
  --openslide-cache-mib 512 \
  --slide-loader-chunk-size 32 --persistent-workers \
  --profile-json "$EAF_WSI_ROOT/logs/tile_cache_profile.json" \
  --compression none
```

Do not run TRIDENT segmentation concurrently on the same GPU while tuning. Existing
valid per-slide files are skipped unless ``--overwrite`` is supplied.

Cache production uses one DataLoader worker pool for each group of 32 missing WSIs.
Batches remain slide-pure and HDF5 files are still finalized atomically one WSI at a
time, while workers can prefetch the next slide before the current one finishes. A
shared-loader failure falls back to the isolated per-slide path for every unfinished
WSI. Set ``--slide-loader-chunk-size 1`` or ``--no-persistent-workers`` to restore the
legacy loader lifecycle for diagnosis.

``--openslide-cache-mib`` installs a decoded-tile cache on each worker's independent
OpenSlide handle. This matters for WSI whose internal TIFF tiles are much larger than
the requested model patch: a 512-pixel patch read from a 4096-pixel JPEG tile can
otherwise trigger repeated decompression. Capacity is per worker, so account for the
aggregate host-memory budget (for example, 4 workers x 512 MiB = 2 GiB).

On HISTAI generic pyramidal TIFFs (JPEG-compressed 4096 px internal tiles, 512 px model
patches), an isolated A100 benchmark over the same 4,861 patches measured:

| batch | workers | cache/worker | tiles/s | data wait | peak CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 512 MiB | 137.8 | 9.7 s | 3.38 GiB |
| **128** | **4** | **512 MiB** | **159.6** | **5.4 s** | **5.06 GiB** |
| 256 | 4 | 512 MiB | 139.9 | 8.3 s | 8.42 GiB |
| 512 | 8 | 256 MiB | 72.0 | 24.0 s | 15.12 GiB |
| 1024 | 8 | 256 MiB | 51.0 | 40.3 s | 28.54 GiB |

The largest fitting batch is not the fastest. A DataLoader worker constructs one whole
batch serially; very large batches therefore increase refill latency even when more
workers prefetch later batches. The HISTAI orchestrator defaults to batch 128, 4
workers, prefetch 4, and 512 MiB per worker. Re-profile on different WSI formats,
storage, models, or GPUs instead of treating these values as universal.

### Train Tile-EAF from the compact cache

First create one validated index over the source cache roots. This verifies cache
identity, completeness, tile counts, and exact coordinate order without copying HDF5
files:

```bash
python scripts/eaf.py cache index-tile \
  --data-root "$EAF_WSI_ROOT" \
  --slides "$EAF_WSI_ROOT/datasets/pretraining/eaf_wsi_pretrain_strict_v1/manifests/slides.csv" \
  --cache-root "$EAF_WSI_ROOT/caches/tile_eaf/histai_eaf_wsi_v1" \
  --cache-root "$EAF_WSI_ROOT/caches/tile_eaf/hest_eaf_wsi_v1" \
  --cache-root "$EAF_WSI_ROOT/caches/tile_eaf/gtex_eaf_wsi_v1" \
  --output "$EAF_WSI_ROOT/datasets/pretraining/eaf_wsi_pretrain_strict_v1/manifests/tile_cache_index.csv"

python scripts/train_wsi_tile_eaf_online.py \
  --model-name titan \
  --manifest "$EAF_WSI_ROOT/datasets/pretraining/eaf_wsi_pretrain_strict_v1/manifests/slides.csv" \
  --data-root "$EAF_WSI_ROOT" \
  --target-cache-index "$EAF_WSI_ROOT/datasets/pretraining/eaf_wsi_pretrain_strict_v1/manifests/tile_cache_index.csv" \
  --source-layer 2 --batch-size 32 --num-workers 8
```

Cached-attention training deliberately disables random spatial augmentation: flipping,
rotating, or jittering pixels without applying the identical permutation to the patch
attention grid would corrupt the target. The loader runs only through the requested
early block and never executes the cached teacher's final blocks.

```text
HISTAI / GTEx raw WSI
        |
        v
TRIDENT segmentation + canonical coords (20x, 512, stride 512)
        |
        +-----------------------+
        |                       |
        v                       v
Tile teacher cache        cold tissue-pixel archive
        |
        v
WSI teacher cache
        |
        v
WSI-EAF training (teacher-free, cache-only)
Tile-EAF training  (teacher-free, but NOT pixel-free: reads final_attention from
                     the cache + tile pixels from raw WSI/archive, recomputes
                     early_tokens online via a cheap partial forward each step)
```

For GTEx, one WSI is one DICOM `SeriesInstanceUID`. Store a representative `.dcm` in the
manifest but keep every DICOM instance in the series directory. Do not flatten a series
into a single symlink before OpenSlide processing.

## Cold archive

The cold archive is not the active EAF training representation. It is insurance against a
future tile encoder. After segmentation, only canonical tissue RGB patches and their
coordinates are retained, initially as JPEG quality 95 inside tar containers. The source
WSI can be released only after:

1. coordinate count is non-zero and expected,
2. required Tile/WSI caches pass validation,
3. archive SHA-256 and patch-count verification pass,
4. at least a small embedding-agreement audit has validated the chosen lossy codec,
5. provenance identifiers required to redownload the original are recorded.

Keep raw HISTAI more conservatively because access is gated. GTEx is public through IDC
and is safer to treat as remotely recoverable after archive validation.

### Lifecycle tracking

`src/wsi_pipeline/archive.py` models the full lifecycle as an explicit state machine:

```text
RAW -> SEGMENTED -> TEACHER_CACHES -> PIXEL_ARCHIVED -> VERIFIED -> RAW_RELEASABLE
```

`classify_stage(SlideLifecycleEvidence(...))` re-verifies each artifact (coordinate
count, cache validity via `validate_cache`, archive checksum/patch count via
`verify_pixel_archive`) rather than trusting that a file's mere presence means it is
valid — a corrupt cache or archive never advances the reported stage.
`release_preconditions(...)` evaluates the five preconditions above (points 1-2 collapse
into "caches validate" once coordinates are folded into the cache itself) and returns
`releasable: bool`, with no side effects. `python scripts/eaf.py archive lifecycle
--slide-id ... [--raw-path] [--coords-path] [--tile-cache ...] [--wsi-cache ...]
[--archive-path] [--remote-recoverable]` exposes this read-only classification.

**There is intentionally no `release-raw` command yet.** Per the cold-archive policy,
raw deletion is a future, explicit, human-triggered step that must only be added once an
operator actually wants to reclaim space for a specific corpus, and even then it must
require `release_preconditions(...)["releasable"] is True` (plus, for HISTAI, extra
manual sign-off given gated access) before touching anything. `embedding_agreement_audit`
supplies the pass/fail policy such a command would depend on; callers compute the two
embedding sets externally (e.g. by running a `TileTeacherAdapter` on original tissue
crops and on the archive's decoded JPEGs) and pass them in for comparison.
