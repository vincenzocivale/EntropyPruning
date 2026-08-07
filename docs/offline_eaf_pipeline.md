# Offline EAF teacher-cache pipeline

## Why offline

EAF should not repeatedly execute a frozen foundation model during every training epoch.
A frozen teacher is run once, its supervision is versioned, and all subsequent EAF runs
read the cache. This makes large HISTAI/GTEx training feasible and guarantees that two EAF
variants trained from the same cache see identical teacher targets.

## Tile EAF

For a configured tile encoder and `early_layer` (default 2), cache creation stores four
arrays per slide:

```text
coords            int32   [N, 2]
early_tokens      float16 [N, T, D]
final_attention   float16 [N, ...]
tile_embeddings   float16 [N, Dout]
```

`final_attention` is the canonical target already consumed by the Tile-EAF objective. A
model adapter is responsible for defining its reduction (for example CLS-to-patch,
averaged across heads). The policy is recorded in `TileCacheSpec.attention_reduction`.

The tile adapter implements `TileTeacherAdapter.extract(images, early_layer)` and returns
`TileTeacherOutput`. Existing CONCH/timm extraction code should be migrated behind that
adapter rather than duplicated in a new script.

Concrete implementation: `HookedViTTileTeacherAdapter` in `src/wsi_pipeline/model_adapters.py`
generalizes the hook recipe already used by
`src/collection/extract_features.py::collect_and_save_dataset` (timm-style `Attention`:
`qkv` → optional `q_norm`/`k_norm` → softmax → `proj`) into a reusable adapter. Block
discovery reuses `src.wsi_pipeline.tile_encoders.hooks.find_transformer_blocks`, so both
plain timm ViTs (`HookedViTTileTeacherAdapter.from_timm("hf-hub:...")`, e.g. UNI) and
CONCH v1.5's `visual.trunk` (`HookedViTTileTeacherAdapter.from_conch()`) are supported —
`from_timm` is validated in `tests/wsi_pipeline/test_model_adapters.py` against a real
`vit_tiny_patch16_224`; `from_conch` follows the same interface as the existing
`ConchV15MultiLayerEncoder` (`src/wsi_pipeline/tile_encoders/conch_v15.py`) but has not
been run against the real gated checkpoint in this environment — it raises immediately if
CONCH's attention module doesn't match the expected timm interface, rather than silently
capturing the wrong tensor.

Note this is a different extraction than `ConchV15MultiLayerEncoder`/`TimmViTMultiLayerEncoder`
(`src/wsi_pipeline/tile_encoders/`): those pool only the CLS token at an early block and
feed the *WSI*-EAF tile-embedding stage (`WSITeacherAdapter.extract`'s `tile_embeddings`
input); `HookedViTTileTeacherAdapter` captures the full early-block patch-token sequence
plus final CLS→patch attention needed by Tile-EAF itself.

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

Cache identity includes model names/revisions, early layer, magnification, patch size,
dtype and attention policy. Two caches with different teacher semantics must never share a
cache directory. `TileCacheSpec.cache_id` and `WSICacheSpec.cache_id` provide stable IDs.

## Large-corpus workflow

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
Tile-EAF / WSI-EAF training (teacher-free, cache-only)
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
