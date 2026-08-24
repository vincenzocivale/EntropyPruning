"""Dataset for training a WSI-EAF forecaster against cached real WSI-FM attention.

Reads, per slide, one of two input bags (`WSIForecasterManifestConfig.hidden_layer`
selects which):

- default (`hidden_layer=None`): the tile encoder's final embeddings
  (`tile_embeddings`, Tile-EAF cache, see `docs/offline_eaf_pipeline.md`) --
  context-free per-tile features, identical regardless of the slide they sit in.
- `hidden_layer=k`: TITAN's own residual-stream output at vision-encoder block
  `k` (`auxiliary/hidden_layer_{k:03d}` in the wsi_eaf output file, produced by
  `TitanAttentionCaptureConfig.hidden_layers` / `titan_attention.py`), already
  contextualized by TITAN's own ALiBi self-attention up to that block, and
  already mapped to input-tile order -- see `src/models/wsi/dense_forecaster.py`
  for the rationale and the matching `WSIDenseForecasterALiBi` architecture.

Either way, the training target is the WSI-FM's real cross-tile attention
(`wsi_eaf` output file, `attention/<key>`), aligned to the input bag by tile
coordinates. Both input and target are fully precomputed -- no tile-encoder or
WSI-FM forward pass runs during training.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.wsi.attention import ManifestAttentionSource, align_attention_to_bag
from src.data.wsi.bag import WSIBag
from src.wsi_pipeline.attention_signal import SignalDiscoveryConfig, assign_case_splits


@dataclass(frozen=True)
class WSIForecasterManifestConfig:
    """Where the two compact caches live and which cohorts/target to use.

    ``tile_eaf_root`` / ``wsi_eaf_root`` are the per-dataset directories, e.g.
    ``$EAF_WSI_ROOT/caches/tile_eaf/<dataset>/<tile_encoder>/<cache_id>`` and
    ``$EAF_WSI_ROOT/caches/wsi_eaf/<dataset>/<tile_encoder>__<wsi_encoder>``
    (see `docs/data_layout.md`). Both are expected to contain one
    subdirectory per cohort, each holding one HDF5 file per slide, with
    matching filenames across the two roots (the WSI-EAF writer records its
    source tile cache path in ``source_tile_cache`` for exactly this pairing).
    """

    tile_eaf_root: Path
    wsi_eaf_root: Path
    attention_key: str = "attention/global_to_tiles_mass_share"
    target_layer: int = -1
    cohorts: tuple[str, ...] | None = None
    # If set, the forecaster's input bag is TITAN's own intermediate hidden state at
    # this 0-based vision-encoder block index (``auxiliary/hidden_layer_{layer:03d}``
    # in the wsi_eaf output file, see `TitanAttentionCaptureConfig.hidden_layers` /
    # `wsi_eaf_infer_wsi_fm.py --titan-hidden-layer`), already mapped to input-tile
    # order -- instead of the tile encoder's context-free `tile_embeddings` from
    # `tile_eaf_root`. See docs/offline_eaf_pipeline.md and
    # src/models/wsi/dense_forecaster.py for why this exists.
    hidden_layer: int | None = None


def build_manifest(config: WSIForecasterManifestConfig) -> pd.DataFrame:
    """Scan both caches, intersect by filename per cohort, return one row per
    slide with columns: slide_id, case_id, project, tile_path, attention_path.
    """
    rows: list[dict] = []
    cohort_dirs = sorted(
        d for d in config.wsi_eaf_root.iterdir() if d.is_dir() and (config.cohorts is None or d.name in config.cohorts)
    )
    for cohort_dir in cohort_dirs:
        cohort = cohort_dir.name
        tile_cohort_dir = config.tile_eaf_root / cohort
        if not tile_cohort_dir.is_dir():
            continue
        attn_stems = {p.stem: p for p in cohort_dir.glob("*.h5")}
        tile_stems = {p.stem: p for p in tile_cohort_dir.glob("*.h5")}
        for stem in sorted(set(attn_stems) & set(tile_stems)):
            rows.append(
                {
                    "slide_id": stem,
                    "case_id": stem,  # refined below from the `case_NNNN` token if present
                    "project": cohort,
                    "tile_path": str(tile_stems[stem]),
                    "attention_path": str(attn_stems[stem]),
                }
            )
    if not rows:
        raise RuntimeError(
            f"No slide had both a tile_eaf and a wsi_eaf cache file under "
            f"{config.tile_eaf_root} / {config.wsi_eaf_root}."
        )
    table = pd.DataFrame(rows)
    # `{cohort}__case_XXXX__slide_...` -> case_id = `{cohort}__case_XXXX`, so
    # multiple slides of the same case are never split across train/val/test.
    table["case_id"] = table["slide_id"].str.extract(r"^(.*__case_[^_]+)__", expand=False).fillna(table["slide_id"])
    return table


def write_manifest_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def build_attention_manifest_csv(table: pd.DataFrame, path: Path) -> None:
    """Write the CSV `ManifestAttentionSource` reads.

    ``coords_path`` points at the *same* wsi_eaf HDF5 file as
    ``attention_path`` (it holds a top-level ``coords`` dataset alongside
    ``attention/...``), so `align_attention_to_bag(..., mode="coords")` can
    verify tile order against the tile_eaf bag's own coords instead of
    trusting that the two caches agree on ordering by construction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "attention_path", "coords_path"])
        writer.writeheader()
        for row in table[["slide_id", "attention_path"]].itertuples(index=False):
            writer.writerow(
                {"slide_id": row.slide_id, "attention_path": row.attention_path, "coords_path": row.attention_path}
            )


def assign_splits(
    table: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 17,
) -> pd.DataFrame:
    """Case-disjoint, per-project-stratified train/validation/test split.

    Reuses `attention_signal.assign_case_splits` so this manifest's splits are
    identical in method (and, given the same seed, identical in outcome) to
    the correlational analysis already run against this same corpus.
    """
    config = SignalDiscoveryConfig(train_fraction=train_fraction, validation_fraction=validation_fraction, seed=seed)
    return assign_case_splits(table, config).rename(columns={"analysis_split": "split"})


def _load_tile_eaf_bag(slide_id: str, path: Path) -> WSIBag:
    with h5py.File(path, "r") as handle:
        tile_features = torch.from_numpy(handle["tile_embeddings"][...]).to(torch.float32)
        coords = torch.from_numpy(handle["coords"][...]).to(torch.long) if "coords" in handle else None
    return WSIBag(slide_id=slide_id, tile_features=tile_features, coords=coords)


def _load_titan_hidden_bag(slide_id: str, path: Path, layer: int) -> WSIBag:
    """Load TITAN's own intermediate hidden state (already tile-aligned by the
    capture code, see `titan_attention.py::capture_titan_attention`) as the bag's
    input features, from the same wsi_eaf output file the attention target reads."""
    key = f"hidden_layer_{layer:03d}"
    with h5py.File(path, "r") as handle:
        if "auxiliary" not in handle or key not in handle["auxiliary"]:
            available = sorted(handle["auxiliary"].keys()) if "auxiliary" in handle else []
            raise KeyError(f"{key!r} not found in {path}'s auxiliary group; available={available}")
        tile_features = torch.from_numpy(handle["auxiliary"][key][...]).to(torch.float32)
        coords = torch.from_numpy(handle["coords"][...]).to(torch.long) if "coords" in handle else None
    if coords is None:
        raise KeyError(f"{path} has no top-level coords dataset; required to align hidden-layer bags")
    return WSIBag(slide_id=slide_id, tile_features=tile_features, coords=coords)


class WSIForecasterDataset(Dataset):
    """One item = one slide's full tile bag + its aligned, normalized attention target.

    Bag size N varies per slide (tens to tens of thousands of tiles); batching
    is the training script's responsibility (variable-N bags do not collate
    naturally -- see `scripts/train_wsi_landmark_forecaster.py`, which uses
    batch_size=1 with gradient accumulation across slides).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        attention_manifest_path: Path,
        config: WSIForecasterManifestConfig,
        split: str,
    ) -> None:
        if "split" not in manifest.columns:
            raise ValueError("manifest is missing a 'split' column; call assign_splits() first.")
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No slides assigned to split={split!r}.")
        self.config = config
        # `attention_manifest_path` is shared read-only across every split's dataset
        # (train/val/test all reference the same slide_id -> attention_path CSV; the
        # `split` filter above already restricts which slide_ids this instance uses).
        self.attention_source = ManifestAttentionSource(
            attention_manifest_path,
            attention_key=config.attention_key,
            coords_key="coords",
            tile_axis=2,
            reduction="mean",
            selections={0: config.target_layer},
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.rows.iloc[index]
        if self.config.hidden_layer is not None:
            # Both the bag and the attention target come from the same wsi_eaf file
            # (`row.attention_path`), so this is TITAN's own intermediate
            # representation, not the tile encoder's output -- see
            # WSIForecasterManifestConfig.hidden_layer.
            bag = _load_titan_hidden_bag(row.slide_id, Path(row.attention_path), self.config.hidden_layer)
        else:
            bag = _load_tile_eaf_bag(row.slide_id, Path(row.tile_path))
        attention = self.attention_source.read(row.slide_id, n_tiles=bag.n_tiles)
        bag, attention = align_attention_to_bag(bag, attention, mode="coords")
        target = attention.values.clamp_min(0.0)
        target = target / target.sum().clamp_min(1e-8)
        coords = bag.coords if bag.coords is not None else torch.zeros((bag.n_tiles, 2), dtype=torch.long)
        return bag.tile_features, coords, target, row.slide_id
