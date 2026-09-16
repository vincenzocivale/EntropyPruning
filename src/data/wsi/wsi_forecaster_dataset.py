"""Dataset contract for WSI-EAF with explicit student-source / full-teacher separation.

The final paper pipeline uses three aligned caches per slide:

* ``tile_input_root``: tile embeddings given to the WSI student. For the final
  pipeline these come from the distilled Tile-EAF-pruned tile encoder.
* ``source_wsi_root``: TITAN run on ``tile_input_root`` with the requested
  intermediate hidden layer cached. WSI-EAF reads its source representation here.
* ``teacher_wsi_root``: frozen full TITAN run on the *full* tile encoder output.
  Final attention and slide embedding targets always come from this root.

Keeping these paths distinct prevents accidental pruned->pruned distillation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.wsi.attention import ManifestAttentionSource, align_attention_to_bag
from src.data.wsi.bag import WSIBag
from src.data.wsi.splits import assign_case_splits
from src.wsi_pipeline.numpy_store import array_names, read_array


@dataclass(frozen=True)
class WSIForecasterManifestConfig:
    tile_input_root: Path
    teacher_wsi_root: Path
    source_wsi_root: Path | None = None
    attention_key: str = "attention/global_to_tiles_mass_share"
    target_layer: int = -1
    cohorts: tuple[str, ...] | None = None
    exclude_cohorts: tuple[str, ...] | None = None
    hidden_layer: int | None = None


def _files_by_stem(directory: Path) -> dict[str, Path]:
    values = {path.stem: path for path in directory.glob("*.h5")}
    values.update({path.stem: path for path in directory.glob("*.npyd")})
    return values


def build_manifest(config: WSIForecasterManifestConfig) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    source_root = config.source_wsi_root or config.teacher_wsi_root
    exclude = set(config.exclude_cohorts or ())
    teacher_cohorts = sorted(
        path for path in config.teacher_wsi_root.iterdir()
        if path.is_dir()
        and (config.cohorts is None or path.name in config.cohorts)
        and path.name not in exclude
    )
    for teacher_dir in teacher_cohorts:
        cohort = teacher_dir.name
        tile_dir = config.tile_input_root / cohort
        source_dir = source_root / cohort
        if not tile_dir.is_dir() or not source_dir.is_dir():
            continue
        teacher = _files_by_stem(teacher_dir)
        source = _files_by_stem(source_dir)
        tiles = _files_by_stem(tile_dir)
        for stem in sorted(set(teacher) & set(source) & set(tiles)):
            rows.append(
                {
                    "slide_id": stem,
                    "case_id": stem,
                    "project": cohort,
                    "tile_path": str(tiles[stem]),
                    "source_wsi_path": str(source[stem]),
                    "teacher_wsi_path": str(teacher[stem]),
                }
            )
    if not rows:
        raise RuntimeError(
            "No slide is shared by tile_input_root, source_wsi_root and teacher_wsi_root"
        )
    table = pd.DataFrame(rows)
    table["case_id"] = (
        table["slide_id"]
        .str.extract(r"^(.*__case_[^_]+)__", expand=False)
        .fillna(table["slide_id"])
    )
    return table


def write_manifest_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def build_attention_manifest_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["slide_id", "attention_path", "coords_path"]
        )
        writer.writeheader()
        for row in table[["slide_id", "teacher_wsi_path"]].itertuples(index=False):
            writer.writerow(
                {
                    "slide_id": row.slide_id,
                    "attention_path": row.teacher_wsi_path,
                    "coords_path": row.teacher_wsi_path,
                }
            )


def assign_splits(
    table: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 17,
) -> pd.DataFrame:
    return assign_case_splits(
        table,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )


def _load_tile_bag(slide_id: str, path: Path) -> WSIBag:
    tile_features = torch.from_numpy(read_array(path, "tile_embeddings")).float()
    coords = (
        torch.from_numpy(read_array(path, "coords")).long()
        if "coords" in array_names(path)
        else None
    )
    return WSIBag(slide_id=slide_id, tile_features=tile_features, coords=coords)


def _load_titan_hidden_bag(slide_id: str, path: Path, layer: int) -> WSIBag:
    key = f"auxiliary/hidden_layer_{layer:03d}"
    names = array_names(path)
    if key not in names:
        available = sorted(name for name in names if name.startswith("auxiliary/"))
        raise KeyError(f"{key!r} not found in {path}; available={available}")
    tile_features = torch.from_numpy(read_array(path, key)).float()
    if "coords" not in names:
        raise KeyError(f"{path} has no coords; hidden-layer alignment is impossible")
    coords = torch.from_numpy(read_array(path, "coords")).long()
    return WSIBag(slide_id=slide_id, tile_features=tile_features, coords=coords)


class WSIForecasterDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        attention_manifest_path: Path,
        config: WSIForecasterManifestConfig,
        split: str,
    ) -> None:
        if "split" not in manifest.columns:
            raise ValueError("manifest is missing split; call assign_splits first")
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No slides assigned to split={split!r}")
        self.config = config
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

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        if self.config.hidden_layer is not None:
            bag = _load_titan_hidden_bag(
                row.slide_id, Path(row.source_wsi_path), self.config.hidden_layer
            )
        else:
            bag = _load_tile_bag(row.slide_id, Path(row.tile_path))
        attention = self.attention_source.read(row.slide_id, n_tiles=bag.n_tiles)
        bag, attention = align_attention_to_bag(bag, attention, mode="coords")
        target = attention.values.clamp_min(0.0)
        target = target / target.sum().clamp_min(1e-8)
        coords = bag.coords if bag.coords is not None else torch.zeros(
            (bag.n_tiles, 2), dtype=torch.long
        )
        return bag.tile_features, coords, target, row.slide_id
