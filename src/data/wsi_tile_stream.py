"""On-the-fly WSI tile sampling for task-agnostic tile-level EAF training.

The module deliberately stores only slide metadata and TRIDENT coordinates. Tile
pixels are read lazily from the WSI by DataLoader workers; embeddings and
attention targets never touch disk.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off", ""}


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    case_id: str
    cohort: str
    raw_path: Path
    coords_path: Path
    split: str
    coord_count: int
    patch_level: int
    patch_size: int
    coordinate_window_size: int


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{text}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _resolve_path(value: str, data_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def _first_coordinate_dataset(handle: h5py.File) -> h5py.Dataset:
    preferred = ("coords", "coordinates", "patches/coords")
    for key in preferred:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            ds = handle[key]
            if ds.ndim == 2 and ds.shape[1] >= 2:
                return ds

    candidates: list[h5py.Dataset] = []

    def visitor(_: str, obj: Any) -> None:
        if (
            isinstance(obj, h5py.Dataset)
            and obj.ndim == 2
            and obj.shape[1] >= 2
        ):
            candidates.append(obj)

    handle.visititems(visitor)
    if not candidates:
        raise ValueError(f"No [N, >=2] coordinate dataset in {handle.filename}")
    return candidates[0]


def _attr_int(attrs: dict[str, Any], names: Sequence[str], default: int) -> int:
    for name in names:
        if name not in attrs:
            continue
        value = np.asarray(attrs[name]).reshape(-1)
        if value.size:
            try:
                return int(round(float(value[0])))
            except (TypeError, ValueError):
                continue
    return default


def inspect_coordinate_file(
    path: Path,
    default_patch_size: int = 512,
    crop_size_at_target_mag: int | None = None,
) -> tuple[int, int, int, int]:
    """Return coordinate and crop metadata without loading all coordinates.

    Returns ``(n_coords, read_level, read_size, coordinate_window_size)``.
    ``read_size`` and ``coordinate_window_size`` are expressed in pixels at
    ``read_level``. Canonical TRIDENT coordinates use level 0.
    """
    with h5py.File(path, "r") as handle:
        ds = _first_coordinate_dataset(handle)
        attrs: dict[str, Any] = dict(handle.attrs)
        attrs.update(dict(ds.attrs))

        requested_patch_size = _attr_int(
            attrs,
            ("patch_size", "patch_size_px", "tile_size", "read_size"),
            default_patch_size,
        )
        patch_size_level0 = _attr_int(attrs, ("patch_size_level0",), -1)
        target_mag = _attr_int(attrs, ("target_magnification", "mag"), -1)
        level0_mag = _attr_int(attrs, ("level0_magnification",), -1)

        # TRIDENT coordinates are always level-0 x/y. Native 40x slides need a
        # larger level-0 crop than native 20x slides for the same 20x tile.
        if patch_size_level0 > 0:
            patch_level = 0
            coordinate_window_size = patch_size_level0
            scale = coordinate_window_size / max(requested_patch_size, 1)
        else:
            patch_level = _attr_int(
                attrs,
                ("patch_level", "level", "read_level", "wsi_level"),
                0,
            )
            if target_mag > 0 and level0_mag > 0 and patch_level == 0:
                scale = level0_mag / target_mag
                coordinate_window_size = int(round(requested_patch_size * scale))
            else:
                scale = 1.0
                coordinate_window_size = requested_patch_size

        read_size = coordinate_window_size
        if crop_size_at_target_mag is not None:
            if crop_size_at_target_mag <= 0:
                raise ValueError("crop_size_at_target_mag must be positive")
            if patch_level != 0:
                raise ValueError(
                    f"Encoder-specific crops require level-0 TRIDENT coordinates: {path}"
                )
            read_size = int(round(crop_size_at_target_mag * scale))
            if read_size > coordinate_window_size:
                raise ValueError(
                    f"Requested {crop_size_at_target_mag}px target-mag crop "
                    f"({read_size}px level 0) exceeds coordinate window "
                    f"({coordinate_window_size}px) in {path}"
                )

        return int(ds.shape[0]), patch_level, read_size, coordinate_window_size


def load_wsi_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    split_column: str = "split",
    val_fraction: float = 0.10,
    split_seed: int = 42,
    include_slide_groups: Sequence[str] = ("diagnostic",),
    exclude_cohorts: Sequence[str] = (),
    default_patch_size: int = 512,
    crop_size_at_target_mag: int | None = None,
    validate_files: bool = True,
) -> dict[str, list[SlideRecord]]:
    """Load canonical WSI metadata and create a case-disjoint in-memory split.

    Existing ``train``/``val``/``validation``/``test`` values in ``split_column``
    are respected. Rows without a split are assigned by a stable hash of
    ``case_id``; no split file is written.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    include_groups = {x.strip().lower() for x in include_slide_groups}
    excluded = {x.strip() for x in exclude_cohorts}

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    records: list[SlideRecord] = []
    case_splits: dict[str, str] = {}
    for row in rows:
        slide_group = (row.get("slide_group") or "diagnostic").strip().lower()
        if include_groups and slide_group not in include_groups:
            continue
        cohort = (row.get("cohort") or "unknown").strip()
        if cohort in excluded:
            continue
        if not _as_bool(row.get("include_in_pretraining"), default=True):
            continue
        if row.get("coords_available") is not None and not _as_bool(
            row.get("coords_available"), default=True
        ):
            continue
        if (row.get("preprocessing_status") or "").strip().lower() == "raw_only":
            continue

        slide_id = (row.get("slide_id") or Path(row.get("file_name", "")).stem).strip()
        case_id = (row.get("case_id") or slide_id).strip()
        raw_value = (row.get("raw_path") or row.get("wsi_path") or "").strip()
        coords_value = (row.get("coords_path") or "").strip()
        if not raw_value or not coords_value:
            continue

        raw_path = _resolve_path(raw_value, data_root)
        coords_path = _resolve_path(coords_value, data_root)
        if validate_files:
            if not raw_path.is_file():
                raise FileNotFoundError(f"Missing WSI for {slide_id}: {raw_path}")
            if not coords_path.is_file():
                raise FileNotFoundError(f"Missing coordinates for {slide_id}: {coords_path}")

        explicit = (row.get(split_column) or "").strip().lower()
        if explicit in {"validation", "valid"}:
            explicit = "val"
        if explicit == "holdout":
            explicit = "test"
        if explicit not in {"train", "val", "test"}:
            explicit = (
                "val"
                if _stable_unit_interval(case_id, split_seed) < val_fraction
                else "train"
            )
        previous = case_splits.setdefault(case_id, explicit)
        if previous != explicit:
            raise ValueError(
                f"Case {case_id!r} appears in multiple splits: {previous}, {explicit}"
            )

        (
            n_coords,
            patch_level,
            patch_size,
            coordinate_window_size,
        ) = inspect_coordinate_file(
            coords_path,
            default_patch_size=default_patch_size,
            crop_size_at_target_mag=crop_size_at_target_mag,
        )
        if n_coords <= 0:
            continue
        records.append(
            SlideRecord(
                slide_id=slide_id,
                case_id=case_id,
                cohort=cohort,
                raw_path=raw_path,
                coords_path=coords_path,
                split=explicit,
                coord_count=n_coords,
                patch_level=patch_level,
                patch_size=patch_size,
                coordinate_window_size=coordinate_window_size,
            )
        )

    split_records: dict[str, list[SlideRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        split_records[record.split].append(record)
    if not split_records["train"] or not split_records["val"]:
        raise ValueError(
            f"Need non-empty train and val splits; got "
            f"train={len(split_records['train'])}, val={len(split_records['val'])}"
        )
    return split_records


class WSITileDataset(Dataset):
    """Read scheduled WSI tiles lazily with worker-local LRU caches."""

    def __init__(
        self,
        records: Sequence[SlideRecord],
        transform: Any,
        *,
        augment: bool,
        slide_cache_size: int = 4,
        coordinate_cache_size: int = 8,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.augment = augment
        self.slide_cache_size = max(1, slide_cache_size)
        self.coordinate_cache_size = max(1, coordinate_cache_size)
        self._slide_cache: OrderedDict[str, Any] | None = None
        self._coord_cache: OrderedDict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return sum(record.coord_count for record in self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_slide_cache"] = None
        state["_coord_cache"] = None
        return state

    def _ensure_caches(self) -> None:
        if self._slide_cache is None:
            self._slide_cache = OrderedDict()
        if self._coord_cache is None:
            self._coord_cache = OrderedDict()

    def _get_slide(self, path: Path) -> Any:
        self._ensure_caches()
        assert self._slide_cache is not None
        key = os.fspath(path)
        if key in self._slide_cache:
            slide = self._slide_cache.pop(key)
            self._slide_cache[key] = slide
            return slide
        try:
            import openslide
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "openslide-python and the OpenSlide shared library are required"
            ) from exc
        slide = openslide.OpenSlide(key)
        self._slide_cache[key] = slide
        while len(self._slide_cache) > self.slide_cache_size:
            _, old = self._slide_cache.popitem(last=False)
            old.close()
        return slide

    def _get_coords(self, path: Path) -> np.ndarray:
        self._ensure_caches()
        assert self._coord_cache is not None
        key = os.fspath(path)
        if key in self._coord_cache:
            coords = self._coord_cache.pop(key)
            self._coord_cache[key] = coords
            return coords
        with h5py.File(path, "r") as handle:
            ds = _first_coordinate_dataset(handle)
            coords = np.asarray(ds[:, :2], dtype=np.int64)
        self._coord_cache[key] = coords
        while len(self._coord_cache) > self.coordinate_cache_size:
            self._coord_cache.popitem(last=False)
        return coords

    def __getitem__(self, index: tuple[int, int]) -> tuple[torch.Tensor, int]:
        slide_index, coord_index = index
        record = self.records[int(slide_index)]
        coords = self._get_coords(record.coords_path)
        x, y = (int(v) for v in coords[int(coord_index)])
        crop_margin = record.coordinate_window_size - record.patch_size
        if crop_margin > 0:
            if self.augment:
                x += random.randint(0, crop_margin)
                y += random.randint(0, crop_margin)
            else:
                offset = crop_margin // 2
                x += offset
                y += offset
        slide = self._get_slide(record.raw_path)
        tile = slide.read_region(
            (x, y),
            record.patch_level,
            (record.patch_size, record.patch_size),
        ).convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                tile = tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                tile = tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotations = random.randrange(4)
            if rotations:
                tile = tile.rotate(90 * rotations, expand=False)
        image = self.transform(tile) if self.transform is not None else tile
        return image, int(slide_index)


class WSIBalancedBatchSampler(Sampler[list[tuple[int, int]]]):
    """Coverage-aware cohort-balanced WSI/tile batch sampler.

    A batch contains tiles from a small number of WSIs, reducing OpenSlide seek
    pressure while retaining cross-WSI diversity. Each epoch rotates through
    cohort-specific slide permutations rather than drawing a tiny iid subset.
    """

    def __init__(
        self,
        records: Sequence[SlideRecord],
        *,
        batch_size: int = 32,
        slides_per_batch: int = 4,
        slides_per_epoch: int = 512,
        tiles_per_slide: int = 24,
        seed: int = 42,
        cohort_balance_power: float = 0.5,
        drop_last: bool = True,
    ) -> None:
        self.records = list(records)
        self.batch_size = batch_size
        self.slides_per_batch = slides_per_batch
        self.slides_per_epoch = min(max(1, slides_per_epoch), max(1, len(records)))
        self.tiles_per_slide = tiles_per_slide
        self.seed = seed
        self.cohort_balance_power = cohort_balance_power
        self.drop_last = drop_last
        self.epoch = 0
        self.last_summary: dict[str, Any] = {}

        if batch_size <= 0 or slides_per_batch <= 0:
            raise ValueError("batch_size and slides_per_batch must be positive")
        if batch_size % slides_per_batch != 0:
            raise ValueError("batch_size must be divisible by slides_per_batch")
        if tiles_per_slide <= 0:
            raise ValueError("tiles_per_slide must be positive")
        if not 0.0 <= cohort_balance_power <= 1.0:
            raise ValueError("cohort_balance_power must be in [0, 1]")

        self.tiles_per_slide_per_batch = batch_size // slides_per_batch
        self.rounds_per_group = math.ceil(
            tiles_per_slide / self.tiles_per_slide_per_batch
        )
        self.by_cohort: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.by_cohort[record.cohort].append(index)
        if not self.by_cohort:
            raise ValueError("No records supplied")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _cohort_targets(self) -> dict[str, int]:
        cohorts = sorted(self.by_cohort)
        weights = np.asarray(
            [len(self.by_cohort[c]) ** (1.0 - self.cohort_balance_power) for c in cohorts],
            dtype=np.float64,
        )
        exact = weights / weights.sum() * self.slides_per_epoch
        base = np.floor(exact).astype(int)
        remainder = self.slides_per_epoch - int(base.sum())
        order = np.argsort(-(exact - base))
        for idx in order[:remainder]:
            base[idx] += 1
        return {cohort: int(count) for cohort, count in zip(cohorts, base)}

    def _select_slides(self) -> list[int]:
        targets = self._cohort_targets()
        selected: list[int] = []
        for cohort, target in targets.items():
            indices = self.by_cohort[cohort]
            cohort_hash = int.from_bytes(
                hashlib.blake2b(cohort.encode(), digest_size=4).digest(), "big"
            )
            # Keep one stable cohort permutation and rotate through it. This
            # guarantees broad coverage across epochs instead of repeatedly
            # drawing a small iid subset.
            perm = np.asarray(indices, dtype=np.int64)
            np.random.default_rng(self.seed + cohort_hash).shuffle(perm)
            offset = (self.epoch * target) % len(perm)
            if target <= len(perm):
                chosen = np.roll(perm, -offset)[:target].tolist()
            else:
                rolled = np.roll(perm, -offset).tolist()
                repeats = math.ceil(target / len(rolled))
                chosen = (rolled * repeats)[:target]
            selected.extend(int(x) for x in chosen)
        np.random.default_rng(self.seed + self.epoch * 7919).shuffle(selected)
        return selected

    def _coord_sample(self, slide_index: int, rng: np.random.Generator) -> list[int]:
        n = self.records[slide_index].coord_count
        replace = n < self.tiles_per_slide
        return rng.choice(n, size=self.tiles_per_slide, replace=replace).astype(int).tolist()

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        selected = self._select_slides()
        if self.drop_last:
            usable = len(selected) - (len(selected) % self.slides_per_batch)
            selected = selected[:usable]
        rng = np.random.default_rng(self.seed + self.epoch * 104729)
        cohort_counts = Counter(self.records[i].cohort for i in selected)
        self.last_summary = {
            "epoch": self.epoch,
            "unique_slides": len(set(selected)),
            "scheduled_slides": len(selected),
            "scheduled_tiles": len(selected) * self.tiles_per_slide,
            "cohort_counts": dict(sorted(cohort_counts.items())),
        }

        for start in range(0, len(selected), self.slides_per_batch):
            group = selected[start : start + self.slides_per_batch]
            if len(group) < self.slides_per_batch and self.drop_last:
                break
            sampled = {idx: self._coord_sample(idx, rng) for idx in group}
            chunk = self.tiles_per_slide_per_batch
            for round_index in range(self.rounds_per_group):
                batch: list[tuple[int, int]] = []
                for slide_index in group:
                    coords = sampled[slide_index]
                    begin = round_index * chunk
                    values = coords[begin : begin + chunk]
                    if len(values) < chunk:
                        values = values + coords[: chunk - len(values)]
                    batch.extend((slide_index, coord_index) for coord_index in values)
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        groups = self.slides_per_epoch // self.slides_per_batch
        if not self.drop_last and self.slides_per_epoch % self.slides_per_batch:
            groups += 1
        return groups * self.rounds_per_group


def build_online_tile_loaders(
    split_records: dict[str, list[SlideRecord]],
    transform: Any,
    *,
    batch_size: int,
    slides_per_batch: int,
    train_slides_per_epoch: int,
    train_tiles_per_slide: int,
    val_slides_per_epoch: int,
    val_tiles_per_slide: int,
    num_workers: int,
    prefetch_factor: int,
    slide_cache_size: int,
    cohort_balance_power: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, WSIBalancedBatchSampler, WSIBalancedBatchSampler]:
    train_dataset = WSITileDataset(
        split_records["train"],
        transform,
        augment=True,
        slide_cache_size=slide_cache_size,
    )
    val_dataset = WSITileDataset(
        split_records["val"],
        transform,
        augment=False,
        slide_cache_size=slide_cache_size,
    )
    train_sampler = WSIBalancedBatchSampler(
        split_records["train"],
        batch_size=batch_size,
        slides_per_batch=slides_per_batch,
        slides_per_epoch=train_slides_per_epoch,
        tiles_per_slide=train_tiles_per_slide,
        seed=seed,
        cohort_balance_power=cohort_balance_power,
    )
    val_sampler = WSIBalancedBatchSampler(
        split_records["val"],
        batch_size=batch_size,
        slides_per_batch=slides_per_batch,
        slides_per_epoch=min(val_slides_per_epoch, len(split_records["val"])),
        tiles_per_slide=val_tiles_per_slide,
        seed=seed + 1,
        cohort_balance_power=0.0,
        drop_last=False,
    )
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **kwargs)
    return train_loader, val_loader, train_sampler, val_sampler
