"""Cold storage for tissue pixels after segmentation.

The archive is intentionally model-agnostic. It stores compressed canonical RGB patches
plus coordinates/provenance, allowing a future tile encoder to be added without keeping
the original multi-resolution WSI online.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PixelArchiveSpec:
    mag: int = 20
    patch_size: int = 512
    codec: str = "jpeg"
    jpeg_quality: int = 95
    schema_version: int = 1

    def validate(self) -> None:
        if self.codec != "jpeg":
            raise ValueError("Initial archive implementation supports codec='jpeg' only")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_pixel_archive(
    output_path: str | Path,
    *,
    slide_id: str,
    case_id: str,
    patches: Iterable[tuple[int, int, object]],
    spec: PixelArchiveSpec = PixelArchiveSpec(),
    source_metadata: dict | None = None,
) -> Path:
    """Write one slide archive from ``(x, y, PIL.Image)`` tissue patches."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pixel archiving requires Pillow") from exc

    spec.validate()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    coords: list[list[int]] = []
    with tarfile.open(output_path, mode="w") as archive:
        for count, (x, y, image) in enumerate(patches, start=1):
            if not isinstance(image, Image.Image):
                raise TypeError("patch image must be a PIL.Image")
            buffer = io.BytesIO()
            image.convert("RGB").save(
                buffer,
                format="JPEG",
                quality=spec.jpeg_quality,
                optimize=True,
            )
            payload = buffer.getvalue()
            name = f"patches/{count - 1:07d}__x{x}__y{y}.jpg"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            coords.append([int(x), int(y)])

        metadata = {
            "schema_version": spec.schema_version,
            "slide_id": slide_id,
            "case_id": case_id,
            "n_patches": count,
            "archive_spec": asdict(spec),
            "coords": coords,
            "source": source_metadata or {},
        }
        payload = json.dumps(metadata, sort_keys=True).encode()
        info = tarfile.TarInfo("metadata.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    (output_path.with_suffix(output_path.suffix + ".sha256")).write_text(
        _sha256(output_path) + "\n"
    )
    return output_path


def verify_pixel_archive(path: str | Path) -> dict:
    path = Path(path)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not checksum_path.exists():
        raise ValueError(f"Missing archive checksum: {checksum_path}")
    expected = checksum_path.read_text().strip()
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Archive checksum mismatch: {path}")
    with tarfile.open(path, mode="r") as archive:
        try:
            metadata_member = archive.getmember("metadata.json")
        except KeyError as exc:
            raise ValueError("Archive is missing metadata.json") from exc
        handle = archive.extractfile(metadata_member)
        assert handle is not None
        metadata = json.loads(handle.read())
        n_jpeg = sum(member.name.endswith(".jpg") for member in archive.getmembers())
    if n_jpeg != int(metadata["n_patches"]):
        raise ValueError(
            f"Patch count mismatch: metadata={metadata['n_patches']} archive={n_jpeg}"
        )
    return metadata | {"sha256": actual, "size_bytes": path.stat().st_size}


# ---------------------------------------------------------------------------
# Preprocessing lifecycle
#
#   RAW -> SEGMENTED -> TEACHER_CACHES -> PIXEL_ARCHIVED -> VERIFIED -> RAW_RELEASABLE
#
# This module never deletes raw data. It only classifies where a slide sits in
# the lifecycle and reports whether the (future, explicit, not-yet-implemented)
# raw-release step's preconditions are currently satisfied. No command in this
# repository performs the actual raw deletion.
# ---------------------------------------------------------------------------


class LifecycleStage(str, Enum):
    RAW = "RAW"
    SEGMENTED = "SEGMENTED"
    TEACHER_CACHES = "TEACHER_CACHES"
    PIXEL_ARCHIVED = "PIXEL_ARCHIVED"
    VERIFIED = "VERIFIED"
    RAW_RELEASABLE = "RAW_RELEASABLE"


@dataclass(frozen=True)
class SlideLifecycleEvidence:
    """Filesystem/metadata evidence used to classify one slide's lifecycle stage.

    Every field is optional evidence, not a claim; ``classify_stage`` and
    ``release_preconditions`` re-verify each artifact (coordinate count, cache
    validity, archive checksum/patch count) rather than trusting flags, so a
    stale or corrupt artifact never silently advances the stage.
    """

    slide_id: str
    raw_path: Path | None = None
    coords_path: Path | None = None
    tile_cache_paths: tuple[Path, ...] = ()
    wsi_cache_paths: tuple[Path, ...] = ()
    pixel_archive_path: Path | None = None
    # Provenance required to redownload the source if raw is ever released
    # (e.g. HISTAI repo_id/repo_path, GTEx series_uid, TCGA/HEST source path).
    provenance: Mapping[str, Any] | None = None
    # Recorded result of an embedding-agreement audit for the archive's codec
    # (see `embedding_agreement_audit`); absent means "not yet audited".
    embedding_agreement: Mapping[str, Any] | None = None
    # Source-recoverability policy: HISTAI is gated (conservative), GTEx is
    # public through IDC (safer to treat as remotely recoverable once verified).
    remote_recoverable: bool = False


def _coords_available(coords_path: Path | None) -> bool:
    if coords_path is None or not coords_path.exists():
        return False
    try:
        import h5py
    except ImportError:
        # Presence without h5py to inspect it is still "segmented" evidence.
        return coords_path.stat().st_size > 0
    try:
        with h5py.File(coords_path, "r") as handle:
            for name in ("coords", "coordinates", "patches/coords"):
                if name in handle and handle[name].shape[0] > 0:
                    return True
            for value in handle.values():
                if hasattr(value, "shape") and len(value.shape) >= 2 and value.shape[0] > 0:
                    return True
    except OSError:
        return False
    return False


def _caches_valid(paths: Sequence[Path]) -> bool:
    if not paths:
        return False
    from .cache_io import validate_cache

    for path in paths:
        if not path.exists():
            return False
        try:
            validate_cache(path)
        except Exception:
            return False
    return True


def _archive_verified(path: Path | None) -> tuple[bool, dict | None]:
    if path is None or not path.exists():
        return False, None
    try:
        return True, verify_pixel_archive(path)
    except Exception:
        return False, None


def classify_stage(evidence: SlideLifecycleEvidence, *, min_cosine: float = 0.99) -> LifecycleStage:
    """Classify a slide's current position in the preprocessing lifecycle.

    Each step re-verifies its own artifact instead of trusting that an artifact's
    mere presence means it is valid, so a corrupt cache or archive never advances
    the reported stage past the last artifact that actually checks out.
    """

    if evidence.raw_path is None or not evidence.raw_path.exists():
        # No raw evidence supplied/found here does not imply raw was deleted;
        # this classifier only reasons about the evidence it was given.
        stage = LifecycleStage.RAW
    else:
        stage = LifecycleStage.RAW

    if not _coords_available(evidence.coords_path):
        return stage
    stage = LifecycleStage.SEGMENTED

    if not _caches_valid(evidence.tile_cache_paths) and not _caches_valid(evidence.wsi_cache_paths):
        return stage
    stage = LifecycleStage.TEACHER_CACHES

    archived, archive_metadata = _archive_verified(evidence.pixel_archive_path)
    if not archived:
        if evidence.pixel_archive_path is not None and evidence.pixel_archive_path.exists():
            # File exists but failed checksum/patch-count verification: still
            # "archived", just not yet trustworthy enough to call VERIFIED.
            return LifecycleStage.PIXEL_ARCHIVED
        return stage
    stage = LifecycleStage.PIXEL_ARCHIVED

    agreement_ok = bool(
        evidence.embedding_agreement
        and evidence.embedding_agreement.get("passed") is True
        and float(evidence.embedding_agreement.get("min_cosine", 0.0)) >= min_cosine
    )
    if not agreement_ok:
        return stage
    stage = LifecycleStage.VERIFIED

    if release_preconditions(evidence, min_cosine=min_cosine)["releasable"]:
        stage = LifecycleStage.RAW_RELEASABLE
    return stage


def embedding_agreement_audit(
    original_embeddings: Any,
    archived_embeddings: Any,
    *,
    min_cosine: float = 0.99,
    codec: str = "jpeg",
    quality: int = 95,
) -> dict:
    """Compare embeddings computed from original vs. archived (re-decoded) tiles.

    Callers compute both embedding sets externally (e.g. by running the same
    ``TileTeacherAdapter`` on original tissue crops and on the archive's decoded
    JPEGs) and pass them in here; this function only owns the pass/fail policy so
    every audit in the repo applies the same threshold consistently.
    """

    import numpy as np

    original = np.asarray(original_embeddings, dtype="float32")
    archived = np.asarray(archived_embeddings, dtype="float32")
    if original.shape != archived.shape:
        raise ValueError(f"Shape mismatch: original={original.shape} archived={archived.shape}")
    num = (original * archived).sum(axis=-1)
    denom = np.linalg.norm(original, axis=-1) * np.linalg.norm(archived, axis=-1)
    cosine = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    result = {
        "codec": codec,
        "quality": quality,
        "n_sampled": int(original.shape[0]),
        "min_cosine": float(cosine.min()) if cosine.size else 0.0,
        "mean_cosine": float(cosine.mean()) if cosine.size else 0.0,
        "threshold": min_cosine,
    }
    result["passed"] = bool(cosine.size) and result["min_cosine"] >= min_cosine
    return result


def release_preconditions(
    evidence: SlideLifecycleEvidence, *, min_cosine: float = 0.99
) -> dict[str, Any]:
    """Evaluate (never enforce/execute) whether raw could be safely released.

    This intentionally has no side effects and deletes nothing. It exists so a
    future, explicit ``release-raw`` command has one canonical place to ask
    "is it safe yet?" — implementing that command is out of scope until an
    operator explicitly wants raw deletion, per the cold-archive policy in
    docs/offline_eaf_pipeline.md.
    """

    checks: dict[str, Any] = {}

    checks["caches_validate"] = _caches_valid(evidence.tile_cache_paths) or _caches_valid(
        evidence.wsi_cache_paths
    )

    archived, archive_metadata = _archive_verified(evidence.pixel_archive_path)
    checks["archive_sha256_and_patch_count_validate"] = archived
    checks["archive_metadata"] = archive_metadata

    provenance = evidence.provenance or {}
    checks["provenance_complete"] = bool(provenance) and all(
        provenance.get(key) for key in ("source", "identifier")
    )

    agreement = evidence.embedding_agreement or {}
    checks["embedding_agreement_audit_passed"] = bool(
        agreement.get("passed") is True and float(agreement.get("min_cosine", 0.0)) >= min_cosine
    )

    checks["releasable"] = all(
        [
            checks["caches_validate"],
            checks["archive_sha256_and_patch_count_validate"],
            checks["provenance_complete"],
            checks["embedding_agreement_audit_passed"],
        ]
    )
    checks["remote_recoverable"] = evidence.remote_recoverable
    return checks
