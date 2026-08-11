from pathlib import Path

import pytest

from src.wsi_pipeline.archive import (
    LifecycleStage,
    SlideLifecycleEvidence,
    classify_stage,
    embedding_agreement_audit,
    release_preconditions,
    write_pixel_archive,
)
from src.wsi_pipeline.cache_contracts import TileCacheSpec
from src.wsi_pipeline.cache_io import TileCacheWriter


def _write_coords_h5(path: Path, n: int = 3) -> Path:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=np.zeros((n, 2), dtype="int32"))
    return path


def _write_tile_cache(path: Path, n: int = 3) -> Path:
    pytest.importorskip("h5py")
    import numpy as np

    spec = TileCacheSpec(tile_encoder="test")
    with TileCacheWriter(path, spec, slide_id="s1", case_id="c1") as writer:
        writer.append(
            coords=np.zeros((n, 2), dtype="int32"),
            final_attention=np.zeros((n, 4), dtype="float16"),
            tile_embeddings=np.zeros((n, 8), dtype="float16"),
        )
    return path


def _write_archive(path: Path, n: int = 3) -> Path:
    Image = pytest.importorskip("PIL.Image")
    patches = [(i, i, Image.new("RGB", (8, 8), (10, 20, 30))) for i in range(n)]
    return write_pixel_archive(path, slide_id="s1", case_id="c1", patches=patches)


def test_lifecycle_progresses_through_every_stage(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    raw = tmp_path / "raw.svs"
    raw.write_bytes(b"fake")

    evidence = SlideLifecycleEvidence(slide_id="s1", raw_path=raw)
    assert classify_stage(evidence) == LifecycleStage.RAW

    coords = _write_coords_h5(tmp_path / "coords.h5")
    evidence = SlideLifecycleEvidence(slide_id="s1", raw_path=raw, coords_path=coords)
    assert classify_stage(evidence) == LifecycleStage.SEGMENTED

    tile_cache = _write_tile_cache(tmp_path / "tile_cache.h5")
    evidence = SlideLifecycleEvidence(
        slide_id="s1", raw_path=raw, coords_path=coords, tile_cache_paths=(tile_cache,)
    )
    assert classify_stage(evidence) == LifecycleStage.TEACHER_CACHES

    archive = _write_archive(tmp_path / "archive.tar")
    evidence = SlideLifecycleEvidence(
        slide_id="s1",
        raw_path=raw,
        coords_path=coords,
        tile_cache_paths=(tile_cache,),
        pixel_archive_path=archive,
    )
    assert classify_stage(evidence) == LifecycleStage.PIXEL_ARCHIVED

    agreement = embedding_agreement_audit([[1.0, 0.0]], [[1.0, 0.0]], min_cosine=0.99)
    assert agreement["passed"] is True
    evidence = SlideLifecycleEvidence(
        slide_id="s1",
        raw_path=raw,
        coords_path=coords,
        tile_cache_paths=(tile_cache,),
        pixel_archive_path=archive,
        embedding_agreement=agreement,
    )
    assert classify_stage(evidence) == LifecycleStage.VERIFIED

    evidence_releasable = SlideLifecycleEvidence(
        slide_id="s1",
        raw_path=raw,
        coords_path=coords,
        tile_cache_paths=(tile_cache,),
        pixel_archive_path=archive,
        embedding_agreement=agreement,
        provenance={"source": "histai", "identifier": "histai/HISTAI-mixed::case_001"},
    )
    assert classify_stage(evidence_releasable) == LifecycleStage.RAW_RELEASABLE


def test_release_preconditions_all_false_on_empty_evidence() -> None:
    evidence = SlideLifecycleEvidence(slide_id="s1")
    checks = release_preconditions(evidence)
    assert checks["releasable"] is False
    assert checks["caches_validate"] is False
    assert checks["archive_sha256_and_patch_count_validate"] is False
    assert checks["provenance_complete"] is False
    assert checks["embedding_agreement_audit_passed"] is False


def test_corrupt_archive_does_not_advance_past_pixel_archived(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    coords = _write_coords_h5(tmp_path / "coords.h5")
    tile_cache = _write_tile_cache(tmp_path / "tile_cache.h5")
    archive = _write_archive(tmp_path / "archive.tar")
    # Corrupt the archive after writing its checksum sidecar.
    archive.write_bytes(b"not a tar file anymore")
    evidence = SlideLifecycleEvidence(
        slide_id="s1",
        coords_path=coords,
        tile_cache_paths=(tile_cache,),
        pixel_archive_path=archive,
    )
    assert classify_stage(evidence) == LifecycleStage.PIXEL_ARCHIVED
    assert release_preconditions(evidence)["archive_sha256_and_patch_count_validate"] is False


def test_embedding_agreement_audit_fails_below_threshold() -> None:
    result = embedding_agreement_audit(
        [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], min_cosine=0.99
    )
    assert result["passed"] is False
    assert result["min_cosine"] < 0.99


def test_embedding_agreement_audit_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        embedding_agreement_audit([[1.0, 0.0]], [[1.0, 0.0, 0.0]])
