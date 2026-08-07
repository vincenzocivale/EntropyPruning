from pathlib import Path

import pytest

from src.wsi_pipeline.archive import PixelArchiveSpec, verify_pixel_archive, write_pixel_archive


def test_pixel_archive_roundtrip(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    patches = [
        (0, 0, Image.new("RGB", (32, 32), (220, 180, 190))),
        (32, 0, Image.new("RGB", (32, 32), (150, 100, 120))),
    ]
    path = tmp_path / "slide.tar"
    write_pixel_archive(
        path,
        slide_id="slide",
        case_id="case",
        patches=patches,
        spec=PixelArchiveSpec(patch_size=32, jpeg_quality=95),
    )
    metadata = verify_pixel_archive(path)
    assert metadata["n_patches"] == 2
    assert metadata["slide_id"] == "slide"
    assert len(metadata["sha256"]) == 64
