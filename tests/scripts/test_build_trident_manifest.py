import csv
import subprocess
import sys
from pathlib import Path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_trident_manifest_cli_writes_manifest_with_relative_paths(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    features_dir = tmp_path / "trident_processed/20x_256px_0px_overlap/features_uni_v1"
    coords_dir = tmp_path / "trident_processed/20x_256px_0px_overlap/patches"
    labels_csv = tmp_path / "labels.csv"
    manifest = tmp_path / "manifest.csv"

    _touch(features_dir / "slide_002.h5")
    _touch(features_dir / "slide_001.h5")
    _touch(coords_dir / "slide_001.h5")
    _touch(coords_dir / "slide_002.h5")

    labels_csv.write_text("slide_id,label\nslide_001,0\nslide_002,1\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_trident_manifest.py",
            "--features-dir",
            str(features_dir),
            "--coords-dir",
            str(coords_dir),
            "--labels-csv",
            str(labels_csv),
            "--output-manifest",
            str(manifest),
            "--require-coords",
            "--require-labels",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 2 rows" in result.stdout

    rows = _read_rows(manifest)
    assert [row["slide_id"] for row in rows] == ["slide_001", "slide_002"]
    assert rows[0]["features_path"] == "trident_processed/20x_256px_0px_overlap/features_uni_v1/slide_001.h5"
    assert rows[0]["coords_path"] == "trident_processed/20x_256px_0px_overlap/patches/slide_001.h5"
    assert rows[0]["label"] == "0"
    assert rows[1]["label"] == "1"


def test_build_trident_manifest_cli_supports_absolute_paths(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    features_dir = tmp_path / "features"
    manifest = tmp_path / "manifest.csv"

    feature_path = features_dir / "slide_001.h5"
    _touch(feature_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/build_trident_manifest.py",
            "--features-dir",
            str(features_dir),
            "--output-manifest",
            str(manifest),
            "--absolute-paths",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = _read_rows(manifest)
    assert rows[0]["features_path"] == str(feature_path.resolve())


def test_build_trident_manifest_cli_rejects_missing_required_coords(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    features_dir = tmp_path / "features"
    coords_dir = tmp_path / "coords"
    manifest = tmp_path / "manifest.csv"

    _touch(features_dir / "slide_001.h5")
    coords_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_trident_manifest.py",
            "--features-dir",
            str(features_dir),
            "--coords-dir",
            str(coords_dir),
            "--output-manifest",
            str(manifest),
            "--require-coords",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing coords" in result.stderr


def test_build_trident_manifest_cli_rejects_missing_required_labels(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    features_dir = tmp_path / "features"
    labels_csv = tmp_path / "labels.csv"
    manifest = tmp_path / "manifest.csv"

    _touch(features_dir / "slide_001.h5")
    labels_csv.write_text("slide_id,label\nslide_999,1\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_trident_manifest.py",
            "--features-dir",
            str(features_dir),
            "--labels-csv",
            str(labels_csv),
            "--output-manifest",
            str(manifest),
            "--require-labels",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing labels" in result.stderr


def test_build_trident_manifest_cli_refuses_overwrite_by_default(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    features_dir = tmp_path / "features"
    manifest = tmp_path / "manifest.csv"

    _touch(features_dir / "slide_001.h5")
    manifest.write_text("already here")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_trident_manifest.py",
            "--features-dir",
            str(features_dir),
            "--output-manifest",
            str(manifest),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output manifest already exists" in result.stderr


def test_build_trident_manifest_cli_overwrites_when_requested(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    features_dir = tmp_path / "features"
    manifest = tmp_path / "manifest.csv"

    _touch(features_dir / "slide_001.h5")
    manifest.write_text("already here")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_trident_manifest.py",
            "--features-dir",
            str(features_dir),
            "--output-manifest",
            str(manifest),
            "--overwrite",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = _read_rows(manifest)
    assert len(rows) == 1
    assert rows[0]["slide_id"] == "slide_001"
