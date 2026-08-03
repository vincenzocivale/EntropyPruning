import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from src.data.wsi import H5WSIFeatureStore, WSIBag


def _make_bag(
    slide_id: str,
    *,
    n_tiles: int,
    feature_dim: int,
    label: int | None,
    with_coords: bool,
    with_attention: bool,
    source: str,
) -> WSIBag:
    tile_features = torch.randn(n_tiles, feature_dim)
    coords = (
        torch.stack(
            [
                torch.arange(n_tiles, dtype=torch.long),
                torch.arange(n_tiles, dtype=torch.long) + 100,
            ],
            dim=1,
        )
        if with_coords
        else None
    )
    attention = torch.softmax(torch.randn(n_tiles), dim=0) if with_attention else None

    return WSIBag(
        slide_id=slide_id,
        tile_features=tile_features,
        coords=coords,
        label=label,
        attention=attention,
        metadata={"source": source},
    )


def test_inspect_wsi_feature_store_cli_prints_summary_and_writes_json(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    output_json = tmp_path / "summary.json"

    store = H5WSIFeatureStore(store_path)
    store.write(
        _make_bag(
            "slide_001",
            n_tiles=4,
            feature_dim=8,
            label=0,
            with_coords=True,
            with_attention=True,
            source="generic_feature_file",
        )
    )
    store.write(
        _make_bag(
            "slide_002",
            n_tiles=6,
            feature_dim=8,
            label=1,
            with_coords=True,
            with_attention=False,
            source="trident",
        )
    )
    store.write(
        _make_bag(
            "slide_003",
            n_tiles=5,
            feature_dim=8,
            label=None,
            with_coords=False,
            with_attention=False,
            source="trident",
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--output-json",
            str(output_json),
            "--max-examples",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_summary = json.loads(result.stdout)
    file_summary = json.loads(output_json.read_text())

    assert stdout_summary == file_summary
    assert stdout_summary["valid"] is True
    assert stdout_summary["n_slides"] == 3
    assert stdout_summary["tile_count"]["min"] == 4
    assert stdout_summary["tile_count"]["max"] == 6
    assert stdout_summary["tile_count"]["total"] == 15
    assert stdout_summary["feature_dims"] == {"8": 3}

    assert stdout_summary["coords"]["n_with"] == 2
    assert stdout_summary["coords"]["n_without"] == 1
    assert stdout_summary["coords"]["examples_without"] == ["slide_003"]

    assert stdout_summary["attention"]["n_with"] == 1
    assert stdout_summary["attention"]["n_without"] == 2

    assert stdout_summary["labels"]["n_with"] == 2
    assert stdout_summary["labels"]["n_without"] == 1
    assert stdout_summary["labels"]["distribution"] == {"0": 1, "1": 1}

    assert stdout_summary["metadata"]["source_distribution"] == {
        "generic_feature_file": 1,
        "trident": 2,
    }


def test_inspect_wsi_feature_store_cli_handles_empty_store(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "empty.h5"

    _ = H5WSIFeatureStore(store_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["valid"] is False
    assert summary["n_slides"] == 0
    assert summary["reason"] == "feature store contains no slides"


def test_inspect_wsi_feature_store_cli_rejects_negative_max_examples(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    store_path = tmp_path / "features.h5"
    _ = H5WSIFeatureStore(store_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_wsi_feature_store.py",
            "--feature-store",
            str(store_path),
            "--max-examples",
            "-1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--max-examples" in result.stderr
