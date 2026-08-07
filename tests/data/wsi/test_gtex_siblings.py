"""GTEx invariant: a WSI is one SeriesInstanceUID with possibly many sibling
.dcm instances that must never be separated into a raw_flat-style view.
"""

from pathlib import Path
from unittest.mock import patch

from src.data.wsi.corpora import _find_gtex_representative, download_gtex, plan_gtex
from src.data.wsi.layout import DatasetRole, StoreLayout
from src.data.wsi.manifest import read_manifest


def test_find_gtex_representative_picks_deterministic_sibling(tmp_path: Path) -> None:
    series_dir = tmp_path / "SM_1.2.3"
    series_dir.mkdir()
    (series_dir / "b.dcm").write_bytes(b"fake")
    (series_dir / "a.dcm").write_bytes(b"fake")

    representative = _find_gtex_representative(tmp_path, "1.2.3")
    assert representative == series_dir / "a.dcm"
    # The sibling instance is untouched, in the same directory as the representative.
    assert (series_dir / "b.dcm").exists()
    assert representative.parent == (series_dir / "b.dcm").parent


def test_find_gtex_representative_missing_series_returns_none(tmp_path: Path) -> None:
    assert _find_gtex_representative(tmp_path, "does-not-exist") is None


def test_download_gtex_keeps_series_siblings_together(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "eaf_root"
    layout = StoreLayout.from_root(data_root)
    plan_path = layout.dataset_dir(DatasetRole.PRETRAINING, "gtex_eaf_wsi_v1") / "manifests" / "plan.csv"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        "source,patient_id,study_uid,series_uid,series_size_MB\n"
        "gtex,GTEX-DONOR1,1.1,2.2,123.0\n"
    )

    def fake_idc_download(cmd, check):  # noqa: ARG001 - matches subprocess.run signature
        # Simulate the official `idc download` CLI materializing every DICOM
        # instance of the series together, exactly as it does for a real series.
        series_uid = cmd[2]
        series_dir = layout.sources / "gtex" / f"SM_{series_uid}"
        series_dir.mkdir(parents=True, exist_ok=True)
        (series_dir / "instance_000.dcm").write_bytes(b"fake")
        (series_dir / "instance_001.dcm").write_bytes(b"fake")

    with patch("subprocess.run", side_effect=fake_idc_download):
        manifest_path = download_gtex(data_root, workers=1)

    records = read_manifest(manifest_path)
    assert len(records) == 1
    record = records[0]
    assert record.series_uid == "2.2"
    assert record.downloaded == "1"

    raw_path = Path(record.raw_path)
    # raw_path is a representative .dcm; every sibling instance from the same
    # series lives right beside it, never split into a separate flat view.
    siblings = sorted(p.name for p in raw_path.parent.glob("*.dcm"))
    assert siblings == ["instance_000.dcm", "instance_001.dcm"]
    assert raw_path.name in siblings
