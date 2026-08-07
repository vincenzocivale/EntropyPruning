from pathlib import Path

from src.data.wsi.layout import DatasetRole, StoreLayout
from src.data.wsi.manifest import SlideRecord, assert_dataset_disjoint, validate_manifest


def test_canonical_layout(tmp_path: Path) -> None:
    layout = StoreLayout.from_root(tmp_path)
    layout.ensure_base_dirs()
    assert layout.dataset_dir(DatasetRole.PRETRAINING, "histai") == (
        tmp_path / "datasets" / "pretraining" / "histai"
    )
    assert layout.dataset_dir(DatasetRole.DOWNSTREAM, "panda") == (
        tmp_path / "datasets" / "downstream" / "panda"
    )
    assert layout.sources.is_dir()
    assert layout.caches.is_dir()
    assert layout.archives.is_dir()


def test_manifest_allows_local_case_namespaces() -> None:
    records = [
        SlideRecord(slide_id="a", case_id="mixed::case_001", source="histai", subset="mixed"),
        SlideRecord(slide_id="b", case_id="breast::case_001", source="histai", subset="breast"),
    ]
    validate_manifest(records)


def test_strict_overlap_guard() -> None:
    pretraining = [SlideRecord(slide_id="same", case_id="a", source="x", subset="s")]
    downstream = [SlideRecord(slide_id="same", case_id="b", source="x", subset="s")]
    try:
        assert_dataset_disjoint(pretraining, downstream)
    except ValueError:
        return
    raise AssertionError("expected overlap guard to fail")
