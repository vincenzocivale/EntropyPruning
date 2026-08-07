from pathlib import Path

import pytest

from src.data.wsi.corpora import build_strict_corpus, register_hest
from src.data.wsi.layout import DatasetRole, StoreLayout
from src.data.wsi.manifest import (
    SlideRecord,
    assert_no_path_under,
    group_disjoint_split,
    read_manifest,
    write_manifest,
)


def test_register_hest_discovers_wsi_without_copying(tmp_path: Path) -> None:
    hest_root = tmp_path / "hest_existing"
    (hest_root / "wsis").mkdir(parents=True)
    (hest_root / "wsis" / "slide_a.tiff").write_bytes(b"fake")
    (hest_root / "wsis" / "slide_b.svs").write_bytes(b"fake")
    (hest_root / "thumbnails" / "slide_a_thumb.tiff").parent.mkdir(parents=True)
    (hest_root / "thumbnails" / "slide_a_thumb.tiff").write_bytes(b"fake")

    data_root = tmp_path / "eaf_root"
    manifest_path = register_hest(data_root, hest_root)
    layout = StoreLayout.from_root(data_root)
    assert manifest_path == layout.manifest_path(DatasetRole.PRETRAINING, "hest_eaf_wsi_v1")

    records = read_manifest(manifest_path)
    assert {r.slide_id for r in records} == {"slide_a", "slide_b"}
    for record in records:
        assert record.source == "hest"
        assert record.downloaded == "1"
        # No copy: raw_path points straight at the original file under hest_root.
        assert Path(record.raw_path).resolve().is_relative_to(hest_root.resolve())


def test_register_hest_raises_when_root_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        register_hest(tmp_path / "eaf_root", tmp_path / "does-not-exist")


def test_build_strict_corpus_excludes_tcga_and_dedupes(tmp_path: Path) -> None:
    data_root = tmp_path / "eaf_root"
    layout = StoreLayout.from_root(data_root)

    tcga_root = layout.sources / "gdc" / "tcga"
    tcga_root.mkdir(parents=True)
    leaked = tcga_root / "cohort" / "leaked.svs"
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes(b"fake")

    histai_dir = layout.sources / "histai"
    histai_dir.mkdir(parents=True)
    slide1 = histai_dir / "s1.tiff"
    slide1.write_bytes(b"fake")

    histai_manifest = layout.manifest_path(DatasetRole.PRETRAINING, "histai_eaf_wsi_v1")
    write_manifest(
        histai_manifest,
        [
            SlideRecord(
                slide_id="mixed__case_001__s1",
                case_id="mixed::case_001",
                source="histai",
                subset="mixed",
                raw_path=str(slide1),
                downloaded="1",
            ),
            SlideRecord(
                slide_id="mixed__case_002__missing",
                case_id="mixed::case_002",
                source="histai",
                subset="mixed",
                raw_path=str(histai_dir / "missing.tiff"),
                downloaded="0",
            ),
        ],
    )

    manifest_path = build_strict_corpus(data_root, sources=("histai",), seed=1)
    records = read_manifest(manifest_path)
    # Only the downloaded, existing slide is included; the pending one is dropped
    # and nothing under the preserved TCGA root is ever pulled in.
    assert [r.slide_id for r in records] == ["mixed__case_001__s1"]
    assert records[0].split in {"train", "val", "holdout"}


def test_build_strict_corpus_leakage_guard(tmp_path: Path) -> None:
    data_root = tmp_path / "eaf_root"
    layout = StoreLayout.from_root(data_root)
    tcga_root = layout.sources / "gdc" / "tcga"
    leaked = tcga_root / "cohort" / "leaked.svs"
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes(b"fake")

    histai_manifest = layout.manifest_path(DatasetRole.PRETRAINING, "histai_eaf_wsi_v1")
    write_manifest(
        histai_manifest,
        [
            SlideRecord(
                slide_id="leaked",
                case_id="mixed::case_001",
                source="histai",
                subset="mixed",
                raw_path=str(leaked),
                downloaded="1",
            )
        ],
    )

    with pytest.raises(ValueError, match="Leakage guard"):
        build_strict_corpus(data_root, sources=("histai",), tcga_root=tcga_root)


def test_group_disjoint_split_keeps_case_together() -> None:
    records = [
        SlideRecord(slide_id=f"s{i}", case_id="caseA", source="x") for i in range(3)
    ] + [SlideRecord(slide_id=f"t{i}", case_id="caseB", source="x") for i in range(3)]
    split = group_disjoint_split(records, seed=0)
    by_case: dict[str, set[str]] = {}
    for row in split:
        by_case.setdefault(row.case_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in by_case.values())


def test_assert_no_path_under_flags_forbidden_root(tmp_path: Path) -> None:
    forbidden = tmp_path / "tcga"
    forbidden.mkdir()
    inside = SlideRecord(slide_id="a", case_id="c", source="x", raw_path=str(forbidden / "f.svs"))
    outside = SlideRecord(slide_id="b", case_id="c", source="x", raw_path=str(tmp_path / "f.svs"))
    assert_no_path_under([outside], forbidden)
    with pytest.raises(ValueError):
        assert_no_path_under([inside], forbidden)
