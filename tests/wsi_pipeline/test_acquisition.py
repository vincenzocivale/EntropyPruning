from __future__ import annotations

from pathlib import Path

from src.wsi_pipeline.acquisition import (
    AcquisitionItem,
    AcquisitionPlan,
    build_plan,
    discover_gdc,
    load_plan,
    save_plan,
    select_budgeted_items,
    write_provider_manifests,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        return _FakeResponse(self.pages.pop(0))


def _gdc_hit(file_id: str, project: str, size: int):
    return {
        "file_id": file_id,
        "file_name": f"{file_id}.svs",
        "file_size": size,
        "md5sum": f"md5-{file_id}",
        "state": "released",
        "experimental_strategy": "Diagnostic Slide",
        "cases": [
            {
                "submitter_id": f"case-{file_id}",
                "project": {"project_id": project},
                "samples": [{"sample_type": "Primary Tumor"}],
            }
        ],
    }


def test_discover_gdc_paginates_and_filters_all_tcga():
    session = _FakeSession(
        [
            {
                "data": {
                    "hits": [_gdc_hit("a", "TCGA-BRCA", 10), _gdc_hit("b", "CPTAC-LUAD", 20)],
                    "pagination": {"total": 3},
                }
            },
            {
                "data": {
                    "hits": [_gdc_hit("c", "TCGA-COAD", 30)],
                    "pagination": {"total": 3},
                }
            },
        ]
    )
    items = discover_gdc(all_tcga=True, page_size=2, session=session)
    assert [item.item_id for item in items] == ["a", "c"]
    assert len(session.calls) == 2
    assert session.calls[1][1]["from"] == 2


def test_budgeted_selection_is_balanced_and_deterministic():
    items = [
        AcquisitionItem("gdc", f"a{i}", f"a{i}.svs", 10, cohort="A") for i in range(4)
    ] + [
        AcquisitionItem("gdc", f"b{i}", f"b{i}.svs", 10, cohort="B") for i in range(4)
    ]
    first = select_budgeted_items(items, max_items=4, per_cohort=3, seed=17)
    second = select_budgeted_items(items, max_items=4, per_cohort=3, seed=17)
    assert first == second
    assert {item.cohort for item in first[:2]} == {"A", "B"}
    assert len(first) == 4


def test_budgeted_selection_obeys_byte_limit():
    items = [
        AcquisitionItem("gdc", "a", "a.svs", 70, cohort="A"),
        AcquisitionItem("gdc", "b", "b.svs", 40, cohort="B"),
        AcquisitionItem("gdc", "c", "c.svs", 30, cohort="C"),
    ]
    selected = select_budgeted_items(items, max_bytes=100, seed=0)
    assert sum(item.size_bytes for item in selected) <= 100


def test_plan_round_trip_and_gdc_manifest(tmp_path: Path):
    item = AcquisitionItem(
        "gdc",
        "uuid",
        "slide.svs",
        123,
        cohort="TCGA-BRCA",
        case_id="TCGA-XX-0001",
        md5="abc",
    )
    plan = build_plan(
        provider="gdc",
        source_id="tcga_gdc",
        all_items=[item],
        filters={"diagnostic_only": True},
        max_items=None,
        max_gib=None,
        per_cohort=None,
        seed=0,
    )
    path = save_plan(plan, tmp_path)
    write_provider_manifests(plan, tmp_path)
    loaded = load_plan(path)
    assert loaded.total_bytes == 123
    assert loaded.items[0].case_id == "TCGA-XX-0001"
    manifest = (tmp_path / "gdc_manifest.tsv").read_text()
    assert "uuid\tslide.svs\tabc\t123\treleased" in manifest
    assert (tmp_path / "slides.csv").exists()
