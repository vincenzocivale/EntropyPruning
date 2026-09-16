import pandas as pd

from src.data.wsi.splits import assign_case_splits


def test_case_disjoint_split_is_deterministic():
    rows = []
    for case in range(20):
        for slide in range(2):
            rows.append({"case_id": f"c{case}", "project": "histai", "slide_id": f"c{case}_s{slide}"})
    table = pd.DataFrame(rows)
    first = assign_case_splits(table, seed=17)
    second = assign_case_splits(table, seed=17)
    assert first["split"].tolist() == second["split"].tolist()
    assert first.groupby("case_id")["split"].nunique().max() == 1
    assert set(first["split"]) == {"train", "validation", "test"}
