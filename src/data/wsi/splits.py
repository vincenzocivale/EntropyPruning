"""Deterministic case-disjoint splits shared by WSI-EAF training stages."""

from __future__ import annotations

import pandas as pd

from src.wsi_pipeline.utils import stable_seed


def assign_case_splits(
    table: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 17,
    split_column: str = "split",
) -> pd.DataFrame:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no test split")
    required = {"case_id", "project"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    cases = table[["case_id", "project"]].drop_duplicates("case_id").copy()
    assignments: list[dict[str, str]] = []
    for project, group in cases.groupby("project", sort=True):
        ordered = group.assign(
            _hash=group["case_id"].astype(str).map(
                lambda case: stable_seed(f"{project}:{case}", seed)
            )
        ).sort_values(["_hash", "case_id"])
        n = len(ordered)
        if n <= 2:
            n_train, n_val = max(1, n - 1), 0
        else:
            n_train = max(1, int(round(train_fraction * n)))
            n_val = max(1, int(round(validation_fraction * n)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        for index, row in enumerate(ordered.itertuples(index=False)):
            split = "train" if index < n_train else (
                "validation" if index < n_train + n_val else "test"
            )
            assignments.append({"case_id": str(row.case_id), split_column: split})

    split_table = pd.DataFrame(assignments)
    result = table.merge(split_table, on="case_id", how="left", validate="many_to_one")
    if result[split_column].isna().any():
        raise RuntimeError("Failed to assign every case to a split")
    return result
