from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SlideEntry:
    slide_id: str
    wsi_path: Path
    case_id: str | None = None
    project: str | None = None


def _resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    manifest_relative = root / path
    if manifest_relative.exists():
        return manifest_relative.resolve()
    collection_relative = root.parent / path
    if collection_relative.exists():
        return collection_relative.resolve()
    return manifest_relative.resolve()


def load_slides(path: Path) -> list[SlideEntry]:
    path = Path(path)
    table = pd.read_csv(path)
    required = {"slide_id", "wsi_path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    rows: list[SlideEntry] = []
    for row in table.to_dict("records"):
        rows.append(
            SlideEntry(
                slide_id=str(row["slide_id"]),
                wsi_path=_resolve(str(row["wsi_path"]), path.parent),
                case_id=None if pd.isna(row.get("case_id")) else str(row.get("case_id")),
                project=None if pd.isna(row.get("project")) else str(row.get("project")),
            )
        )
    return rows


def load_artifacts(
    path: Path,
    *,
    artifact_type: str | None = None,
    feature_set_id: str | None = None,
) -> pd.DataFrame:
    path = Path(path)
    table = pd.read_csv(path)
    if "slide_id" not in table or "path" not in table:
        raise ValueError(f"Artifact registry needs slide_id,path: {path}")
    if artifact_type is not None:
        if "artifact_type" not in table:
            raise ValueError("artifact_type filter requested but registry lacks the column")
        table = table[table["artifact_type"] == artifact_type]
    if feature_set_id is not None:
        if "feature_set_id" not in table:
            raise ValueError("feature_set_id filter requested but registry lacks the column")
        table = table[table["feature_set_id"] == feature_set_id]
    table = table.copy()
    table["path"] = table["path"].map(lambda value: str(_resolve(str(value), path.parent)))
    return table.reset_index(drop=True)


def write_manifest(rows: Iterable[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
