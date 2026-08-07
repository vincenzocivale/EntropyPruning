#!/usr/bin/env python3
"""Prepare an additive, dataset-disjoint EAF-WSI pretraining corpus.

This script NEVER deletes or moves existing datasets. In particular, an existing
TCGA corpus is only registered as preserved and is excluded from the strict-v1
manifest by construction.

All output is written under the *canonical* `$EAF_WSI_ROOT` store described in
`docs/data_layout.md`: new HISTAI slides get their one physical copy under
`sources/histai/`, and every generated dataset (`histai_eaf_wsi_v1`,
`eaf_wsi_pretrain_strict_v1`) lives at `datasets/pretraining/<name>/` next to
`tcga_eaf_multicohort_v1`, with the same `manifests/` (+ `views/raw_flat/`
symlinks for newly downloaded slides) sub-layout. There is no second,
parallel root — `--data-root`/`EAF_WSI_ROOT` is the only root this script
knows about, matching every other script in the repo.

Typical workflow
----------------
1) Initialize the new view and register existing TCGA/HEST roots. TCGA
   defaults to `$EAF_WSI_ROOT/sources/gdc/tcga` (the canonical location) and
   only needs `--tcga-root` if that raw corpus lives elsewhere::

    python scripts/wsi_prepare_strict_pretraining.py init \
      --data-root "$EAF_WSI_ROOT" \
      --hest-root /path/to/existing/hest/wsis

2) Plan HISTAI without downloading any WSI::

    python scripts/wsi_prepare_strict_pretraining.py plan-histai \
      --data-root "$EAF_WSI_ROOT"

3) Download exactly the planned HISTAI WSI subset (requires accepted HF gates)::

    HF_XET_HIGH_PERFORMANCE=1 python scripts/wsi_prepare_strict_pretraining.py download-histai \
      --data-root "$EAF_WSI_ROOT" --workers 8

4) Inventory existing HEST WSIs and build strict splits::

    python scripts/wsi_prepare_strict_pretraining.py scan-hest \
      --data-root "$EAF_WSI_ROOT" \
      --hest-root /path/to/existing/hest/wsis

    python scripts/wsi_prepare_strict_pretraining.py build \
      --data-root "$EAF_WSI_ROOT"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from tqdm.auto import tqdm

VIEW_NAME = "eaf_wsi_pretrain_strict_v1"
HISTAI_DATASET_NAME = "histai_eaf_wsi_v1"

HISTAI_SUBSETS = (
    "HISTAI-mixed",
    "HISTAI-skin-b2",
    "HISTAI-skin-b1",
    "HISTAI-colorectal-b1",
    "HISTAI-breast",
    "HISTAI-thorax",
    "HISTAI-gastrointestinal",
    "HISTAI-hematologic",
    "HISTAI-colorectal-b2",
)

WSI_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".dcm"}
EXCLUDED_PATH_PARTS = {
    "thumbnails",
    "thumbnail",
    "spatial_plots",
    "spatial_plot",
    "tissue_seg",
    "patches",
    "trident",
    "features",
}

# Canonical manifest schema (see docs/data_layout.md#canonical-slide-manifest).
MANIFEST_FIELDS = [
    "slide_id",
    "case_id",
    "cohort",
    "source",
    "subset",
    "stain",
    "raw_path",
    "coords_path",
    "downloaded",
]


def _paths(data_root: Path) -> dict[str, Path]:
    """Resolve every output path under the single canonical `$EAF_WSI_ROOT` store.

    Everything nests under `datasets/pretraining/<dataset_name>/`, mirroring the
    existing `tcga_eaf_multicohort_v1` layout, instead of a second top-level
    `pretraining/` + `views/` root.
    """
    data_root = data_root.expanduser().resolve()
    pretraining = data_root / "datasets" / "pretraining"
    histai_dataset = pretraining / HISTAI_DATASET_NAME
    strict_dataset = pretraining / VIEW_NAME
    return {
        "root": data_root,
        "pretraining": pretraining,
        "tcga_sources_default": data_root / "sources" / "gdc" / "tcga",
        "sources_histai": data_root / "sources" / "histai",
        "histai_dataset": histai_dataset,
        "histai_manifests": histai_dataset / "manifests",
        "histai_views_raw_flat": histai_dataset / "views" / "raw_flat",
        "histai_plan": histai_dataset / "manifests" / "plan.csv",
        "histai_manifest": histai_dataset / "manifests" / "slides.csv",
        "strict_dataset": strict_dataset,
        "strict_manifests": strict_dataset / "manifests",
        "registry": strict_dataset / "registry.json",
        "hest_manifest": strict_dataset / "manifests" / "hest_existing.csv",
        "strict_manifest": strict_dataset / "manifests" / "slides.csv",
    }


def _mkdirs(p: dict[str, Path]) -> None:
    p["pretraining"].mkdir(parents=True, exist_ok=True)
    p["sources_histai"].mkdir(parents=True, exist_ok=True)
    p["histai_manifests"].mkdir(parents=True, exist_ok=True)
    p["histai_views_raw_flat"].mkdir(parents=True, exist_ok=True)
    p["strict_manifests"].mkdir(parents=True, exist_ok=True)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _write_dataset_yaml(dataset_dir: Path, name: str, description: str) -> None:
    """Write a minimal `dataset.yaml`, matching the field the canonical layout
    expects at the root of every `datasets/pretraining/<name>/` directory."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.exists():
        return
    yaml_path.write_text(
        f'name: "{name}"\n'
        f'description: "{description}"\n'
        'manifest: "manifests/slides.csv"\n'
    )


def _ensure_symlink(link_path: Path, target: Path) -> None:
    """Create/refresh a symlink, matching the "views are symlinks, sources hold
    the one physical copy" rule from docs/data_layout.md."""
    target = target.resolve()
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() == target:
            return
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


def cmd_init(args: argparse.Namespace) -> None:
    p = _paths(args.data_root)
    _mkdirs(p)

    if args.tcga_root:
        tcga = args.tcga_root.expanduser().resolve()
        if not tcga.exists():
            raise FileNotFoundError(f"Existing TCGA root does not exist: {tcga}")
    else:
        # Canonical location per docs/data_layout.md: sources/gdc/tcga/<cohort>/...
        tcga = p["tcga_sources_default"] if p["tcga_sources_default"].exists() else None

    hest = args.hest_root.expanduser().resolve() if args.hest_root else None
    if hest is not None and not hest.exists():
        raise FileNotFoundError(f"Existing HEST root does not exist: {hest}")

    registry = {
        "schema_version": 2,
        "view": VIEW_NAME,
        "data_root": str(p["root"]),
        "policy": "dataset-disjoint pretraining; TCGA preserved but excluded",
        "sources": {
            "tcga": {
                "path": str(tcga) if tcga else None,
                "role": "preserved_not_in_strict_v1",
                "mutable": False,
            },
            "hest": {
                "path": str(hest) if hest else None,
                "role": "strict_candidate_existing",
                "mutable": False,
            },
            "histai": {
                "path": str(p["histai_dataset"]),
                "raw_sources_path": str(p["sources_histai"]),
                "role": "strict_pretraining",
                "mutable": True,
            },
            "gtex": {
                "path": str(p["pretraining"] / "gtex_eaf_wsi_v1"),
                "role": "strict_candidate_after_dicom_smoke_test",
                "mutable": True,
            },
        },
    }
    p["registry"].write_text(json.dumps(registry, indent=2) + "\n")
    _write_dataset_yaml(
        p["histai_dataset"],
        name=HISTAI_DATASET_NAME,
        description="HISTAI H&E WSIs acquired for EAF-WSI strict-v1 pretraining.",
    )
    _write_dataset_yaml(
        p["strict_dataset"],
        name=VIEW_NAME,
        description="Dataset-disjoint EAF pretraining corpus (HISTAI + existing HEST; TCGA excluded).",
    )

    print(f"[eaf-wsi-data] initialized: {p['strict_dataset']}")
    if tcga:
        print(f"[eaf-wsi-data] TCGA preserved in place and excluded from strict-v1: {tcga}")
    else:
        print("[eaf-wsi-data] no TCGA root found/given; leakage guard will be skipped")
    if hest:
        print(f"[eaf-wsi-data] HEST registered read-only: {hest}")
    print(f"[eaf-wsi-data] new HISTAI slides will be stored under: {p['sources_histai']}")


def _histai_case_id(filename: str) -> str | None:
    for part in Path(filename).parts:
        if part.startswith("case_"):
            return part
    return None


def _is_histai_he(filename: str) -> bool:
    name = Path(filename).name
    # Official layouts include slide_H&E_0.tiff and
    # slide_<magnification>_H&E_<n>.tiff.
    return bool(re.match(r"^slide_(?:[^_]+_)?H&E_\d+\.tiff$", name, flags=re.IGNORECASE))


def _histai_magnification(filename: str) -> str:
    """Best-effort magnification parsed from the HISTAI filename."""
    name = Path(filename).name.lower()
    match = re.search(r"slide_([^_]+)_h&e_\d+\.tiff$", name)
    return match.group(1) if match else ""


def _histai_slide_rank(filename: str) -> tuple[int, str]:
    """Prefer native 20x H&E, then unknown, then other magnifications."""
    mag = _histai_magnification(filename)
    if mag in {"20x", "x20", "20"}:
        priority = 0
    elif not mag:
        priority = 1
    else:
        priority = 2
    return priority, filename


def _one_histai_slide_per_case(files: list[str]) -> list[str]:
    """Select exactly one deterministic H&E WSI for every HISTAI case."""
    by_case: dict[str, list[str]] = defaultdict(list)
    for filename in files:
        case = _histai_case_id(filename)
        if case is not None:
            by_case[case].append(filename)

    selected: list[str] = []
    for case in sorted(by_case):
        selected.append(min(by_case[case], key=_histai_slide_rank))
    return selected


def cmd_plan_histai(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install dependency: pip install -U huggingface_hub hf_xet") from exc

    p = _paths(args.data_root)
    _mkdirs(p)
    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    rows: list[dict[str, str]] = []
    for subset in HISTAI_SUBSETS:
        repo_id = f"histai/{subset}"
        print(f"[eaf-wsi-data] listing {repo_id} ...", flush=True)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        eligible = [f for f in files if _is_histai_he(f) and _histai_case_id(f)]
        selected = _one_histai_slide_per_case(eligible)

        for filename in selected:
            rows.append(
                {
                    "source": "histai",
                    "subset": subset,
                    "repo_id": repo_id,
                    "case_id": _histai_case_id(filename) or "",
                    "repo_path": filename,
                    "stain": "H&E",
                    "native_magnification": _histai_magnification(filename),
                    "planned": "1",
                }
            )

        n_cases = len({_histai_case_id(f) for f in eligible})
        print(
            f"[eaf-wsi-data] {subset}: eligible_he={len(eligible)} "
            f"unique_cases={n_cases} selected={len(selected)}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No eligible HISTAI H&E WSI found.")

    with p["histai_plan"].open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[eaf-wsi-data] wrote HISTAI one-WSI-per-case plan: "
        f"{p['histai_plan']} ({len(rows)} WSIs / cases)"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_download_histai(args: argparse.Namespace) -> None:
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from huggingface_hub import hf_hub_download
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies: pip install -U huggingface_hub hf_xet tqdm"
        ) from exc

    p = _paths(args.data_root)
    _mkdirs(p)

    if not p["histai_plan"].exists():
        raise FileNotFoundError(
            f"Missing plan. Run plan-histai first: {p['histai_plan']}"
        )

    token = args.token or os.environ.get("HF_TOKEN")

    # ------------------------------------------------------------------
    # Load full frozen plan, then optionally select only requested subsets.
    # ------------------------------------------------------------------

    all_rows = _read_csv(p["histai_plan"])

    requested_subsets = set(args.subset or [])

    if requested_subsets:
        rows = [
            row
            for row in all_rows
            if row["subset"] in requested_subsets
        ]
    else:
        rows = all_rows

    if not rows:
        raise RuntimeError(
            "No HISTAI rows selected. "
            f"Requested subsets: {sorted(requested_subsets)}"
        )

    print(
        f"[eaf-wsi-data] selected for this run: "
        f"{len(rows)}/{len(all_rows)} planned WSI",
        flush=True,
    )

    if requested_subsets:
        print(
            "[eaf-wsi-data] subsets: "
            + ", ".join(sorted(requested_subsets)),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Group selected rows by repository.
    # ------------------------------------------------------------------

    by_repo: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_repo[row["repo_id"]].append(row)

    print(
        f"[eaf-wsi-data] HISTAI download: "
        f"{len(rows)} WSI across {len(by_repo)} repositories",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Download one specific planned file at a time.
    #
    # This avoids snapshot_download(... allow_patterns=[20k paths]),
    # which can spend a long time resolving/filtering the repository
    # before showing any progress.
    # ------------------------------------------------------------------

    failures: list[tuple[dict[str, str], str]] = []

    def download_one(row: dict[str, str]) -> tuple[dict[str, str], Path]:
        subset = row["subset"]
        repo_id = row["repo_id"]
        repo_path = row["repo_path"]

        local_dir = p["sources_histai"] / subset
        local_dir.mkdir(parents=True, exist_ok=True)

        expected_path = local_dir / repo_path

        # Fast local resume check.
        if expected_path.is_file():
            return row, expected_path

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            local_dir=local_dir,
            token=token,
        )

        return row, Path(downloaded_path)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row): row
            for row in rows
        }

        with tqdm(
            total=len(futures),
            desc="HISTAI WSI",
            unit="WSI",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                row = futures[future]

                try:
                    future.result()
                except Exception as exc:
                    failures.append((row, repr(exc)))

                    tqdm.write(
                        "[eaf-wsi-data] FAILED "
                        f"{row['subset']} / {row['case_id']} / "
                        f"{row['repo_path']}: {exc}"
                    )

                progress.set_postfix(
                    failed=len(failures),
                    refresh=False,
                )
                progress.update(1)

    # ------------------------------------------------------------------
    # Rebuild canonical manifest from the FULL frozen HISTAI plan.
    #
    # This is intentional:
    #   downloaded=1 -> already available
    #   downloaded=0 -> gated/not downloaded yet
    # ------------------------------------------------------------------

    out_rows: list[dict[str, str]] = []

    downloaded_total = 0
    requested_missing = 0

    requested_keys = {
        (row["repo_id"], row["repo_path"])
        for row in rows
    }

    for row in tqdm(
        all_rows,
        total=len(all_rows),
        desc="Building HISTAI manifest",
        unit="WSI",
        dynamic_ncols=True,
    ):
        local = (
            p["sources_histai"]
            / row["subset"]
            / row["repo_path"]
        )

        exists = local.is_file()

        if exists:
            downloaded_total += 1

        if (
            not exists
            and (row["repo_id"], row["repo_path"]) in requested_keys
        ):
            requested_missing += 1

        slide_id = Path(row["repo_path"]).stem

        if exists:
            # Important: slide_H&E_0.tiff repeats across cases.
            # Include subset + case_id to avoid raw_flat collisions.
            link_name = (
                f"{row['subset']}__"
                f"{row['case_id']}__"
                f"{slide_id}"
                f"{local.suffix}"
            )

            _ensure_symlink(
                p["histai_views_raw_flat"] / link_name,
                local,
            )

        out_rows.append(
            {
                "slide_id": slide_id,
                "case_id": row["case_id"],
                "cohort": row["subset"],
                "source": "histai",
                "subset": row["subset"],
                "stain": "H&E",
                "raw_path": (
                    str(local.resolve())
                    if exists
                    else str(local)
                ),
                "coords_path": "",
                "downloaded": "1" if exists else "0",
            }
        )

    with p["histai_manifest"].open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=MANIFEST_FIELDS,
        )
        writer.writeheader()
        writer.writerows(out_rows)

    _write_dataset_yaml(
        p["histai_dataset"],
        name=HISTAI_DATASET_NAME,
        description=(
            "HISTAI H&E WSIs acquired for "
            "EAF-WSI strict-v1 pretraining."
        ),
    )

    print()
    print(
        f"[eaf-wsi-data] HISTAI slide manifest: "
        f"{p['histai_manifest']}",
        flush=True,
    )

    print(
        f"[eaf-wsi-data] CURRENT HISTAI POOL: "
        f"{downloaded_total}/{len(all_rows)} WSI available "
        f"({100.0 * downloaded_total / len(all_rows):.2f}%)",
        flush=True,
    )

    print(
        f"[eaf-wsi-data] THIS RUN: "
        f"{len(rows) - requested_missing}/{len(rows)} requested WSI available",
        flush=True,
    )

    if failures:
        print(
            f"[eaf-wsi-data] failures during this run: {len(failures)}",
            flush=True,
        )

    if requested_missing:
        raise RuntimeError(
            f"HISTAI run finished with "
            f"{requested_missing}/{len(rows)} requested files missing."
        )
        
def _discover_hest_wsi(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    candidates: list[Path] = []
    preferred: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in WSI_SUFFIXES:
            continue
        lower_parts = {p.lower() for p in path.parts}
        if lower_parts & EXCLUDED_PATH_PARTS:
            continue
        candidates.append(path)
        if "wsis" in lower_parts or "raw_wsi" in lower_parts:
            preferred.append(path)
    return sorted(preferred if preferred else candidates)


def cmd_scan_hest(args: argparse.Namespace) -> None:
    p = _paths(args.data_root)
    _mkdirs(p)
    hest_root = args.hest_root.expanduser().resolve()
    if not hest_root.exists():
        raise FileNotFoundError(hest_root)

    slides = _discover_hest_wsi(hest_root)
    rows = []
    for slide in slides:
        slide_id = slide.stem
        rows.append(
            {
                "slide_id": slide_id,
                # If a patient-level HEST mapping is available later, replace this
                # with the patient/study grouping before the final paper split.
                "case_id": slide_id,
                "cohort": "hest",
                "source": "hest",
                "subset": "existing",
                "stain": "H&E",
                "raw_path": str(slide),
                "coords_path": "",
                "downloaded": "1",
            }
        )
    if not rows:
        raise RuntimeError(f"No WSI files discovered under {hest_root}")

    with p["hest_manifest"].open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eaf-wsi-data] HEST inventory: {p['hest_manifest']} ({len(rows)} WSIs)")


def _load_registry(p: dict[str, Path]) -> dict:
    if not p["registry"].exists():
        raise FileNotFoundError(f"Run init first: {p['registry']}")
    return json.loads(p["registry"].read_text())


def _group_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = f"{row['source']}::{row['case_id']}"
        by_group[group].append(row)

    groups = list(by_group)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_val = max(1, round(n * 0.05)) if n >= 20 else max(1, n // 10)
    n_holdout = max(1, round(n * 0.05)) if n >= 20 else max(1, n // 10)
    n_train = max(0, n - n_val - n_holdout)
    assignment = {
        **{g: "train" for g in groups[:n_train]},
        **{g: "val" for g in groups[n_train : n_train + n_val]},
        **{g: "holdout" for g in groups[n_train + n_val :]},
    }

    split_rows = {"train": [], "val": [], "holdout": []}
    for group, items in by_group.items():
        split = assignment[group]
        for row in items:
            new_row = dict(row)
            new_row["split"] = split
            split_rows[split].append(new_row)
    return split_rows


def cmd_build(args: argparse.Namespace) -> None:
    p = _paths(args.data_root)
    _mkdirs(p)
    registry = _load_registry(p)

    manifests: list[Path] = []
    if p["histai_manifest"].exists():
        manifests.append(p["histai_manifest"])
    if p["hest_manifest"].exists():
        manifests.append(p["hest_manifest"])
    if not manifests:
        raise RuntimeError("No source manifests found. Download HISTAI and/or scan HEST first.")

    rows: list[dict[str, str]] = []
    tcga_path_raw = registry["sources"]["tcga"].get("path")
    tcga_root = Path(tcga_path_raw).resolve() if tcga_path_raw else None

    seen_paths: set[Path] = set()
    for manifest in manifests:
        for row in _read_csv(manifest):
            if row.get("downloaded", "1") != "1":
                continue
            raw_path = Path(row["raw_path"]).expanduser().resolve()
            if not raw_path.is_file():
                continue
            if tcga_root is not None and _is_relative_to(raw_path, tcga_root):
                raise RuntimeError(
                    f"Strict-v1 leakage guard: source path resolves inside preserved TCGA: {raw_path}"
                )
            if raw_path in seen_paths:
                continue
            seen_paths.add(raw_path)
            rows.append(
                {
                    "slide_id": row["slide_id"],
                    "case_id": row["case_id"],
                    "cohort": row["cohort"],
                    "source": row["source"],
                    "subset": row["subset"],
                    "stain": row.get("stain", "H&E"),
                    "raw_path": str(raw_path),
                    "coords_path": row.get("coords_path", ""),
                }
            )

    # Canonical manifest schema: split lives as a column on the single
    # manifests/slides.csv (docs/data_layout.md "optional split column"), not
    # as separate splits/*.csv files.
    split_rows = _group_split(rows, args.seed)
    all_rows = split_rows["train"] + split_rows["val"] + split_rows["holdout"]
    fieldnames = list(all_rows[0])
    with p["strict_manifest"].open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    by_source: dict[str, int] = defaultdict(int)
    for row in all_rows:
        by_source[row["source"]] += 1
    print(f"[eaf-wsi-data] strict manifest: {p['strict_manifest']} ({len(all_rows)} WSIs)")
    for source, count in sorted(by_source.items()):
        print(f"  - {source}: {count}")
    for split in ("train", "val", "holdout"):
        print(f"  - {split}: {len(split_rows[split])}")
    if tcga_root:
        print(f"[eaf-wsi-data] verified: no strict-v1 path is under preserved TCGA root {tcga_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_root(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--data-root",
            type=Path,
            default=Path(os.environ["EAF_WSI_ROOT"]) if os.environ.get("EAF_WSI_ROOT") else None,
            required="EAF_WSI_ROOT" not in os.environ,
            help="Canonical WSI data root (see docs/data_layout.md). Defaults to $EAF_WSI_ROOT.",
        )

    p = sub.add_parser("init", help="Register existing datasets in-place without moving them.")
    add_data_root(p)
    p.add_argument(
        "--tcga-root",
        type=Path,
        help="Existing TCGA raw-slide root; defaults to <data-root>/sources/gdc/tcga.",
    )
    p.add_argument("--hest-root", type=Path)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser(
        "plan-histai",
        help="Select exactly one H&E WSI per HISTAI case; downloads no slide pixels.",
    )
    add_data_root(p)
    p.add_argument("--token", help="HF token; defaults to HF_TOKEN / cached HF login.")
    p.set_defaults(func=cmd_plan_histai)

    p = sub.add_parser(
        "download-histai",
        help="Download files listed in the frozen HISTAI plan."
    )
    add_data_root(p)

    p.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    p.add_argument(
        "--subset",
        action="append",
        choices=HISTAI_SUBSETS,
        help=(
            "Download only this HISTAI subset. "
            "May be supplied multiple times. "
            "If omitted, all subsets are attempted."
        ),
    )

    p.add_argument(
        "--token",
        help="HF token; defaults to HF_TOKEN / cached HF login."
    )

    p.set_defaults(func=cmd_download_histai)

    p = sub.add_parser("scan-hest", help="Inventory an existing HEST raw-WSI tree without copying it.")
    add_data_root(p)
    p.add_argument("--hest-root", type=Path, required=True)
    p.set_defaults(func=cmd_scan_hest)

    p = sub.add_parser("build", help="Build strict manifest/splits from downloaded HISTAI + existing HEST.")
    add_data_root(p)
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(func=cmd_build)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
