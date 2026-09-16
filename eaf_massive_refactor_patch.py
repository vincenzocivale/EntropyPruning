#!/usr/bin/env python3
"""Massive cleanup/refactor patch for EntropyPruning.

Base audited against commit f541e9d8e88aa06e112c2bb23a88646403536658.
Run from the repository root on a dedicated branch.

The patch deliberately removes historical/side pipelines and keeps one paper-driven path:
HISTAI -> Tile-EAF -> tile embedding distillation -> WSI teacher/source caches ->
WSI-EAF -> TITAN distillation -> THUNDER/EAGLE evaluation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

BASE_COMMIT = "f541e9d8e88aa06e112c2bb23a88646403536658"

REMOVE_PATHS = [
    # old planning / oversized experimental program
    "docs/continuity.md",
    "docs/experimental_protocols.md",
    "docs/experimental_roadmap.md",
    "docs/experimental_runbook.md",

    # side analyses that are not part of the final protocol
    "scripts/analysis/analyze_wsi_attention_embeddings.py",
    "scripts/analysis/wsi_run_attention_signal_discovery.py",
    "scripts/evaluation/eval_tile_eaf_paired_downstream.py",
    "scripts/evaluation/evaluate_wsi_coarsened_store_agreement.py",
    "scripts/features/create_wsi_morphology_coarsened_store.py",

    # old supervised THUNDER / multi-head pipeline
    "scripts/training/train_multi_thunder_classifier.py",

    # old TCGA / HEST / multisource pretraining machinery
    "scripts/data/audit_materialize_hest_eaf.py",
    "scripts/data/build_eaf_multisource_manifest.py",
    "scripts/data/build_hest_eaf_inventory.py",
    "scripts/data/build_tcga_thunder_clean_inventory.py",
    "scripts/data/download_hest_eaf_wsis.py",
    "scripts/data/download_tcga_thunder_clean.py",
    "scripts/data/manage_unlabeled_wsi.py",

    # competing pruning baselines (explicitly removed)
    "src/baselines",

    # old standalone analysis / collection modules
    "src/analysis",
    "src/collection",

    # superseded generic image-classification data stack
    "src/data/dataset.py",
    "src/data/h5_dataset.py",
    "src/data/loaders.py",
    "src/data/transforms.py",
    "src/data/thunder_multi.py",
    "src/data/wsi/morphology_coarsening.py",

    # old downstream classifiers / alternate pruning wrappers
    "src/models/classifier.py",
    "src/models/generic_pruned_classifier.py",
    "src/models/multi_head_classifier.py",
    "src/models/pruned_classifier.py",
    "src/models/thunder_classifier.py",
    "src/models/wsi/landmark_forecaster.py",

    # duplicate/obsolete attention-distillation implementation
    "src/training/online_attention_distillation.py",

    # side pipelines
    "src/wsi_pipeline/acquisition.py",
    "src/wsi_pipeline/attention_signal.py",
    "src/wsi_pipeline/experiment_catalog.py",
    "src/wsi_pipeline/majority_analysis.py",
    "src/wsi_pipeline/offline_cache.py",

    # tests tied only to removed code
    "tests/analysis",
    "tests/data/test_thunder_multi.py",
    "tests/data/wsi/test_corpora.py",
    "tests/data/wsi/test_gtex_siblings.py",
    "tests/data/wsi/test_morphology_coarsening.py",
    "tests/scripts/test_analyze_wsi_attention_embeddings.py",
    "tests/training/test_online_attention_distillation.py",
    "tests/wsi_pipeline/test_acquisition.py",
    "tests/wsi_pipeline/test_attention_signal.py",
    "tests/wsi_pipeline/test_experiment_catalog.py",
    "tests/wsi_pipeline/test_majority_analysis.py",
]

RENAMES = {
    "scripts/training/train_wsi_tile_eaf_online.py": "scripts/training/train_tile_eaf.py",
    "scripts/training/finetune_wsi_tile_encoder_pruned_online.py": "scripts/training/distill_tile_encoder.py",
    "scripts/training/train_wsi_landmark_forecaster.py": "scripts/training/train_wsi_eaf.py",
    "scripts/training/finetune_wsi_titan_pruned.py": "scripts/training/distill_wsi_titan.py",
    "scripts/features/wsi_eaf_infer_wsi_fm.py": "scripts/features/cache_wsi_teacher.py",
    "scripts/evaluation/eval_wsi_linear_probing.py": "scripts/evaluation/evaluate_wsi_eagle.py",
}

REFERENCE_REPLACEMENTS = {
    "train_wsi_tile_eaf_online.py": "train_tile_eaf.py",
    "finetune_wsi_tile_encoder_pruned_online.py": "distill_tile_encoder.py",
    "train_wsi_landmark_forecaster.py": "train_wsi_eaf.py",
    "finetune_wsi_titan_pruned.py": "distill_wsi_titan.py",
    "wsi_eaf_infer_wsi_fm.py": "cache_wsi_teacher.py",
    "eval_wsi_linear_probing.py": "evaluate_wsi_eagle.py",
}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def git_root() -> Path:
    try:
        root = run("git", "rev-parse", "--show-toplevel").stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Run this patch from inside the EntropyPruning git repository") from exc
    return Path(root)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def remove(root: Path, rel: str) -> None:
    path = root / rel
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def rename(root: Path, src: str, dst: str) -> None:
    source, target = root / src, root / dst
    if not source.exists():
        if target.exists():
            return
        raise RuntimeError(f"Expected source for rename is missing: {src}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"Rename destination already exists: {dst}")
    source.rename(target)


def replace_all_text_references(root: Path) -> None:
    for path in list(root.rglob("*.py")) + list(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in REFERENCE_REPLACEMENTS.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence in {path}: {old!r}; found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def rewrite_corpora(root: Path) -> None:
    write(root / "src/data/wsi/corpora.py", r'''
        """HISTAI-only acquisition for task-agnostic EAF training."""

        from __future__ import annotations

        import csv
        import json
        import os
        import re
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from pathlib import Path
        from typing import Iterable

        from .layout import DatasetRole, StoreLayout
        from .manifest import SlideRecord, write_manifest

        HISTAI_DATASET = "histai_eaf_wsi_v1"

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


        def _write_rows(path: Path, rows: list[dict[str, str]]) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not rows:
                raise ValueError("Cannot write an empty HISTAI plan")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return path


        def _read_rows(path: Path) -> list[dict[str, str]]:
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))


        def _ensure_symlink(link_path: Path, target: Path) -> None:
            target = target.resolve()
            if link_path.is_symlink() or link_path.exists():
                if link_path.resolve() == target:
                    return
                link_path.unlink()
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(target)


        def _histai_case_id(repo_path: str) -> str | None:
            return next((part for part in Path(repo_path).parts if part.startswith("case_")), None)


        def _is_histai_he(repo_path: str) -> bool:
            return bool(
                re.match(
                    r"^slide_(?:[^_]+_)?H&E_\d+\.tiff$",
                    Path(repo_path).name,
                    flags=re.IGNORECASE,
                )
            )


        def _histai_magnification(repo_path: str) -> str:
            match = re.search(r"slide_([^_]+)_h&e_\d+\.tiff$", Path(repo_path).name.lower())
            return match.group(1) if match else ""


        def _histai_rank(repo_path: str) -> tuple[int, str]:
            mag = _histai_magnification(repo_path)
            if mag in {"20x", "x20", "20"}:
                return (0, repo_path)
            if not mag:
                return (1, repo_path)
            return (2, repo_path)


        def plan_histai(
            data_root: str | Path,
            *,
            token: str | None = None,
            subsets: Iterable[str] | None = None,
            force: bool = False,
        ) -> Path:
            """Freeze one deterministic H&E WSI per HISTAI case."""
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError("Install huggingface_hub to plan HISTAI") from exc

            layout = StoreLayout.from_root(data_root)
            layout.ensure_base_dirs()
            chosen = tuple(subsets) if subsets else HISTAI_SUBSETS
            invalid = sorted(set(chosen) - set(HISTAI_SUBSETS))
            if invalid:
                raise ValueError(f"Unknown HISTAI subsets: {invalid}")

            dataset_dir = layout.dataset_dir(DatasetRole.PRETRAINING, HISTAI_DATASET)
            plan = dataset_dir / "manifests" / "plan.csv"
            access_path = dataset_dir / "manifests" / "histai_subset_access.json"
            existing_rows = _read_rows(plan) if plan.exists() else []
            existing_subsets = {row["subset"] for row in existing_rows}
            to_list = list(chosen) if force else [s for s in chosen if s not in existing_subsets]
            if not to_list:
                return plan

            api = HfApi(token=token or os.environ.get("HF_TOKEN"))
            access = json.loads(access_path.read_text()) if access_path.exists() else {}
            kept_rows = [row for row in existing_rows if row["subset"] not in to_list]
            new_rows: list[dict[str, str]] = []
            for subset in to_list:
                repo_id = f"histai/{subset}"
                try:
                    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                except Exception as exc:
                    access[subset] = {"accessible": False, "error": repr(exc)}
                    print(f"[eaf-data] {subset}: inaccessible ({exc}); skipped", flush=True)
                    continue
                by_case: dict[str, list[str]] = defaultdict(list)
                for repo_path in files:
                    case_id = _histai_case_id(repo_path)
                    if case_id and _is_histai_he(repo_path):
                        by_case[case_id].append(repo_path)
                for case_id in sorted(by_case):
                    repo_path = min(by_case[case_id], key=_histai_rank)
                    new_rows.append(
                        {
                            "source": "histai",
                            "subset": subset,
                            "repo_id": repo_id,
                            "case_id": case_id,
                            "repo_path": repo_path,
                            "native_magnification": _histai_magnification(repo_path),
                        }
                    )
                access[subset] = {"accessible": True, "n_cases": len(by_case)}
                print(f"[eaf-data] {subset}: {len(by_case)} H&E cases selected", flush=True)

            access_path.parent.mkdir(parents=True, exist_ok=True)
            access_path.write_text(json.dumps(access, indent=2, sort_keys=True) + "\n")
            rows = kept_rows + new_rows
            if not rows:
                raise RuntimeError(f"No HISTAI subset was accessible; see {access_path}")
            return _write_rows(plan, rows)


        def download_histai(
            data_root: str | Path,
            *,
            subsets: Iterable[str] | None = None,
            workers: int = 4,
            token: str | None = None,
        ) -> Path:
            """Incrementally download only the H&E WSI frozen by ``plan_histai``."""
            try:
                from huggingface_hub import hf_hub_download
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise RuntimeError("Install huggingface_hub, hf_xet and tqdm") from exc

            layout = StoreLayout.from_root(data_root)
            dataset_dir = layout.dataset_dir(DatasetRole.PRETRAINING, HISTAI_DATASET)
            plan = dataset_dir / "manifests" / "plan.csv"
            if not plan.exists():
                raise FileNotFoundError(f"Missing HISTAI plan: {plan}")
            all_rows = _read_rows(plan)
            requested = set(subsets or ())
            rows = [row for row in all_rows if not requested or row["subset"] in requested]
            if not rows:
                raise ValueError("No HISTAI rows selected")

            auth = token or os.environ.get("HF_TOKEN")
            failures: list[str] = []

            def download_one(row: dict[str, str]) -> None:
                local_dir = layout.sources / "histai" / row["subset"]
                expected = local_dir / row["repo_path"]
                if expected.is_file():
                    return
                local_dir.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=row["repo_id"],
                    repo_type="dataset",
                    filename=row["repo_path"],
                    local_dir=local_dir,
                    token=auth,
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_rows = {pool.submit(download_one, row): row for row in rows}
                for future in tqdm(as_completed(future_rows), total=len(future_rows), desc="HISTAI", unit="WSI"):
                    row = future_rows[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(f"{row['repo_id']}:{row['repo_path']}::{exc}")

            views_raw_flat = dataset_dir / "views" / "raw_flat"
            records: list[SlideRecord] = []
            for row in all_rows:
                local = layout.sources / "histai" / row["subset"] / row["repo_path"]
                slide_stem = Path(row["repo_path"]).stem
                slide_id = f"{row['subset']}__{row['case_id']}__{slide_stem}"
                exists = local.is_file()
                if exists:
                    _ensure_symlink(views_raw_flat / f"{slide_id}{local.suffix}", local)
                records.append(
                    SlideRecord(
                        slide_id=slide_id,
                        case_id=f"{row['subset']}::{row['case_id']}",
                        source="histai",
                        cohort=row["subset"],
                        subset=row["subset"],
                        raw_path=str(local),
                        downloaded="1" if exists else "0",
                        metadata_json=(
                            '{"repo_id":"%s","repo_path":"%s","native_magnification":"%s"}'
                            % (row["repo_id"], row["repo_path"], row["native_magnification"])
                        ),
                    )
                )
            manifest = write_manifest(dataset_dir / "manifests" / "slides.csv", records)
            if failures:
                failure_path = dataset_dir / "manifests" / "download_failures.txt"
                failure_path.write_text("\n".join(failures) + "\n")
                print(f"[eaf-data] HISTAI failures: {len(failures)} -> {failure_path}")
            return manifest
    ''')


def rewrite_eaf_cli(root: Path) -> None:
    path = root / "scripts/eaf.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"from src\.data\.wsi\.corpora import \(.*?\n\)",
        "from src.data.wsi.corpora import HISTAI_SUBSETS, download_histai, plan_histai",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace("from src.wsi_pipeline.experiment_catalog import audit_experiments, write_catalog  # noqa: E402\n", "")
    text = re.sub(
        r"\n\ndef cmd_experiments_audit\(.*?\n\ndef cmd_plan_histai",
        "\n\ndef cmd_plan_histai",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"\n\ndef cmd_plan_gtex\(.*?\n\ndef cmd_validate_cache",
        "\n\ndef cmd_validate_cache",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"\n    experiments = sub\.add_parser\(\"experiments\".*?p\.set_defaults\(func=cmd_experiments_audit\)\n",
        "\n",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"\n    p = data_sub\.add_parser\(\"plan-gtex\"\).*?p\.set_defaults\(func=cmd_build_strict\)\n",
        "\n",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace('data = sub.add_parser("data", help="Plan/download pretraining corpora")',
                        'data = sub.add_parser("data", help="Plan/download the unlabeled HISTAI training corpus")')
    text = text.replace("Canonical strict slides.csv", "HISTAI slides.csv")
    text = text.replace("the same path scripts/training/train_tile_eaf.py and distill_tile_encoder.py use", "the same model registry used by the Tile-EAF training/distillation scripts")
    path.write_text(text, encoding="utf-8")


def rewrite_models_init(root: Path) -> None:
    write(root / "src/models/__init__.py", '''
        """Core EAF model components."""

        from .backbone_adapter import ThunderBackboneAdapter
        from .forecaster import AttentionForecaster

        __all__ = ["AttentionForecaster", "ThunderBackboneAdapter"]
    ''')
    write(root / "src/data/__init__.py", '''
        """Data utilities for HISTAI training and THUNDER/EAGLE evaluation."""
    ''')


def rewrite_thunder_loaders(root: Path) -> None:
    write(root / "src/data/thunder_loaders.py", r'''
        """Thin THUNDER benchmark loader used only for downstream evaluation."""

        from __future__ import annotations

        from pathlib import Path

        import h5py
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

        import thunder
        from thunder.utils.data import PatchDataset, get_data


        class _TupleDataset(Dataset):
            def __init__(self, patch_ds):
                self._ds = patch_ds

            def __len__(self):
                return len(self._ds)

            def __getitem__(self, idx):
                item = self._ds[idx]
                label = np.asarray(item["label"]).reshape(()).item()
                return item["image"], int(label)


        def discover_thunder_datasets(base_data_folder: str | Path) -> list[str]:
            splits = Path(base_data_folder) / "data_splits"
            if not splits.is_dir():
                raise FileNotFoundError(f"THUNDER data_splits/ not found: {splits}")
            return sorted(path.stem for path in splits.glob("*.json"))


        def build_thunder_loaders(
            dataset_name: str,
            base_data_folder: str,
            transform,
            batch_size: int = 32,
            num_workers: int = 4,
            *,
            balanced_train: bool = False,
            drop_last_train: bool = False,
        ):
            """Return deterministic train/val/test loaders for one THUNDER dataset.

            ``balanced_train=False`` is the evaluation default: every original training
            example is embedded exactly once. Class balancing belongs in the downstream
            linear head, not in feature extraction.
            """
            split_path = Path(base_data_folder) / "data_splits" / f"{dataset_name}.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Data split not found: {split_path}")

            data = get_data(dataset_name, base_data_folder)
            cfg_path = Path(thunder.__file__).parent / "config" / "dataset" / f"{dataset_name}.yaml"
            h5_format = False
            if cfg_path.exists():
                cfg = OmegaConf.load(cfg_path)
                class_names = list(cfg.classes)
                n_classes = int(cfg.nb_classes)
                h5_format = bool(getattr(cfg, "h5_format", False))
            else:
                labels = np.asarray(data["train"]["labels"]).reshape(-1).astype(int)
                n_classes = int(labels.max()) + 1
                class_names = [f"class_{i}" for i in range(n_classes)]

            def make(split: str) -> _TupleDataset:
                return _TupleDataset(
                    PatchDataset(
                        images=data[split]["images"],
                        labels=data[split]["labels"],
                        transform=transform,
                        task_type="linear_probing",
                        dataset_name=dataset_name,
                        base_data_folder=base_data_folder,
                        embeddings_folder=None,
                        image_pre_loading=False,
                        embedding_pre_loading=False,
                        div_patches=False,
                        h5_format=h5_format,
                    )
                )

            train_ds, val_ds, test_ds = make("train"), make("val"), make("test")
            kwargs = dict(
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )

            if balanced_train:
                if h5_format:
                    labels_path = Path(base_data_folder) / dataset_name / str(data["train"]["labels"])
                    with h5py.File(labels_path, "r") as handle:
                        train_labels = np.asarray(handle["y"]).reshape(-1).astype(int)
                else:
                    train_labels = np.asarray(data["train"]["labels"]).reshape(-1).astype(int)
                counts = np.bincount(train_labels)
                weights = torch.from_numpy((1.0 / counts)[train_labels]).double()
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                train_loader = DataLoader(train_ds, sampler=sampler, drop_last=drop_last_train, **kwargs)
            else:
                train_loader = DataLoader(train_ds, shuffle=False, drop_last=False, **kwargs)

            return (
                train_loader,
                DataLoader(val_ds, shuffle=False, **kwargs),
                DataLoader(test_ds, shuffle=False, **kwargs),
                class_names,
                n_classes,
            )
    ''')


def rewrite_splits(root: Path) -> None:
    write(root / "src/data/wsi/splits.py", r'''
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
    ''')


def rewrite_wsi_forecaster_dataset(root: Path) -> None:
    write(root / "src/data/wsi/wsi_forecaster_dataset.py", r'''
        """Dataset contract for WSI-EAF with explicit student-source / full-teacher separation.

        The final paper pipeline uses three aligned caches per slide:

        * ``tile_input_root``: tile embeddings given to the WSI student. For the final
          pipeline these come from the distilled Tile-EAF-pruned tile encoder.
        * ``source_wsi_root``: TITAN run on ``tile_input_root`` with the requested
          intermediate hidden layer cached. WSI-EAF reads its source representation here.
        * ``teacher_wsi_root``: frozen full TITAN run on the *full* tile encoder output.
          Final attention and slide embedding targets always come from this root.

        Keeping these paths distinct prevents accidental pruned->pruned distillation.
        """

        from __future__ import annotations

        import csv
        from dataclasses import dataclass
        from pathlib import Path

        import pandas as pd
        import torch
        from torch.utils.data import Dataset

        from src.data.wsi.attention import ManifestAttentionSource, align_attention_to_bag
        from src.data.wsi.bag import WSIBag
        from src.data.wsi.splits import assign_case_splits
        from src.wsi_pipeline.numpy_store import array_names, read_array


        @dataclass(frozen=True)
        class WSIForecasterManifestConfig:
            tile_input_root: Path
            teacher_wsi_root: Path
            source_wsi_root: Path | None = None
            attention_key: str = "attention/global_to_tiles_mass_share"
            target_layer: int = -1
            cohorts: tuple[str, ...] | None = None
            exclude_cohorts: tuple[str, ...] | None = None
            hidden_layer: int | None = None


        def _files_by_stem(directory: Path) -> dict[str, Path]:
            values = {path.stem: path for path in directory.glob("*.h5")}
            values.update({path.stem: path for path in directory.glob("*.npyd")})
            return values


        def build_manifest(config: WSIForecasterManifestConfig) -> pd.DataFrame:
            rows: list[dict[str, str]] = []
            source_root = config.source_wsi_root or config.teacher_wsi_root
            exclude = set(config.exclude_cohorts or ())
            teacher_cohorts = sorted(
                path for path in config.teacher_wsi_root.iterdir()
                if path.is_dir()
                and (config.cohorts is None or path.name in config.cohorts)
                and path.name not in exclude
            )
            for teacher_dir in teacher_cohorts:
                cohort = teacher_dir.name
                tile_dir = config.tile_input_root / cohort
                source_dir = source_root / cohort
                if not tile_dir.is_dir() or not source_dir.is_dir():
                    continue
                teacher = _files_by_stem(teacher_dir)
                source = _files_by_stem(source_dir)
                tiles = _files_by_stem(tile_dir)
                for stem in sorted(set(teacher) & set(source) & set(tiles)):
                    rows.append(
                        {
                            "slide_id": stem,
                            "case_id": stem,
                            "project": cohort,
                            "tile_path": str(tiles[stem]),
                            "source_wsi_path": str(source[stem]),
                            "teacher_wsi_path": str(teacher[stem]),
                        }
                    )
            if not rows:
                raise RuntimeError(
                    "No slide is shared by tile_input_root, source_wsi_root and teacher_wsi_root"
                )
            table = pd.DataFrame(rows)
            table["case_id"] = (
                table["slide_id"]
                .str.extract(r"^(.*__case_[^_]+)__", expand=False)
                .fillna(table["slide_id"])
            )
            return table


        def write_manifest_csv(table: pd.DataFrame, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)


        def build_attention_manifest_csv(table: pd.DataFrame, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["slide_id", "attention_path", "coords_path"]
                )
                writer.writeheader()
                for row in table[["slide_id", "teacher_wsi_path"]].itertuples(index=False):
                    writer.writerow(
                        {
                            "slide_id": row.slide_id,
                            "attention_path": row.teacher_wsi_path,
                            "coords_path": row.teacher_wsi_path,
                        }
                    )


        def assign_splits(
            table: pd.DataFrame,
            *,
            train_fraction: float = 0.70,
            validation_fraction: float = 0.15,
            seed: int = 17,
        ) -> pd.DataFrame:
            return assign_case_splits(
                table,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                seed=seed,
            )


        def _load_tile_bag(slide_id: str, path: Path) -> WSIBag:
            tile_features = torch.from_numpy(read_array(path, "tile_embeddings")).float()
            coords = (
                torch.from_numpy(read_array(path, "coords")).long()
                if "coords" in array_names(path)
                else None
            )
            return WSIBag(slide_id=slide_id, tile_features=tile_features, coords=coords)


        def _load_titan_hidden_bag(slide_id: str, path: Path, layer: int) -> WSIBag:
            key = f"auxiliary/hidden_layer_{layer:03d}"
            names = array_names(path)
            if key not in names:
                available = sorted(name for name in names if name.startswith("auxiliary/"))
                raise KeyError(f"{key!r} not found in {path}; available={available}")
            tile_features = torch.from_numpy(read_array(path, key)).float()
            if "coords" not in names:
                raise KeyError(f"{path} has no coords; hidden-layer alignment is impossible")
            coords = torch.from_numpy(read_array(path, "coords")).long()
            return WSIBag(slide_id=slide_id, tile_features=tile_features, coords=coords)


        class WSIForecasterDataset(Dataset):
            def __init__(
                self,
                manifest: pd.DataFrame,
                *,
                attention_manifest_path: Path,
                config: WSIForecasterManifestConfig,
                split: str,
            ) -> None:
                if "split" not in manifest.columns:
                    raise ValueError("manifest is missing split; call assign_splits first")
                self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
                if self.rows.empty:
                    raise ValueError(f"No slides assigned to split={split!r}")
                self.config = config
                self.attention_source = ManifestAttentionSource(
                    attention_manifest_path,
                    attention_key=config.attention_key,
                    coords_key="coords",
                    tile_axis=2,
                    reduction="mean",
                    selections={0: config.target_layer},
                )

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                row = self.rows.iloc[index]
                if self.config.hidden_layer is not None:
                    bag = _load_titan_hidden_bag(
                        row.slide_id, Path(row.source_wsi_path), self.config.hidden_layer
                    )
                else:
                    bag = _load_tile_bag(row.slide_id, Path(row.tile_path))
                attention = self.attention_source.read(row.slide_id, n_tiles=bag.n_tiles)
                bag, attention = align_attention_to_bag(bag, attention, mode="coords")
                target = attention.values.clamp_min(0.0)
                target = target / target.sum().clamp_min(1e-8)
                coords = bag.coords if bag.coords is not None else torch.zeros(
                    (bag.n_tiles, 2), dtype=torch.long
                )
                return bag.tile_features, coords, target, row.slide_id
    ''')

    write(root / "src/data/wsi/wsi_pruned_titan_dataset.py", r'''
        """WSI student inputs from the pruned tile encoder, targets from full TITAN."""

        from __future__ import annotations

        from pathlib import Path

        import pandas as pd
        import torch
        from torch.utils.data import Dataset

        from src.wsi_pipeline.numpy_store import read_array


        def _load_tile_bag(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.from_numpy(read_array(path, "tile_embeddings")).float(),
                torch.from_numpy(read_array(path, "coords")).long(),
            )


        def _load_teacher_embedding(path: Path) -> torch.Tensor:
            embedding = torch.from_numpy(read_array(path, "slide_embedding")).float()
            while embedding.dim() > 1 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            return embedding


        class WSIPrunedTitanDataset(Dataset):
            def __init__(self, manifest: pd.DataFrame, *, split: str) -> None:
                if "split" not in manifest.columns:
                    raise ValueError("manifest is missing split; call assign_splits first")
                self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
                if self.rows.empty:
                    raise ValueError(f"No slides assigned to split={split!r}")

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                row = self.rows.iloc[index]
                tile_features, coords = _load_tile_bag(Path(row.tile_path))
                teacher_embedding = _load_teacher_embedding(Path(row.teacher_wsi_path))
                if len(tile_features) != len(coords):
                    raise ValueError(
                        f"slide {row.slide_id}: tile/coord mismatch {len(tile_features)} != {len(coords)}"
                    )
                return tile_features, coords, teacher_embedding, row.slide_id
    ''')


def patch_wsi_training_scripts(root: Path) -> None:
    train = root / "scripts/training/train_wsi_eaf.py"
    text = train.read_text(encoding="utf-8")
    text = text.replace('parser.add_argument("--tile-eaf-root", type=Path, required=True)\n    parser.add_argument("--wsi-eaf-root", type=Path, required=True)',
        'parser.add_argument("--tile-input-root", type=Path, required=True, help="Tile embeddings used by the WSI student; final runs use the distilled Tile-EAF-pruned encoder cache")\n    parser.add_argument("--source-wsi-root", type=Path, required=True, help="WSI cache built from --tile-input-root; intermediate hidden states are read here")\n    parser.add_argument("--teacher-wsi-root", type=Path, required=True, help="Frozen full WSI teacher cache built from the unpruned tile encoder; attention targets are read here")')
    text = text.replace("args.tile_eaf_root", "args.tile_input_root")
    text = text.replace("args.wsi_eaf_root", "args.teacher_wsi_root")
    text = text.replace("tile_eaf_root=args.tile_input_root,\n        wsi_eaf_root=args.teacher_wsi_root,",
                        "tile_input_root=args.tile_input_root,\n        source_wsi_root=args.source_wsi_root,\n        teacher_wsi_root=args.teacher_wsi_root,")
    text = text.replace("matches the correlational-analysis splits", "case-disjoint HISTAI split seed")
    text = text.replace("WSI-EAF attention-signal investigation notes for why\nfinal-layer embeddings (not early-layer) and learned weights (not raw content\nsimilarity) are both required ingredients.\n\n", "")
    train.write_text(text, encoding="utf-8")

    distill = root / "scripts/training/distill_wsi_titan.py"
    text = distill.read_text(encoding="utf-8")
    text = text.replace('parser.add_argument("--tile-eaf-root", type=Path, required=True)\n    parser.add_argument("--wsi-eaf-root", type=Path, required=True)',
        'parser.add_argument("--tile-input-root", type=Path, required=True, help="Tile embeddings from the distilled Tile-EAF-pruned encoder")\n    parser.add_argument("--teacher-wsi-root", type=Path, required=True, help="Full TITAN teacher cache built from full tile embeddings")')
    text = text.replace("args.tile_eaf_root", "args.tile_input_root")
    text = text.replace("args.wsi_eaf_root", "args.teacher_wsi_root")
    text = text.replace("tile_eaf_root=args.tile_input_root,\n        wsi_eaf_root=args.teacher_wsi_root,",
                        "tile_input_root=args.tile_input_root,\n        teacher_wsi_root=args.teacher_wsi_root,")
    text = text.replace("point at a pruned-input cache", "identify the pruned tile-input cache")
    distill.write_text(text, encoding="utf-8")

    evaluate = root / "scripts/evaluation/evaluate_wsi_eagle.py"
    text = evaluate.read_text(encoding="utf-8")
    text = text.replace("--wsi-eaf-root", "--teacher-wsi-root")
    text = text.replace("--tile-eaf-root", "--tile-input-root")
    text = text.replace("args.wsi_eaf_root", "args.teacher_wsi_root")
    text = text.replace("args.tile_eaf_root", "args.tile_input_root")
    text = text.replace("wsi_eaf_root", "teacher_wsi_root")
    text = text.replace("tile_eaf_root", "tile_input_root")
    evaluate.write_text(text, encoding="utf-8")


def write_tile_thunder_evaluation(root: Path) -> None:
    write(root / "scripts/evaluation/evaluate_tile_thunder.py", r'''
        #!/usr/bin/env python3
        """Evaluate the same tile encoder full vs EAF-pruned on THUNDER datasets.

        EAF weights and LoRA distillation weights are frozen. Only a linear logistic
        regression head is fitted downstream, with the same official THUNDER splits
        for both representations. This script never trains EAF on THUNDER labels.
        """

        from __future__ import annotations

        import argparse
        import csv
        import json
        import sys
        import time
        from pathlib import Path

        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

        from thunder.models.pretrained_models import get_model_from_name
        from src.data.thunder_loaders import build_thunder_loaders, discover_thunder_datasets
        from src.models import AttentionForecaster, ThunderBackboneAdapter
        from src.models.online_tile_eaf import PrunedLoRAEncoder, unwrap_checkpoint_state
        from src.utils import set_seed


        def _autocast(device: torch.device, dtype: str):
            amp = torch.bfloat16 if dtype == "bf16" else torch.float16
            return torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda")


        def _load_student(args, device: torch.device):
            payload = torch.load(args.pruned_checkpoint, map_location="cpu", weights_only=False)
            config = payload.get("config", {})
            model_name = args.model_name or config.get("model_name") or payload.get("base_model")
            if not model_name:
                raise ValueError("Could not resolve tile model name from CLI/checkpoint")
            raw, transform, _ = get_model_from_name(model_name, str(device))
            raw = raw.to(device)
            adapter = ThunderBackboneAdapter(raw, transform=transform)

            forecaster_path = args.forecaster_checkpoint or payload.get("forecaster_checkpoint")
            if not forecaster_path:
                raise ValueError("Could not resolve forecaster checkpoint")
            fpayload = torch.load(forecaster_path, map_location="cpu", weights_only=False)
            fcfg = fpayload.get("config", fpayload.get("args", {}))
            forecaster = AttentionForecaster(
                embed_dim=adapter.embed_dim,
                hidden=int(config.get("hidden", fcfg.get("hidden", 256))),
                n_heads=int(config.get("n_heads", fcfg.get("n_heads", 4))),
                n_layers=int(config.get("n_layers", fcfg.get("n_layers", 2))),
                dropout=0.0,
            )
            forecaster.load_state_dict(unwrap_checkpoint_state(fpayload), strict=True)
            forecaster = forecaster.to(device).eval()

            student = PrunedLoRAEncoder(
                raw,
                adapter,
                forecaster,
                prune_layer=int(config["prune_layer"]),
                keep_ratio=float(config["keep_ratio"]),
                lora_r=int(config.get("lora_r", 8)),
                lora_alpha=int(config.get("lora_alpha", 32)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
            ).to(device)
            student.load_trainable_state_dict(payload["trainable_state_dict"])
            student.eval()
            return student, transform, model_name, config


        @torch.inference_mode()
        def extract_pair(loader, student, *, device, amp_dtype):
            full, pruned, labels = [], [], []
            full_seconds = 0.0
            pruned_seconds = 0.0
            for images, y in loader:
                images = images.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with _autocast(device, amp_dtype):
                    full_embedding = student.full_teacher_embedding(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                full_seconds += time.perf_counter() - start

                start = time.perf_counter()
                with _autocast(device, amp_dtype):
                    pruned_embedding = student(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                pruned_seconds += time.perf_counter() - start

                full.append(full_embedding.float().cpu().numpy())
                pruned.append(pruned_embedding.float().cpu().numpy())
                labels.append(y.numpy())
            return (
                np.concatenate(full),
                np.concatenate(pruned),
                np.concatenate(labels).astype(int),
                full_seconds,
                pruned_seconds,
            )


        def choose_c(X_train, y_train, X_val, y_val, c_grid):
            scaler = StandardScaler().fit(X_train)
            train = scaler.transform(X_train)
            val = scaler.transform(X_val)
            best = None
            for c in c_grid:
                clf = LogisticRegression(
                    C=c, max_iter=3000, class_weight="balanced", multi_class="auto"
                )
                clf.fit(train, y_train)
                score = balanced_accuracy_score(y_val, clf.predict(val))
                candidate = (float(score), -float(c), float(c))
                if best is None or candidate > best:
                    best = candidate
            return best[2]


        def fit_and_score(X_train, y_train, X_val, y_val, X_test, y_test, c_grid):
            c = choose_c(X_train, y_train, X_val, y_val, c_grid)
            X_fit = np.concatenate([X_train, X_val])
            y_fit = np.concatenate([y_train, y_val])
            scaler = StandardScaler().fit(X_fit)
            clf = LogisticRegression(
                C=c, max_iter=3000, class_weight="balanced", multi_class="auto"
            )
            clf.fit(scaler.transform(X_fit), y_fit)
            X_test_s = scaler.transform(X_test)
            pred = clf.predict(X_test_s)
            proba = clf.predict_proba(X_test_s)
            result = {
                "C": c,
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
            classes = np.unique(y_test)
            if len(classes) == 2:
                result["auroc"] = float(roc_auc_score(y_test, proba[:, 1]))
            elif len(classes) > 2:
                try:
                    result["auroc_ovr_macro"] = float(
                        roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                    )
                except ValueError:
                    pass
            return result


        def main() -> int:
            parser = argparse.ArgumentParser(description=__doc__)
            parser.add_argument("--base-data-folder", required=True)
            parser.add_argument("--dataset", action="append", help="Repeatable; omit to evaluate every installed THUNDER dataset")
            parser.add_argument("--pruned-checkpoint", type=Path, required=True)
            parser.add_argument("--forecaster-checkpoint", type=Path)
            parser.add_argument("--model-name", help="Override model name stored in the distillation checkpoint")
            parser.add_argument("--batch-size", type=int, default=64)
            parser.add_argument("--num-workers", type=int, default=8)
            parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
            parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--output", type=Path, default=Path("results/tile_thunder.csv"))
            args = parser.parse_args()

            set_seed(args.seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            student, transform, model_name, config = _load_student(args, device)
            datasets = args.dataset or discover_thunder_datasets(args.base_data_folder)
            rows = []

            for dataset in datasets:
                print(f"[THUNDER] {dataset}", flush=True)
                train_loader, val_loader, test_loader, class_names, n_classes = build_thunder_loaders(
                    dataset,
                    args.base_data_folder,
                    transform,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    balanced_train=False,
                )
                split_values = {}
                timing = {"full": 0.0, "pruned": 0.0}
                for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
                    full, pruned, y, full_s, pruned_s = extract_pair(
                        loader, student, device=device, amp_dtype=args.amp_dtype
                    )
                    split_values[split] = {"full": full, "pruned": pruned, "y": y}
                    timing["full"] += full_s
                    timing["pruned"] += pruned_s

                for arm in ("full", "pruned"):
                    metrics = fit_and_score(
                        split_values["train"][arm], split_values["train"]["y"],
                        split_values["val"][arm], split_values["val"]["y"],
                        split_values["test"][arm], split_values["test"]["y"],
                        args.c_grid,
                    )
                    n_images = sum(len(split_values[s]["y"]) for s in ("train", "val", "test"))
                    rows.append(
                        {
                            "dataset": dataset,
                            "arm": arm,
                            "model": model_name,
                            "n_classes": n_classes,
                            "class_names": json.dumps(class_names),
                            "keep_ratio": 1.0 if arm == "full" else float(config["keep_ratio"]),
                            "prune_layer": "" if arm == "full" else int(config["prune_layer"]),
                            "embedding_seconds": timing[arm],
                            "images_per_second": n_images / max(timing[arm], 1e-9),
                            **metrics,
                        }
                    )

            args.output.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted({key for row in rows for key in row})
            with args.output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote {len(rows)} rows -> {args.output}")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
    ''')


def rewrite_docs(root: Path) -> None:
    write(root / "README.md", r'''
        # EAF

        This repository now contains one paper-driven pipeline only: train EAF without
        downstream labels on HISTAI, distill pruned tile/WSI encoders to reproduce their
        unpruned embeddings, and evaluate frozen representations on THUNDER and the public
        WSI tasks used by EAGLE.

        ## Scientific pipeline

        ```text
        HISTAI WSI (unlabeled)
          |
          +-- full tile teacher cache
          |     |
          |     +-- train Tile-EAF: early patch tokens -> final teacher attention
          |     |
          |     +-- distill pruned tile encoder -> full tile embedding
          |             |
          |             +-- pruned tile-input cache
          |
          +-- full TITAN teacher cache ------------------------------+
          |                                                         |
          +-- TITAN source cache built from pruned tile inputs      |
                |                                                    |
                +-- train WSI-EAF: intermediate TITAN state          |
                |                 -> FULL teacher attention           |
                |                                                    |
                +-- distill pruned TITAN + pruned tile inputs -------+
                                   -> FULL teacher slide embedding
        ```

        The separation between `source_wsi_root` and `teacher_wsi_root` is an invariant:
        WSI-EAF may observe hidden states produced from pruned tile inputs, but its target
        remains the full pipeline. This prevents accidental pruned-to-pruned distillation.

        ## Supported entry points

        Data/cache operations:

        ```bash
        python scripts/eaf.py data plan-histai --data-root "$EAF_WSI_ROOT"
        python scripts/eaf.py data download-histai --data-root "$EAF_WSI_ROOT"
        python scripts/eaf.py cache tile --help
        ```

        Training:

        ```text
        scripts/training/train_tile_eaf.py
        scripts/training/distill_tile_encoder.py
        scripts/training/train_wsi_eaf.py
        scripts/training/distill_wsi_titan.py
        ```

        Teacher/source cache creation:

        ```text
        scripts/features/cache_wsi_teacher.py
        ```

        Evaluation:

        ```text
        scripts/evaluation/evaluate_tile_thunder.py
        scripts/evaluation/evaluate_wsi_eagle.py
        ```

        Read `docs/pipeline.md` for the exact cache contracts and launch order.

        ## Scope

        Historical pruning baselines, supervised THUNDER pretraining, TCGA/HEST/GTEx EAF
        pretraining, morphology coarsening, attention-signal discovery, and experimental
        side pipelines were intentionally removed. The optional "HISTAI + biological"
        experiment must be added as a new, explicit corpus extension rather than reviving
        those historical pipelines.
    ''')

    write(root / "docs/pipeline.md", r'''
        # EAF paper pipeline

        ## Invariants

        1. **No downstream labels in EAF training.** Tile-EAF, WSI-EAF and both embedding
           distillation stages train on HISTAI only.
        2. **Teacher stays full.** Distillation targets are always generated by the
           corresponding unpruned model.
        3. **Student chain is realistic.** Final WSI distillation consumes embeddings from
           the already distilled/pruned tile encoder.
        4. **WSI source and teacher caches are different objects.** `source_wsi_root` may
           contain hidden states produced from pruned tile embeddings; `teacher_wsi_root`
           contains attention/slide embeddings from the full pipeline.
        5. **THUNDER/EAGLE are evaluation only.** Only lightweight downstream heads may use
           their labels.

        ## 1. HISTAI preparation

        ```bash
        export EAF_WSI_ROOT=/path/to/WSI
        python scripts/eaf.py data plan-histai --data-root "$EAF_WSI_ROOT"
        python scripts/eaf.py data download-histai --data-root "$EAF_WSI_ROOT"
        ```

        Produce/attach TRIDENT coordinates and keep one canonical HISTAI manifest. No TCGA,
        HEST or GTEx slide belongs to the main EAF training corpus.

        ## 2. Full tile teacher cache

        ```bash
        python scripts/eaf.py cache tile \
          --data-root "$EAF_WSI_ROOT" \
          --manifest <histai_slides_with_coords.csv> \
          --encoder <tile-model> \
          --output-dir <tile_full_cache> \
          --dataset histai
        ```

        This stores the frozen tile encoder's final attention and full embedding.

        ## 3. Train Tile-EAF

        ```bash
        python scripts/training/train_tile_eaf.py --help
        ```

        Source = patch tokens after the selected pruning layer. Target = final-layer teacher
        attention. The tile teacher remains frozen.

        ## 4. Distill the pruned tile encoder

        ```bash
        python scripts/training/distill_tile_encoder.py \
          --manifest <histai_manifest> \
          --data-root "$EAF_WSI_ROOT" \
          --forecaster-ckpt <tile_eaf.pt> \
          --target-cache-index <full_tile_cache_index.csv> \
          --keep-ratio 0.20 \
          --model-name <tile-model>
        ```

        Student = EAF-pruned tile encoder with trainable LoRA in the remaining blocks.
        Target = embedding produced by the same unpruned tile encoder on the same image.

        Build the student tile-input cache:

        ```bash
        python scripts/eaf.py cache tile \
          --data-root "$EAF_WSI_ROOT" \
          --manifest <histai_slides_with_coords.csv> \
          --pruned-adapter-ckpt <distilled_tile.pt> \
          --output-dir <tile_pruned_cache> \
          --dataset histai
        ```

        ## 5. Build two WSI caches

        Use `scripts/features/cache_wsi_teacher.py` twice.

        **Full teacher cache**
        - input: `<tile_full_cache>`
        - save full TITAN attention + slide embedding
        - output: `<wsi_full_teacher_cache>`

        **Student-source cache**
        - input: `<tile_pruned_cache>`
        - save TITAN hidden state at the intended WSI pruning layer
        - output: `<wsi_pruned_source_cache>`

        The second cache is a *source representation*, not the final teacher target.

        ## 6. Train WSI-EAF

        ```bash
        python scripts/training/train_wsi_eaf.py \
          --tile-input-root <tile_pruned_cache> \
          --source-wsi-root <wsi_pruned_source_cache> \
          --teacher-wsi-root <wsi_full_teacher_cache> \
          --input-source titan_hidden \
          --titan-hidden-layer <L> \
          --architecture dense_alibi \
          --tile-encoder <tile-model>
        ```

        Input = intermediate TITAN state produced from the pruned tile encoder. Target =
        full-TITAN final attention generated from the unpruned tile encoder.

        ## 7. Distill the pruned WSI encoder

        ```bash
        python scripts/training/distill_wsi_titan.py \
          --tile-input-root <tile_pruned_cache> \
          --teacher-wsi-root <wsi_full_teacher_cache> \
          --forecaster-checkpoint <wsi_eaf.pt> \
          --keep-ratio 0.20
        ```

        Student = pruned/LoRA TITAN fed by the pruned tile encoder. Target = slide embedding
        from the full tile encoder + full TITAN pipeline.

        ## 8. Evaluation only

        Tile level / THUNDER:

        ```bash
        python scripts/evaluation/evaluate_tile_thunder.py \
          --base-data-folder <thunder_root> \
          --pruned-checkpoint <distilled_tile.pt>
        ```

        WSI level / public EAGLE-compatible tasks:

        ```bash
        python scripts/evaluation/evaluate_wsi_eagle.py \
          --teacher-wsi-root <benchmark_full_titan_cache> \
          --tile-input-root <benchmark_pruned_tile_cache> \
          --pruned-checkpoint <distilled_wsi.pt>
        ```

        Both evaluations compare baseline and pruned representations on the same examples
        and train only downstream heads.

        ## 9. HISTAI + biological extension

        Keep this as a separate manifest/corpus ID and run the exact same four training
        stages. Do not mix it into the primary HISTAI run. The biological source still has
        to be selected and frozen before implementation.
    ''')

    write(root / "docs/data_layout.md", r'''
        # Data layout

        `$EAF_WSI_ROOT` is the single runtime root. The git repository contains code and
        small metadata only.

        ```text
        $EAF_WSI_ROOT/
        ├── sources/       raw WSI
        ├── datasets/
        │   ├── pretraining/histai_eaf_wsi_v1/
        │   └── downstream/                 # THUNDER/EAGLE labels/manifests only
        ├── caches/
        │   ├── tile_full/
        │   ├── tile_pruned/
        │   ├── wsi_full_teacher/
        │   └── wsi_pruned_source/
        ├── checkpoints/
        ├── results/
        └── logs/
        ```

        The primary training corpus is HISTAI. Downstream benchmark data never enters an
        EAF/embedding-distillation training manifest.

        New generated numerical artifacts use `.npyd` directories and `metadata.json`;
        historical HDF5 remains readable during migration.
    ''')

    write(root / "CLAUDE.md", r'''
        # EAF contributor notes

        Use the Conda environment in `environment.yml` and run commands from the repository root.
        `$EAF_WSI_ROOT` is the single runtime root for data, caches, checkpoints, results and logs.

        The primary EAF training corpus is unlabeled HISTAI. THUNDER and EAGLE-compatible
        cohorts are downstream evaluation only. The final WSI experiment must keep the pruned
        student source cache separate from the full teacher cache.

        Supported workflow: `README.md` and `docs/pipeline.md`.
    ''')

    write(root / "AGENTS.md", r'''
        # Repository Guidelines

        ## Scope

        This repository supports one experimental chain only:

        `HISTAI -> Tile-EAF -> tile embedding distillation -> WSI-EAF -> WSI embedding distillation -> THUNDER/EAGLE evaluation`.

        Do not reintroduce supervised THUNDER pretraining, TCGA/HEST/GTEx EAF pretraining,
        morphology coarsening, signal-discovery branches, or competing pruning baselines.

        ## Scientific invariants

        - EAF and both distillation stages use unlabeled HISTAI data only.
        - Downstream labels are used only by evaluation heads.
        - Tile distillation targets the same frozen unpruned tile encoder.
        - WSI student inputs come from the distilled/pruned tile encoder.
        - WSI targets come from the full tile + full WSI teacher pipeline.
        - Never collapse `source_wsi_root` and `teacher_wsi_root` in the final combined run.
        - Patient/case grouping must remain disjoint across train/validation/test.

        ## Code structure

        Reusable logic belongs under `src/`; `scripts/` contains thin entry points.
        Supported training entry points are `train_tile_eaf.py`, `distill_tile_encoder.py`,
        `train_wsi_eaf.py`, and `distill_wsi_titan.py`. Supported evaluation entry points
        are `evaluate_tile_thunder.py` and `evaluate_wsi_eagle.py`.

        Runtime artifacts never belong in git. Use `$EAF_WSI_ROOT` for datasets, caches,
        checkpoints, results and logs.

        ## Tests

        Run `pytest` from the repository root. Optional heavy dependencies must be guarded
        with `pytest.importorskip(...)` or imported lazily.
    ''')


def write_tests(root: Path) -> None:
    write(root / "tests/data/wsi/test_histai_corpora.py", r'''
        from src.data.wsi.corpora import _histai_case_id, _histai_rank, _is_histai_he


        def test_histai_he_selection_and_case_id():
            path = "train/case_0042/slide_20x_H&E_0.tiff"
            assert _histai_case_id(path) == "case_0042"
            assert _is_histai_he(path)
            assert _histai_rank(path)[0] == 0


        def test_histai_rejects_non_he():
            assert not _is_histai_he("case_1/slide_IHC_0.tiff")
    ''')

    write(root / "tests/data/wsi/test_splits.py", r'''
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
    ''')


def clean_stale_references(root: Path) -> None:
    # Remove stale cross-reference to the deleted duplicate teacher implementation.
    path = root / "src/models/online_tile_eaf.py"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "        # applied in HookedViTTileTeacherAdapter._make_final_attn_hook\n"
            "        # (src/wsi_pipeline/model_adapters.py) and\n"
            "        # FrozenTimmAttentionTeacher._cls_patch_attention\n"
            "        # (src/training/online_attention_distillation.py).\n",
            "        # Same memory-saving CLS-row computation used by the offline tile teacher.\n",
        )
        path.write_text(text, encoding="utf-8")

    path = root / "src/wsi_pipeline/model_adapters.py"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        text = re.sub(
            r"\s*-- mirroring the\n\s*auto-resolve-from-checkpoint pattern in\n\s*``scripts/training/train_multi_thunder_classifier\.py``'s ``--pruned-adapter-ckpt`` path\.",
            ". The checkpoint is self-describing and independent of downstream training.",
            text,
            count=1,
        )
        path.write_text(text, encoding="utf-8")

    path = root / "src/models/wsi/dense_forecaster.py"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        text = re.sub(
            r"\nReplaces the earlier `WSILandmarkForecaster`.*?(?=\n\n)",
            "\nThe implementation intentionally uses standard dense self-attention to mirror the WSI teacher block structure.",
            text,
            count=1,
            flags=re.S,
        )
        path.write_text(text, encoding="utf-8")


def remove_eaf_experiment_audit(root: Path) -> None:
    path = root / "scripts/eaf.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("from src.wsi_pipeline.experiment_catalog import audit_experiments, write_catalog  # noqa: E402\n", "")
    text = re.sub(r"\n\ndef cmd_experiments_audit\(.*?(?=\n\ndef cmd_plan_histai)", "", text, count=1, flags=re.S)
    text = re.sub(r"\n    experiments = sub\.add_parser\(\"experiments\".*?(?=\n    data = sub\.add_parser)", "", text, count=1, flags=re.S)
    path.write_text(text, encoding="utf-8")


def validate(root: Path) -> None:
    targets = [
        "scripts/eaf.py",
        "scripts/training/train_tile_eaf.py",
        "scripts/training/distill_tile_encoder.py",
        "scripts/training/train_wsi_eaf.py",
        "scripts/training/distill_wsi_titan.py",
        "scripts/features/cache_wsi_teacher.py",
        "scripts/evaluation/evaluate_tile_thunder.py",
        "scripts/evaluation/evaluate_wsi_eagle.py",
        "src/data/wsi/corpora.py",
        "src/data/wsi/splits.py",
        "src/data/wsi/wsi_forecaster_dataset.py",
        "src/data/wsi/wsi_pruned_titan_dataset.py",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", *targets], cwd=root, text=True
    )
    if result.returncode != 0:
        raise SystemExit("Python syntax validation failed")

    forbidden = [
        "src.baselines",
        "morphology_coarsening",
        "attention_signal",
        "train_multi_thunder_classifier",
        "build_eaf_multisource_manifest",
        "plan_gtex(",
        "register_hest(",
    ]
    failures = []
    for needle in forbidden:
        proc = subprocess.run(
            ["git", "grep", "-n", "--", needle], cwd=root, text=True, capture_output=True
        )
        if proc.returncode == 0:
            failures.append((needle, proc.stdout.strip()))
    if failures:
        print("\nStale references remain:", file=sys.stderr)
        for needle, matches in failures:
            print(f"\n[{needle}]\n{matches}", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-head-check", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    root = git_root()
    os.chdir(root)
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    if not args.skip_head_check and head != BASE_COMMIT:
        raise SystemExit(
            f"This patch was audited against {BASE_COMMIT}, but HEAD is {head}. "
            "Rebase/create a branch from that commit or rerun with --skip-head-check after reviewing the diff."
        )
    dirty = run("git", "status", "--porcelain").stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit("Working tree is dirty. Commit/stash first, or pass --allow-dirty deliberately.")

    print(f"Applying EAF paper-pipeline refactor in {root}")
    for rel in REMOVE_PATHS:
        remove(root, rel)
    for src, dst in RENAMES.items():
        rename(root, src, dst)

    replace_all_text_references(root)
    rewrite_corpora(root)
    rewrite_eaf_cli(root)
    remove_eaf_experiment_audit(root)
    rewrite_models_init(root)
    rewrite_thunder_loaders(root)
    rewrite_splits(root)
    rewrite_wsi_forecaster_dataset(root)
    patch_wsi_training_scripts(root)
    write_tile_thunder_evaluation(root)
    rewrite_docs(root)
    write_tests(root)
    clean_stale_references(root)

    if not args.no_validate:
        validate(root)

    print("\nRefactor applied. Review with:")
    print("  git status --short")
    print("  git diff --stat")
    print("  git diff")
    print("\nThen run the repository test suite in the EAF environment:")
    print("  pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
