"""Downstream task discovery across cohorts."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")

from scripts.evaluation.evaluate_wsi_eagle import _cv_splits, _paired_indices, collect_embeddings, discover_tasks


def test_complementary_single_class_cohorts_form_one_combined_task(tmp_path):
    root = tmp_path / "labels"
    for cohort, label in (("LUAD", "0"), ("LUSC", "1")):
        directory = root / cohort / "labels"
        directory.mkdir(parents=True)
        (directory / "subtyping.csv").write_text(
            f"slide_id,label\n{cohort}_1,{label}\n{cohort}_2,{label}\n"
        )
    tasks = discover_tasks(root)
    assert [(task.cohort, task.task) for task in tasks] == [
        ("LUAD", "subtyping"), ("LUSC", "subtyping"),
        ("LUAD+LUSC", "subtyping"),
    ]
    calls = []

    def embedding(cohort_dir, slide_id):
        calls.append((cohort_dir.name, slide_id))
        return np.asarray([0.0 if cohort_dir.name == "LUAD" else 1.0])

    X, y, groups, keys, found, missing = collect_embeddings(
        tasks[-1], cache_root=tmp_path / "cache", embed_fn=embedding, min_slides=2,
    )
    assert (found, missing) == (4, 0)
    assert X.shape == (4, 1)
    assert y.tolist() == ["0", "0", "1", "1"]
    assert len(set(groups)) == 4
    assert keys.tolist() == ["LUAD/LUAD_1", "LUAD/LUAD_2", "LUSC/LUSC_1", "LUSC/LUSC_2"]
    assert calls == [("LUAD", "LUAD_1"), ("LUAD", "LUAD_2"),
                     ("LUSC", "LUSC_1"), ("LUSC", "LUSC_2")]


def test_cv_keeps_slides_from_same_patient_together():
    X = np.arange(24).reshape(12, 2)
    y = np.asarray([0] * 6 + [1] * 6)
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"])
    folds, splits = _cv_splits(X, y, groups, folds=3, seed=17)
    assert folds == 3
    assert all(set(groups[train]).isdisjoint(groups[test]) for train, test in splits)


def test_paired_indices_use_shared_slides_in_baseline_order():
    baseline = np.asarray(["a", "b", "c"])
    pruned = np.asarray(["c", "a", "d"])
    first, second = _paired_indices([baseline, pruned])
    assert first.tolist() == [0, 2]
    assert second.tolist() == [1, 0]
