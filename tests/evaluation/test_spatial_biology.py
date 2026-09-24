"""Synthetic scientific-contract tests: no downloads or trained models."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.evaluation.spatial.data import aligned_matrix, program_targets, read_spots, sha256
from src.evaluation.spatial.metrics import coverage_masks, map_metrics, paired_summary
from src.evaluation.spatial.protocol import evaluate, load_config, validate_inputs
from src.evaluation.spatial.cli import main


@pytest.fixture
def cohort(tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    for patient in range(6):
        split = "train" if patient < 2 else "validation" if patient < 4 else "test"
        for spot in range(8):
            rows.append(dict(spot_id=f"p{patient}_s{spot}", slide_id=f"s{patient}", patient_id=f"p{patient}",
                             cohort="study", split=split, x=spot * 10, y=0,
                             niche="immune" if spot < 4 else "stromal"))
    spots = pd.DataFrame(rows)
    spots.to_csv(tmp_path / "spots.csv", index=False)
    expression = rng.uniform(0.5, 5, (len(spots), 4))
    np.savez(tmp_path / "expression.npz", spot_id=spots.spot_id.to_numpy(dtype=str),
             genes=np.array(["A", "B", "C", "D"]), expression=expression)
    (tmp_path / "sets.gmt").write_text("immune\tsynthetic\tA\tB\nstromal\tsynthetic\tC\tD\n")
    for name in ("full", "eaf"):
        # Reversed rows ensure the implementation actually aligns identifiers.
        np.savez(tmp_path / f"{name}.npz", spot_id=spots.spot_id.to_numpy(dtype=str)[::-1], embeddings=expression[::-1])
        np.savez(tmp_path / f"{name}_slide.npz", slide_id=np.array([f"s{i}" for i in range(6)]),
                 embeddings=expression.reshape(6, 8, 4).mean(axis=1))
    selection = spots[["slide_id", "x", "y"]].copy()
    selection["width"], selection["height"] = 10, 10
    selection["kept"] = (spots.niche == "immune").astype(int)
    selection.to_csv(tmp_path / "selection.csv", index=False)
    config = '''version = 1
coordinate_system = "level0_xy"
reference = "full"
spots = "spots.csv"
expression = "expression.npz"
normalization = "log1p_cp10k"
signatures = "sets.gmt"
bootstrap = 30
max_plot_slides = 1
'''
    for name in ("full", "eaf"):
        config += f'''\n[[methods]]
name = "{name}"
embeddings = "{name}.npz"
slide_embeddings = "{name}_slide.npz"
citation = "synthetic"
supervision = "synthetic"
pretraining_overlap = "none"
'''
        if name == "eaf":
            config += 'selection = "selection.csv"\n'
    path = tmp_path / "config.toml"
    path.write_text(config)
    return path, spots, expression


def test_paired_evaluation_detects_lost_niche(cohort):
    path, _, _ = cohort
    config = load_config(path)
    data = validate_inputs(config)
    result = evaluate(config, data)
    summary = result["summary"]
    paired = summary[(summary.method == "eaf") & summary.level.isin(["spot", "slide"])]
    assert np.allclose(paired.delta.dropna(), 0)
    lost = result["coverage"].query("method == 'eaf' and niche == 'stromal'")
    assert lost.lost.all() and (lost.coverage == 0).all()
    full = result["coverage"].query("method == 'full'")
    assert (full.coverage == 1).all()
    assert set(summary.level) == {"spot", "slide", "coverage"}


def test_missing_ids_not_silently_intersected(tmp_path):
    np.savez(tmp_path / "bad.npz", spot_id=np.array(["a", "b"]), embeddings=np.ones((2, 3)))
    with pytest.raises(ValueError, match="coverage"):
        aligned_matrix(tmp_path / "bad.npz", ["a", "c"])


def test_patient_leakage_and_duplicate_coordinates(cohort):
    path, spots, _ = cohort
    spots.loc[spots.patient_id == "p4", "patient_id"] = "p0"
    spots.to_csv(path.parent / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Patient leakage"):
        read_spots(path.parent / "bad.csv")


def test_target_statistics_ignore_test_expression():
    x = np.array([[1., 2], [3, 4], [5, 6], [7, 8]])
    train = np.array([True, True, False, False])
    spec = {"p": {"source": "test", "genes": ["a", "b"]}}
    original = program_targets(x, ["a", "b"], train, spec)
    x[~train] += 1000
    changed = program_targets(x, ["a", "b"], train, spec)
    for a, b in zip(original[2], changed[2]):
        np.testing.assert_equal(a, b)
    np.testing.assert_equal(original[0][train], changed[0][train])


def test_constant_and_missing_genes_are_audited():
    x = np.array([[1., 3], [2, 3], [4, 3]])
    specs = {"good": {"source": "test", "genes": ["a"]},
             "bad": {"source": "test", "genes": ["b", "missing"]}}
    _, names, _, audit = program_targets(x, ["a", "b"], np.array([True, True, False]), specs)
    assert names == ["good"]
    assert audit["bad"]["constant"] == ["b"] and audit["bad"]["missing"] == ["missing"]


def test_spatial_shuffle_and_constant_scores():
    coords = np.column_stack([np.arange(8), np.zeros(8)])
    truth = np.arange(8, dtype=float)
    exact = map_metrics(truth, truth, coords)
    bad = map_metrics(truth, truth[::-1], coords)
    assert exact["edge_rmse"] == 0 and bad["edge_rmse"] > 0
    assert exact["pearson"] == 1 and bad["pearson"] == -1
    assert np.isnan(map_metrics(truth, np.ones(8), coords)["pearson"])


def test_geometry_counts_overlaps_once_and_excludes_outside():
    coords = np.array([[0, 0], [5, 5], [10, 0], [20, 0]])
    rectangles = pd.DataFrame([dict(x=0, y=0, width=10, height=10, kept=1),
                               dict(x=5, y=0, width=10, height=10, kept=0)])
    covered, kept = coverage_masks(coords, rectangles)
    np.testing.assert_equal(covered, [True, True, True, False])
    np.testing.assert_equal(kept, [True, True, False, False])


def test_external_predictions_apply_same_program_transform(cohort):
    path, spots, expression = cohort
    test = spots.split == "test"
    np.savez(path.parent / "prediction.npz", spot_id=spots.loc[test, "spot_id"].to_numpy(dtype=str),
             genes=np.array(["A", "B", "C", "D"]), expression=expression[test])
    with path.open("a") as handle:
        handle.write(f'''\n[[methods]]
name = "external"
predictions = "prediction.npz"
citation = "synthetic"
supervision = "paired RNA"
pretraining_overlap = "none"
split_sha256 = "{sha256(path.parent / 'spots.csv')}"
test_expression_used = false
training_patients = ["p0", "p1"]
validation_patients = ["p2", "p3"]
''')
    config = load_config(path)
    data = validate_inputs(config)
    result = evaluate(config, data)
    np.testing.assert_allclose(result["predictions"]["external"], result["truth"])
    config["methods"][-1]["training_patients"] = ["p4"]
    with pytest.raises(ValueError, match="training_patients"):
        validate_inputs(config)
    config["methods"][-1]["test_expression_used"] = True
    with pytest.raises(ValueError, match="test_expression_used"):
        validate_inputs(config)


def test_cli_outputs_and_registry_block(cohort, tmp_path, capsys):
    path, _, _ = cohort
    assert main(["validate", "--config", str(path)]) == 0
    assert "valid_inputs_not_launch_authorization" in capsys.readouterr().out
    registry = tmp_path / "registry.toml"
    registry_text = f'''version = 1
[experiments.hest_biological_conch15_titan]
family = "wsi_eaf"
stage = "evaluation"
status = "blocked"
protocol_sha256 = "{sha256(path)}"
[experiments.hest_biological_conch15_titan.variants.final]
'''
    registry.write_text(registry_text)
    args = ["evaluate", "--config", str(path), "--experiment-registry", str(registry),
            "--data-root", str(tmp_path / "runtime"), "--no-plots"]
    with pytest.raises(RuntimeError, match="blocked"):
        main(args)
    assert not (tmp_path / "runtime").exists()
    registry.write_text(registry_text.replace('status = "blocked"', 'status = "ready"'))
    assert main(args) == 0
    out = tmp_path / "runtime/results/wsi_eaf/evaluation/hest_biological_conch15_titan/final/seed_42"
    assert (out / "patient_summary.csv").exists()
    assert json.loads((out / "protocol.json").read_text())["schema"] == "eaf.spatial_biology.v1"
    with pytest.raises(FileExistsError):
        main(args)


def test_report_writes_figures(cohort, tmp_path):
    pytest.importorskip("matplotlib")
    from src.evaluation.spatial.report import write_report
    path, _, _ = cohort
    config = load_config(path)
    data = validate_inputs(config)
    write_report(tmp_path, config, data, evaluate(config, data))
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3
    assert (tmp_path / "report.md").exists()


def test_sparse_h5ad_reading_and_identifier_alignment(cohort, monkeypatch):
    ad = pytest.importorskip("anndata")
    if hasattr(ad.settings, "allow_write_nullable_strings"):
        monkeypatch.setattr(ad.settings, "allow_write_nullable_strings", True)
    from scipy.sparse import csr_matrix
    from src.evaluation.spatial.data import read_expression
    path, spots, expression = cohort
    data = ad.AnnData(X=csr_matrix(expression[::-1]))
    data.obs_names = spots.spot_id.to_numpy()[::-1]
    data.var_names = ["A", "B", "C", "D"]
    data.write_h5ad(path.parent / "expression.h5ad")
    actual, genes = read_expression(path.parent / "expression.h5ad", spots.spot_id, normalization="log1p_cp10k")
    np.testing.assert_equal(actual, expression)
    assert genes.tolist() == ["A", "B", "C", "D"]


def test_bootstrap_balances_patients_not_sections():
    rows = []
    for patient, n, delta in (("a", 10, 1.0), ("b", 1, 3.0)):
        for _ in range(n):
            for method, value in (("full", 0.0), ("eaf", delta)):
                rows.append(dict(level="spot", target="x", stratum="__all__", metric="rmse",
                                 patient_id=patient, method=method, value=value))
    result = paired_summary(pd.DataFrame(rows), "full", bootstrap=100)
    eaf = result[result.method == "eaf"].iloc[0]
    assert eaf.delta == 2.0 and eaf.n_paired == 2


def test_selection_grid_mismatch_rejected(cohort):
    path, _, _ = cohort
    config = load_config(path)
    table = pd.read_csv(path.parent / "selection.csv")
    table.loc[0, "x"] += 1
    table.to_csv(path.parent / "other.csv", index=False)
    config["methods"][0]["selection"] = str(path.parent / "other.csv")
    with pytest.raises(ValueError, match="same full tile grid"):
        validate_inputs(config)
