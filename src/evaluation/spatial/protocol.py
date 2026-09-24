"""Cache-based biological evaluation with shared frozen downstream heads."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tomllib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import aligned_matrix, program_targets, read_expression, read_signatures, read_spots, sha256
from .metrics import map_metrics, niche_coverage, paired_summary


def load_config(path):
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text())
    if config.get("version") != 1 or config.get("coordinate_system") != "level0_xy":
        raise ValueError("Require version=1 and coordinate_system='level0_xy'")
    file_keys = {"spots", "expression", "signatures", "embeddings", "predictions", "selection", "slide_embeddings", "provenance"}
    for section in [config, *config.get("methods", [])]:
        for key in file_keys & section.keys():
            value = os.path.expandvars(section[key])
            if "$" in value:
                raise ValueError(f"Unresolved environment variable in {key}: {value}")
            section[key] = str((path.parent / value).resolve())
    config["config_path"] = str(path)
    if int(config.get("bootstrap", 2000)) < 1 or int(config.get("max_plot_slides", 6)) < 0:
        raise ValueError("bootstrap must be positive and max_plot_slides nonnegative")
    methods = config.get("methods", [])
    names = [m["name"] for m in methods]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two uniquely named methods are required")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError("Method names must contain only letters, digits, underscores or hyphens")
    if config.get("reference", "full") not in names:
        raise ValueError("Reference method is absent")
    for method in methods:
        if ("embeddings" in method) == ("predictions" in method):
            raise ValueError("Each method requires exactly one of embeddings or predictions")
        for field in ("citation", "supervision", "pretraining_overlap"):
            if not method.get(field):
                raise ValueError(f"Method {method['name']} must declare {field}")
    return config


def read_selection(path, slides):
    table = pd.read_csv(path, dtype={"slide_id": str}, keep_default_na=False)
    columns = {"slide_id", "x", "y", "width", "height", "kept"}
    if columns - set(table) or table.empty:
        raise ValueError("Selection requires slide_id,x,y,width,height,kept")
    if not set(slides) <= set(table.slide_id):
        raise ValueError("Selection missing evaluated slides")
    for column in columns - {"slide_id"}:
        table[column] = pd.to_numeric(table[column])
    if not np.isfinite(table[list(columns - {"slide_id"})].to_numpy()).all():
        raise ValueError("Nonfinite selection geometry")
    if (table[["width", "height"]] <= 0).any().any() or not table.kept.isin([0, 1]).all():
        raise ValueError("Selection widths/heights must be positive and kept must be 0/1")
    if table.duplicated(["slide_id", "x", "y", "width", "height"]).any():
        raise ValueError("Duplicate selection rectangles")
    return table


def validate_inputs(config):
    spots = read_spots(config["spots"])
    expression, genes = read_expression(config["expression"], spots.spot_id,
                                        normalization=config["normalization"], layer=config.get("expression_layer"))
    signatures = read_signatures(config["signatures"])
    train = spots.split.to_numpy() == "train"
    targets, names, transform, audit = program_targets(
        expression, genes, train, signatures, config.get("min_signature_coverage", 0.8))
    slides = spots.drop_duplicates("slide_id").reset_index(drop=True)
    matrices, selections, slide_matrices, predictions = {}, {}, {}, {}
    input_hashes = {key: sha256(config[key]) for key in ("spots", "expression", "signatures", "config_path")}
    provenance = {}
    for method in config["methods"]:
        name = method["name"]
        if "provenance" in method:
            provenance[name] = json.loads(Path(method["provenance"]).read_text())
        if "embeddings" in method:
            matrices[name] = aligned_matrix(method["embeddings"], spots.spot_id)
        else:
            # External ST models must use the same fixed split and no measured test RNA.
            if method.get("split_sha256") != input_hashes["spots"] or method.get("test_expression_used") is not False:
                raise ValueError("External predictions require matching split_sha256 and test_expression_used=false")
            allowed_train = set(spots.loc[spots.split == "train", "patient_id"])
            allowed_val = set(spots.loc[spots.split == "validation", "patient_id"])
            for field, allowed in (("training_patients", allowed_train), ("validation_patients", allowed_val)):
                if field not in method or not set(method[field]) <= allowed:
                    raise ValueError(f"External predictions must declare leakage-free {field}")
            test_ids = spots.loc[spots.split == "test", "spot_id"]
            predicted_expression, predicted_genes = read_expression(
                method["predictions"], test_ids, normalization="log1p_cp10k")
            lookup = {gene: i for i, gene in enumerate(predicted_genes)}
            mean, scale, weights = transform
            used = np.flatnonzero(weights.any(axis=1))
            if not set(genes[used]) <= set(predicted_genes):
                raise ValueError("External prediction is missing genes used in program targets")
            values = predicted_expression[:, [lookup[g] for g in genes[used]]]
            predictions[name] = ((values - mean[used]) / scale[used]) @ weights[used]
        if "selection" in method:
            selections[name] = read_selection(method["selection"], slides.slide_id)
        if "slide_embeddings" in method:
            slide_matrices[name] = aligned_matrix(method["slide_embeddings"], slides.slide_id, id_key="slide_id")
        for key in ("embeddings", "predictions", "selection", "slide_embeddings", "provenance"):
            if key in method:
                input_hashes[f"{name}.{key}"] = sha256(method[key])
    # Selection comparisons require exactly the same original tile grid.
    grid = None
    for selection in selections.values():
        current = set(map(tuple, selection[["slide_id", "x", "y", "width", "height"]].to_numpy()))
        if grid is not None and current != grid:
            raise ValueError("Selection methods must share the same full tile grid")
        grid = current
    reference = config.get("reference", "full")
    if selections and reference not in selections:
        # Full teacher processes the original grid; derive its coverage exactly.
        selections[reference] = next(iter(selections.values())).assign(kept=1)
    if slide_matrices and reference not in slide_matrices:
        raise ValueError("Slide embedding evaluation requires the reference slide embeddings")
    return dict(spots=spots, slides=slides, targets=targets, names=names, genes=genes, transform=transform,
                audit=audit, matrices=matrices, selections=selections, slide_matrices=slide_matrices,
                predictions=predictions, input_hashes=input_hashes, provenance=provenance)


def fit_predict(matrices, targets, metadata, *, pca_components=256, alphas=(0.01, 0.1, 1, 10, 100, 1000), seed=42):
    """Train-only transforms; patient-balanced validation MSE chooses alpha per target."""
    if pca_components < 1 or not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Positive PCA dimension and finite positive ridge alphas required")
    train, val, test = [metadata.split.to_numpy() == split for split in ("train", "validation", "test")]
    if train.sum() < 2 or not val.any() or not test.any():
        raise ValueError("Need >=2 training rows and nonempty validation and test sets")
    dimension = min(pca_components, int(train.sum()) - 1, *(x.shape[1] for x in matrices.values()))
    predictions, fits = {}, {}
    # Equal patient contribution to fitting, irrespective of spot/section counts.
    train_patients = metadata.loc[train, "patient_id"]
    counts = train_patients.value_counts()
    sample_weight = train_patients.map(lambda p: 1 / counts[p]).to_numpy(copy=True)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    for name, values in matrices.items():
        scaler = StandardScaler().fit(values[train])
        pca = PCA(n_components=dimension, svd_solver="full", random_state=seed).fit(scaler.transform(values[train]))
        xtrain, xval, xtest = [pca.transform(scaler.transform(values[mask])) for mask in (train, val, test)]
        best = np.full(targets.shape[1], np.inf)
        selected = np.zeros(targets.shape[1])
        output = np.empty((int(test.sum()), targets.shape[1]))
        for alpha in sorted(alphas):
            model = Ridge(alpha=alpha).fit(xtrain, targets[train], sample_weight=sample_weight)
            error = (model.predict(xval) - targets[val]) ** 2
            loss = pd.DataFrame(error).groupby(metadata.loc[val, "patient_id"].to_numpy()).mean().mean().to_numpy()
            improved = loss < best
            best[improved], selected[improved] = loss[improved], alpha
            output[:, improved] = model.predict(xtest)[:, improved]
        predictions[name] = output
        fits[name] = {"pca_components": dimension, "alphas": selected.tolist(), "validation_mse": best.tolist()}
    return predictions, fits


def evaluate(config, data, *, seed=42):
    spots, targets, names = data["spots"], data["targets"], data["names"]
    predictions = dict(data["predictions"])
    fits = {}
    if data["matrices"]:
        predicted, fits = fit_predict(data["matrices"], targets, spots,
                                      pca_components=config.get("pca_components", 256),
                                      alphas=config.get("alphas", [0.01, 0.1, 1, 10, 100, 1000]), seed=seed)
        predictions.update(predicted)
    test = spots.split.to_numpy() == "test"
    test_spots = spots[test].reset_index(drop=True)
    truth = targets[test]
    rows = []
    for method, prediction in predictions.items():
        for slide, group in test_spots.groupby("slide_id", sort=True):
            strata = [("__all__", group)]
            if "niche" in group:
                strata += [(n, g) for n, g in group.groupby("niche") if n]
            for stratum, subset in strata:
                idx = subset.index.to_numpy()
                for j, target in enumerate(names):
                    for metric, value in map_metrics(truth[idx, j], prediction[idx, j], subset[["x", "y"]].to_numpy()).items():
                        rows.append(dict(level="spot", method=method, target=target, stratum=stratum,
                                         metric=metric, value=value, slide_id=slide, patient_id=group.patient_id.iloc[0],
                                         n_spots=len(idx), status="ok" if np.isfinite(value) else "undefined_constant_or_too_few_spots"))
    coverage = [niche_coverage(test_spots, selection, method) for method, selection in data["selections"].items()]
    for table in coverage:
        for row in table.itertuples(index=False):
            for metric in ("coverage", "relative_coverage"):
                rows.append(dict(level="coverage", method=row.method, target="niche_coverage", stratum=row.niche,
                                 metric=metric, value=getattr(row, metric), slide_id=row.slide_id,
                                 patient_id=row.patient_id, n_spots=row.n_eligible, status=row.status))
    slide_fits = {}
    slide_predictions, slide_targets, slide_names = {}, None, []
    if data["slide_matrices"]:
        slides = data["slides"]
        aggregates = []
        niche_names = sorted(set(spots.loc[spots.split == "train", "niche"]) - {""}) if "niche" in spots else []
        for slide in slides.slide_id:
            mask = spots.slide_id.to_numpy() == slide
            values = [*targets[mask].mean(axis=0), *targets[mask].std(axis=0)]
            if niche_names:
                annotations = spots.loc[mask, "niche"]
                if (annotations == "").any():
                    raise ValueError("WSI niche proportions require complete spot niche annotations")
                values.extend(float((annotations == n).mean()) for n in niche_names)
            aggregates.append(values)
        slide_targets = np.asarray(aggregates)
        slide_names = [f"mean:{n}" for n in names] + [f"std:{n}" for n in names] + [f"proportion:{n}" for n in niche_names]
        slide_predictions, slide_fits = fit_predict(data["slide_matrices"], slide_targets, slides,
                                                   pca_components=config.get("pca_components", 256),
                                                   alphas=config.get("alphas", [0.01, 0.1, 1, 10, 100, 1000]), seed=seed)
        stest = slides.split.to_numpy() == "test"
        for method, prediction in slide_predictions.items():
            for i, (_, slide) in enumerate(slides[stest].iterrows()):
                for j, target in enumerate(slide_names):
                    rows.append(dict(level="slide", method=method, target=target, stratum="__all__",
                                     metric="absolute_error", value=abs(prediction[i, j] - slide_targets[stest][i, j]),
                                     slide_id=slide.slide_id, patient_id=slide.patient_id, n_spots=0, status="ok"))
    results = pd.DataFrame(rows)
    summary = paired_summary(results, config.get("reference", "full"), seed=seed, bootstrap=config.get("bootstrap", 2000))
    return dict(results=results, summary=summary, coverage=pd.concat(coverage, ignore_index=True) if coverage else pd.DataFrame(),
                test_spots=test_spots, truth=truth, predictions=predictions, fits=fits, slide_fits=slide_fits,
                slide_predictions=slide_predictions, slide_targets=slide_targets, slide_names=slide_names)


def save_outputs(output, config, data, result):
    output = Path(output)
    for name in ("results", "coverage"):
        result[name].to_csv(output / f"{name}.csv", index=False)
    result["summary"].to_csv(output / "patient_summary.csv", index=False)
    test_spots = result["test_spots"]
    test_spots.to_csv(output / "test_spots.csv", index=False)
    np.savez_compressed(output / "program_targets.npz", spot_id=test_spots.spot_id.to_numpy(dtype=str),
                        targets=np.asarray(data["names"]), values=result["truth"])
    mean, scale, weights = data["transform"]
    np.savez_compressed(output / "program_transform.npz", genes=data["genes"], targets=np.asarray(data["names"]),
                        mean=mean, scale=scale, weights=weights)
    for name, values in result["predictions"].items():
        np.savez_compressed(output / f"predictions_{name}.npz", spot_id=test_spots.spot_id.to_numpy(dtype=str),
                            targets=np.asarray(data["names"]), values=values)
    if result["slide_predictions"]:
        slides = data["slides"]
        mask = slides.split.to_numpy() == "test"
        np.savez_compressed(output / "slide_targets.npz", slide_id=slides.loc[mask, "slide_id"].to_numpy(dtype=str),
                            targets=np.asarray(result["slide_names"]), values=result["slide_targets"][mask])
        for name, values in result["slide_predictions"].items():
            np.savez_compressed(output / f"slide_predictions_{name}.npz", values=values,
                                slide_id=slides.loc[mask, "slide_id"].to_numpy(dtype=str), targets=np.asarray(result["slide_names"]))
    record = dict(schema="eaf.spatial_biology.v1", config=config, hashes=data["input_hashes"],
                  model_provenance=data["provenance"],
                  signatures=data["audit"], ridge_fits=result["fits"], slide_fits=result["slide_fits"],
                  selection_status="evaluated" if data["selections"] else "skipped_no_selection",
                  niche_status="provided" if "niche" in data["spots"] else "skipped_no_annotations",
                  slide_status="evaluated" if data["slide_matrices"] else "skipped_no_slide_embeddings")
    (output / "protocol.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
