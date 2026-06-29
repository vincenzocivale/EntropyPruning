#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

WORKDIR="${WORKDIR:-/tmp/eaf_generic_import_smoke}"
FEATURE_DIM="${FEATURE_DIM:-8}"
N_SLIDES="${N_SLIDES:-4}"
MIN_TILES="${MIN_TILES:-4}"
SEED="${SEED:-123}"

FEATURES_DIR="${WORKDIR}/features"
COORDS_DIR="${WORKDIR}/coords"
LABELS_CSV="${WORKDIR}/labels.csv"
MANIFEST_CSV="${WORKDIR}/manifest_generic.csv"
OUTPUT_STORE="${WORKDIR}/features_generic_eaf.h5"
INSPECTION_JSON="${WORKDIR}/feature_store_inspection.json"

rm -rf "${WORKDIR}"
mkdir -p "${FEATURES_DIR}" "${COORDS_DIR}"

echo "[generic-smoke] workdir=${WORKDIR}"
echo "[generic-smoke] creating fake generic feature files"

"${PYTHON_BIN}" - <<PY
from pathlib import Path

import numpy as np
import torch

features_dir = Path("${FEATURES_DIR}")
coords_dir = Path("${COORDS_DIR}")
labels_csv = Path("${LABELS_CSV}")

feature_dim = int("${FEATURE_DIM}")
n_slides = int("${N_SLIDES}")
min_tiles = int("${MIN_TILES}")
seed = int("${SEED}")

rng = np.random.default_rng(seed)
torch.manual_seed(seed)

suffixes = [".pt", ".npy", ".npz"]

with labels_csv.open("w") as handle:
    handle.write("slide_id,label\\n")

    for index in range(n_slides):
        slide_id = f"slide_{index:03d}"
        n_tiles = min_tiles + index
        label = index % 2
        suffix = suffixes[index % len(suffixes)]

        features = rng.normal(size=(n_tiles, feature_dim)).astype("float32")
        coords = np.stack(
            [
                np.arange(n_tiles, dtype="int64") * 256,
                np.arange(n_tiles, dtype="int64") * 128,
            ],
            axis=1,
        )

        if suffix == ".pt":
            torch.save(torch.from_numpy(features), features_dir / f"{slide_id}.pt")
            torch.save(torch.from_numpy(coords), coords_dir / f"{slide_id}.pt")
        elif suffix == ".npy":
            np.save(features_dir / f"{slide_id}.npy", features)
            np.save(coords_dir / f"{slide_id}.npy", coords)
        elif suffix == ".npz":
            np.savez(features_dir / f"{slide_id}.npz", features=features)
            np.savez(coords_dir / f"{slide_id}.npz", coords=coords)
        else:
            raise AssertionError(suffix)

        handle.write(f"{slide_id},{label}\\n")
PY

echo "[generic-smoke] building manifest"
"${PYTHON_BIN}" scripts/build_generic_feature_manifest.py \
  --features-dir "${FEATURES_DIR}" \
  --coords-dir "${COORDS_DIR}" \
  --labels-csv "${LABELS_CSV}" \
  --output-manifest "${MANIFEST_CSV}" \
  --require-coords \
  --require-labels \
  --overwrite

echo "[generic-smoke] importing manifest into EAF HDF5 feature store"
"${PYTHON_BIN}" scripts/import_generic_feature_store.py \
  --manifest "${MANIFEST_CSV}" \
  --output-feature-store "${OUTPUT_STORE}" \
  --feature-dim "${FEATURE_DIM}" \
  --overwrite

echo "[generic-smoke] validating imported EAF feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${OUTPUT_STORE}" \
  --feature-dim "${FEATURE_DIM}" \
  --require-coords

echo "[generic-smoke] inspecting imported EAF feature store"
"${PYTHON_BIN}" scripts/inspect_wsi_feature_store.py \
  --feature-store "${OUTPUT_STORE}" \
  --output-json "${INSPECTION_JSON}"

cat <<EOF
[generic-smoke] done
workdir: ${WORKDIR}
features_dir: ${FEATURES_DIR}
coords_dir: ${COORDS_DIR}
labels_csv: ${LABELS_CSV}
manifest_csv: ${MANIFEST_CSV}
output_store: ${OUTPUT_STORE}
inspection_json: ${INSPECTION_JSON}
EOF
