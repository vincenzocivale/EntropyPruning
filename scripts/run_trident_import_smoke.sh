#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

WORKDIR="${WORKDIR:-/tmp/eaf_trident_import_smoke}"
FEATURE_DIM="${FEATURE_DIM:-8}"
N_SLIDES="${N_SLIDES:-4}"
MIN_TILES="${MIN_TILES:-4}"
SEED="${SEED:-123}"

TRIDENT_ROOT="${WORKDIR}/trident_processed/20x_256px_0px_overlap"
FEATURES_DIR="${TRIDENT_ROOT}/features_uni_v1"
COORDS_DIR="${TRIDENT_ROOT}/patches"
LABELS_CSV="${WORKDIR}/labels.csv"
MANIFEST_CSV="${WORKDIR}/manifest_trident.csv"
OUTPUT_STORE="${WORKDIR}/features_trident_eaf.h5"
INSPECTION_JSON="${WORKDIR}/feature_store_inspection.json"

rm -rf "${WORKDIR}"
mkdir -p "${FEATURES_DIR}" "${COORDS_DIR}"

echo "[trident-smoke] workdir=${WORKDIR}"
echo "[trident-smoke] creating fake TRIDENT HDF5 outputs"

"${PYTHON_BIN}" - <<PY
from pathlib import Path

import h5py
import numpy as np

workdir = Path("${WORKDIR}")
features_dir = Path("${FEATURES_DIR}")
coords_dir = Path("${COORDS_DIR}")
labels_csv = Path("${LABELS_CSV}")

feature_dim = int("${FEATURE_DIM}")
n_slides = int("${N_SLIDES}")
min_tiles = int("${MIN_TILES}")
seed = int("${SEED}")

rng = np.random.default_rng(seed)

with labels_csv.open("w") as handle:
    handle.write("slide_id,label\\n")

    for index in range(n_slides):
        slide_id = f"slide_{index:03d}"
        n_tiles = min_tiles + index
        label = index % 2

        features = rng.normal(size=(n_tiles, feature_dim)).astype("float32")
        coords = np.stack(
            [
                np.arange(n_tiles, dtype="int64") * 256,
                np.arange(n_tiles, dtype="int64") * 128,
            ],
            axis=1,
        )

        with h5py.File(features_dir / f"{slide_id}.h5", "w") as h5:
            h5.create_dataset("features", data=features)

        with h5py.File(coords_dir / f"{slide_id}.h5", "w") as h5:
            h5.create_dataset("coords", data=coords)

        handle.write(f"{slide_id},{label}\\n")
PY

echo "[trident-smoke] building manifest"
"${PYTHON_BIN}" scripts/build_trident_manifest.py \
  --features-dir "${FEATURES_DIR}" \
  --coords-dir "${COORDS_DIR}" \
  --labels-csv "${LABELS_CSV}" \
  --output-manifest "${MANIFEST_CSV}" \
  --require-coords \
  --require-labels \
  --overwrite

echo "[trident-smoke] importing manifest into EAF HDF5 feature store"
"${PYTHON_BIN}" scripts/import_trident_feature_store.py \
  --manifest "${MANIFEST_CSV}" \
  --output-feature-store "${OUTPUT_STORE}" \
  --feature-dim "${FEATURE_DIM}" \
  --overwrite

echo "[trident-smoke] validating imported EAF feature store"
"${PYTHON_BIN}" scripts/validate_wsi_feature_store.py \
  --feature-store "${OUTPUT_STORE}" \
  --feature-dim "${FEATURE_DIM}" \
  --require-coords

echo "[trident-smoke] inspecting imported EAF feature store"
"${PYTHON_BIN}" scripts/inspect_wsi_feature_store.py \
  --feature-store "${OUTPUT_STORE}" \
  --output-json "${INSPECTION_JSON}"

cat <<EOF
[trident-smoke] done
workdir: ${WORKDIR}
features_dir: ${FEATURES_DIR}
coords_dir: ${COORDS_DIR}
labels_csv: ${LABELS_CSV}
manifest_csv: ${MANIFEST_CSV}
output_store: ${OUTPUT_STORE}
inspection_json: ${INSPECTION_JSON}
EOF
