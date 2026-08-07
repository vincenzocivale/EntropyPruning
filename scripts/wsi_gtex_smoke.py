#!/usr/bin/env python3
"""Plan/download a tiny GTEx DICOM WSI smoke set from NCI IDC.

GTEx is intentionally NOT added to strict-v1 automatically. Run this smoke test
first and verify that the TRIDENT/OpenSlide environment can read the downloaded
DICOM WSI before scaling to ~1,500 slides.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise SystemExit("Install dependency: pip install -U idc-index") from exc

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    client = IDCClient()
    client.fetch_index("sm_index")
    query = f"""
    SELECT DISTINCT
      i.PatientID AS donor_id,
      i.StudyInstanceUID,
      i.SeriesInstanceUID,
      sm.primaryAnatomicStructure_CodeMeaning AS tissue
    FROM index AS i
    JOIN sm_index AS sm
      ON i.SeriesInstanceUID = sm.SeriesInstanceUID
    WHERE lower(i.collection_id) = 'gtex'
      AND i.Modality = 'SM'
    ORDER BY donor_id, tissue, i.SeriesInstanceUID
    LIMIT {int(args.n)}
    """
    df = client.sql_query(query)
    manifest = out / "gtex_smoke.csv"
    df.to_csv(manifest, index=False)
    print(f"[eaf-wsi-data] GTEx smoke manifest: {manifest} ({len(df)} series)")

    if args.download:
        for uid in df["SeriesInstanceUID"].tolist():
            print(f"[eaf-wsi-data] downloading GTEx series {uid}", flush=True)
            client.download_dicom_series(seriesInstanceUID=uid, downloadDir=str(out / "dicom"))
        print(f"[eaf-wsi-data] downloaded GTEx smoke set under {out / 'dicom'}")
    else:
        print("[eaf-wsi-data] plan only; add --download to fetch pixels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
