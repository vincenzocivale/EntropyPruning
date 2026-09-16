#!/usr/bin/env python
"""Fetch TCGA labels for the EAGLE benchmark replication (Phase D) from the
public cBioPortal REST API (no auth) and write them in the
<labels-root>/<cohort>/labels/<task>.csv format (slide_id,label) that
scripts/evaluation/evaluate_wsi_eagle.py's discover_tasks() expects.

Scope: core metadata via API; the benchmark workflow is in docs/pipeline.md -- clinical/
mutation attributes exposed as structured cBioPortal fields only. Tasks whose
labels only exist in a paper's supplementary tables (STAD EBV/Lauren) are out
of scope here; BRCA hormone-receptor status (ESR1/PGR/ERBB2) is also out of
scope for the TCGA arm (not present in the PanCanAtlas 2018 clinical fields --
in the EAGLE paper these are CPTAC-arm tasks; get them from the already-
downloaded Patho-Bench splits instead, see the CPTAC labels script).

Joins on patient barcode (first 12 chars of a slide_id), against every audited
MPP-valid slide for each TCGA cohort (manifests/<PROJECT>_custom_list_of_wsis.csv).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import requests

API = "https://www.cbioportal.org/api"
EAF_WSI_ROOT = Path("/data2/home/vcivale/projects/imaging/data/WSI")
TCGA_MANIFESTS = EAF_WSI_ROOT / "datasets/downstream/wsi_level/eagle_tcga_v1/manifests"
# evaluate_wsi_eagle.py's discover_tasks() scans
# <labels-root>/<cohort>/labels/*.csv directly under the default labels-root
# (<data-root>/datasets/downstream/wsi_level) -- cohort dirs go straight
# there, not nested under eagle_tcga_v1/.
LABELS_ROOT = EAF_WSI_ROOT / "datasets/downstream/wsi_level"

GENES = {"BRAF": 673, "KRAS": 3845, "PIK3CA": 5290, "EGFR": 1956, "STK11": 6794, "TP53": 7157}
MSI_THRESHOLD = 10.0  # MSIsensor score >= 10 -> MSI-H, standard cutoff

STUDIES = {
    "TCGA-COAD": "coadread_tcga_pan_can_atlas_2018",
    "TCGA-READ": "coadread_tcga_pan_can_atlas_2018",
    "TCGA-BRCA": "brca_tcga_pan_can_atlas_2018",
    "TCGA-STAD": "stad_tcga_pan_can_atlas_2018",
    "TCGA-LUAD": "luad_tcga_pan_can_atlas_2018",
    "TCGA-LUSC": "lusc_tcga_pan_can_atlas_2018",
}


def load_audited_slides(project: str) -> dict[str, str]:
    """slide_id (stem) -> patient_id (first 12 chars of TCGA barcode)."""
    csv_path = TCGA_MANIFESTS / f"{project}_custom_list_of_wsis.csv"
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            fname = row["wsi"].split("/")[-1]
            slide_id = fname[:-4] if fname.endswith(".svs") else fname
            out[slide_id] = slide_id[:12]
    return out


def fetch_clinical(study: str, attribute_ids: list[str]) -> dict[str, dict[str, str]]:
    """patient_id -> {attribute_id: value}. Some attributes (e.g.
    MSI_SENSOR_SCORE) are SAMPLE-scoped rather than PATIENT-scoped in
    cBioPortal, so both are queried and merged; a sample-level value is
    attached to its patient (fine here since these are one-sample-per-
    patient TCGA cohorts for our purposes)."""
    out: dict[str, dict[str, str]] = {}
    for data_type in ("PATIENT", "SAMPLE"):
        resp = requests.get(
            f"{API}/studies/{study}/clinical-data",
            params={"clinicalDataType": data_type, "projection": "SUMMARY"},
        )
        resp.raise_for_status()
        for row in resp.json():
            if row["clinicalAttributeId"] not in attribute_ids:
                continue
            out.setdefault(row["patientId"], {}).setdefault(row["clinicalAttributeId"], row["value"])
    return out


def fetch_mutations(study: str, entrez_id: int) -> set[str]:
    """Patients with a mutation in this gene."""
    resp = requests.post(
        f"{API}/molecular-profiles/{study}_mutations/mutations/fetch",
        json={"entrezGeneIds": [entrez_id], "sampleListId": f"{study}_all"},
    )
    resp.raise_for_status()
    return {row["patientId"] for row in resp.json()}


def write_task(project: str, task: str, slide_to_label: dict[str, int]) -> None:
    out_dir = LABELS_ROOT / project / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide_id", "label"])
        for slide_id, label in sorted(slide_to_label.items()):
            w.writerow([slide_id, label])
    print(f"  {project}/{task}.csv: {len(slide_to_label)} labeled slides")


def main() -> int:
    for project, study in STUDIES.items():
        print(f"=== {project} ({study}) ===")
        slide_to_patient = load_audited_slides(project)
        patients = set(slide_to_patient.values())
        print(f"  {len(slide_to_patient)} audited slides, {len(patients)} patients")

        clinical = fetch_clinical(
            study, ["MSI_SENSOR_SCORE", "PATH_M_STAGE", "PATH_N_STAGE", "ICD_O_3_SITE"]
        )

        # Sidedness (CRC): ICD-O-3 site C18.0-C18.4 = right colon (cecum through
        # hepatic flexure), C18.5-C18.7 = left colon (splenic flexure through
        # sigmoid), C19-C20 = rectum -- excluded, sidedness is a colon-only concept.
        if project in ("TCGA-COAD", "TCGA-READ"):
            sidedness = {}
            for slide_id, pid in slide_to_patient.items():
                site = clinical.get(pid, {}).get("ICD_O_3_SITE", "")
                if not site.startswith("C18"):
                    continue
                try:
                    subsite = float(site[1:])
                except ValueError:
                    continue
                sidedness[slide_id] = 0 if subsite <= 18.4 else 1
            if sidedness:
                write_task(project, "sidedness", sidedness)

        # MSI status (binary), CRC + STAD
        if project in ("TCGA-COAD", "TCGA-READ", "TCGA-STAD"):
            msi = {}
            for slide_id, pid in slide_to_patient.items():
                score = clinical.get(pid, {}).get("MSI_SENSOR_SCORE")
                if score is None:
                    continue
                try:
                    msi[slide_id] = int(float(score) >= MSI_THRESHOLD)
                except ValueError:
                    continue
            if msi:
                write_task(project, "msi_status", msi)

        # M-status (M0 vs M1), N-status (N0 vs N+)
        m_status, n_status = {}, {}
        for slide_id, pid in slide_to_patient.items():
            m = clinical.get(pid, {}).get("PATH_M_STAGE", "")
            if m in ("M0", "CM0", "PM0"):
                m_status[slide_id] = 0
            elif m in ("M1", "CM1", "PM1"):
                m_status[slide_id] = 1
            n = clinical.get(pid, {}).get("PATH_N_STAGE", "")
            if n in ("N0",):
                n_status[slide_id] = 0
            elif n and n.startswith("N") and n[1] in "123":
                n_status[slide_id] = 1
        if m_status:
            write_task(project, "m_status", m_status)
        if n_status:
            write_task(project, "n_status", n_status)

        # Gene mutations
        gene_map = {
            "TCGA-COAD": ["BRAF", "KRAS"],
            "TCGA-READ": ["BRAF", "KRAS"],
            "TCGA-BRCA": ["PIK3CA"],
            "TCGA-STAD": ["TP53"],
            "TCGA-LUAD": ["EGFR", "STK11", "TP53"],
            "TCGA-LUSC": ["EGFR", "STK11", "TP53"],
        }
        for gene in gene_map.get(project, []):
            mutated_patients = fetch_mutations(study, GENES[gene])
            labels = {
                slide_id: int(pid in mutated_patients)
                for slide_id, pid in slide_to_patient.items()
            }
            write_task(project, f"{gene.lower()}_mutation", labels)

    # NSCLC subtyping: trivial, no API needed -- label is which cohort the slide is from.
    luad_slides = load_audited_slides("TCGA-LUAD")
    lusc_slides = load_audited_slides("TCGA-LUSC")
    subtyping_luad = {s: 0 for s in luad_slides}
    subtyping_lusc = {s: 1 for s in lusc_slides}
    write_task("TCGA-LUAD", "nsclc_subtyping", subtyping_luad)
    write_task("TCGA-LUSC", "nsclc_subtyping", subtyping_lusc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
