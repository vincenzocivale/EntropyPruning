from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from idc_index import IDCClient


WSI_ROOT = Path(
    "/data2/home/vcivale/projects/imaging/data/WSI"
)

CATALOG_ROOT = WSI_ROOT / "catalog"

DATASET_ROOT = (
    WSI_ROOT
    / "datasets/downstream/wsi_level"
    / "eaf_thunder_nsclc_ood_clean_v1"
)

MANIFEST_ROOT = DATASET_ROOT / "manifests"


# ------------------------------------------------------------
# Collezioni ammesse nel corpus strict non-polmonare.
# ------------------------------------------------------------

STRICT_CORE = {
    "cptac_brca",
    "cptac_ccrcc",
    "cptac_coad",
    "cptac_gbm",
    "cptac_hnscc",
    "cptac_ov",
    "cptac_pda",
    "cptac_ucec",
    "cptac_sar",
    "cptac_cm",
}


# AML è mantenuta separata perché può contenere preparati
# ematologici/midollari con caratteristiche diverse dalle WSI
# H&E solide. Deve superare un audit visivo prima dell'inclusione.
OPTIONAL_AFTER_QC = {
    "cptac_aml",
}


# Collezioni distinte da TCGA, ma riservate perché vogliamo una
# valutazione WSI LUAD-vs-LUSC fortemente OOD.
RESERVED_NSCLC = {
    "cptac_luad",
    "cptac_lscc",
    "nlst",
}


CANDIDATE_COLLECTIONS = (
    STRICT_CORE
    | OPTIONAL_AFTER_QC
    | RESERVED_NSCLC
)


# ------------------------------------------------------------
# Benchmark firewall.
# ------------------------------------------------------------

BENCHMARK_RULES = [
    {
        "benchmark_id": "thunder_tcga_uniform",
        "benchmark_level": "tile",
        "source_family": "TCGA",
        "blocked_collection": "tcga_*",
        "exclusion_scope": "all_tcga",
        "reason": (
            "TCGA Uniform contains patches from diagnostic "
            "slides across 32 TCGA cancer types."
        ),
    },
    {
        "benchmark_id": "thunder_tcga_tils",
        "benchmark_level": "tile",
        "source_family": "TCGA",
        "blocked_collection": "tcga_*",
        "exclusion_scope": "all_tcga",
        "reason": (
            "TCGA TILS contains patches from multiple TCGA "
            "cancer types."
        ),
    },
    {
        "benchmark_id": "thunder_tcga_crc_msi",
        "benchmark_level": "tile",
        "source_family": "TCGA",
        "blocked_collection": "tcga_coad|tcga_read",
        "exclusion_scope": "collection_and_patient",
        "reason": "THUNDER TCGA CRC-MSI benchmark.",
    },
    {
        "benchmark_id": "thunder_camelyon17_wilds",
        "benchmark_level": "tile",
        "source_family": "CAMELYON",
        "blocked_collection": "camelyon17",
        "exclusion_scope": "entire_collection",
        "reason": "THUNDER Camelyon17-WILDS benchmark.",
    },
    {
        "benchmark_id": "thunder_patch_camelyon",
        "benchmark_level": "tile",
        "source_family": "CAMELYON",
        "blocked_collection": "camelyon16|patch_camelyon",
        "exclusion_scope": "entire_collection",
        "reason": "THUNDER PatchCamelyon benchmark.",
    },
    {
        "benchmark_id": "wsi_tcga_nsclc",
        "benchmark_level": "wsi",
        "source_family": "TCGA",
        "blocked_collection": "tcga_luad|tcga_lusc",
        "exclusion_scope": "entire_collection",
        "reason": "Primary WSI LUAD-vs-LUSC benchmark.",
    },
    {
        "benchmark_id": "wsi_nsclc_external",
        "benchmark_level": "wsi",
        "source_family": "CPTAC|NLST",
        "blocked_collection": (
            "cptac_luad|cptac_lscc|nlst"
        ),
        "exclusion_scope": "entire_collection",
        "reason": (
            "Reserved external/OOD lung pathology collections."
        ),
    },
    {
        "benchmark_id": "wsi_tcga_crc",
        "benchmark_level": "wsi",
        "source_family": "TCGA",
        "blocked_collection": "tcga_coad|tcga_read",
        "exclusion_scope": "entire_collection",
        "reason": (
            "EAGLE-parity WSI CRC benchmark, TCGA arm "
            "(arXiv:2502.13027)."
        ),
    },
    {
        "benchmark_id": "wsi_crc_external",
        "benchmark_level": "wsi",
        "source_family": "CPTAC",
        "blocked_collection": "cptac_coad",
        "exclusion_scope": "entire_collection",
        "reason": (
            "EAGLE-parity external WSI CRC benchmark, CPTAC arm. "
            "DACHS (paper's third CRC arm, n=3604) is a private "
            "DKFZ cohort and is not reproduced here."
        ),
    },
    {
        "benchmark_id": "wsi_tcga_brca",
        "benchmark_level": "wsi",
        "source_family": "TCGA",
        "blocked_collection": "tcga_brca",
        "exclusion_scope": "entire_collection",
        "reason": (
            "EAGLE-parity WSI BRCA benchmark, TCGA arm "
            "(arXiv:2502.13027)."
        ),
    },
    {
        "benchmark_id": "wsi_brca_external",
        "benchmark_level": "wsi",
        "source_family": "CPTAC",
        "blocked_collection": "cptac_brca",
        "exclusion_scope": "entire_collection",
        "reason": (
            "EAGLE-parity external WSI BRCA benchmark, CPTAC arm. "
            "IEO Milan (paper's third BRCA arm, n=451) is a "
            "private cohort and is not reproduced here."
        ),
    },
    {
        "benchmark_id": "wsi_tcga_stad",
        "benchmark_level": "wsi",
        "source_family": "TCGA",
        "blocked_collection": "tcga_stad",
        "exclusion_scope": "entire_collection",
        "reason": (
            "EAGLE-parity WSI STAD benchmark, TCGA arm "
            "(arXiv:2502.13027). No public external STAD arm "
            "exists; the paper's Bern/Kiel cohorts are private "
            "and are not reproduced here."
        ),
    },
]


def collection_policy(collection_id: str) -> tuple[str, str]:
    if collection_id in STRICT_CORE:
        return "include", ""

    if collection_id in OPTIONAL_AFTER_QC:
        return (
            "quarantine",
            "Requires stain and specimen-type quality control",
        )

    if collection_id in RESERVED_NSCLC:
        return (
            "reserve",
            "Reserved for NSCLC WSI evaluation",
        )

    return "exclude", "Not approved by strict corpus policy"


def main() -> None:
    CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    client = IDCClient.client()

    quoted = ", ".join(
        f"'{collection}'"
        for collection in sorted(CANDIDATE_COLLECTIONS)
    )

    query = f"""
    SELECT
        collection_id,
        PatientID,
        StudyInstanceUID,
        SeriesInstanceUID,
        Modality,
        SeriesDescription,
        series_size_MB
    FROM index
    WHERE Modality = 'SM'
      AND collection_id IN ({quoted})
    """

    inventory = client.sql_query(query)

    if not isinstance(inventory, pd.DataFrame):
        inventory = pd.DataFrame(inventory)

    if inventory.empty:
        raise RuntimeError(
            "La query IDC non ha restituito serie Slide Microscopy."
        )

    inventory = inventory.drop_duplicates(
        subset=["SeriesInstanceUID"]
    ).copy()

    policies = inventory["collection_id"].map(
        collection_policy
    )

    inventory["pretraining_policy"] = [
        policy for policy, _ in policies
    ]
    inventory["exclusion_reason"] = [
        reason for _, reason in policies
    ]

    inventory["labels_used"] = False
    inventory["source_provider"] = "IDC/TCIA"
    inventory["source_format"] = "DICOM-SM"
    inventory["raw_downloaded"] = False
    inventory["original_svs_resolved"] = False

    inventory = inventory.sort_values(
        [
            "pretraining_policy",
            "collection_id",
            "PatientID",
            "SeriesInstanceUID",
        ]
    )

    inventory_path = (
        MANIFEST_ROOT / "idc_sm_inventory.csv"
    )
    inventory.to_csv(inventory_path, index=False)

    eligible = inventory[
        inventory["pretraining_policy"] == "include"
    ].copy()

    reserved = inventory[
        inventory["pretraining_policy"] == "reserve"
    ].copy()

    quarantine = inventory[
        inventory["pretraining_policy"] == "quarantine"
    ].copy()

    eligible.to_csv(
        MANIFEST_ROOT / "eligible_sm_series.csv",
        index=False,
    )
    reserved.to_csv(
        MANIFEST_ROOT / "reserved_sm_series.csv",
        index=False,
    )
    quarantine.to_csv(
        MANIFEST_ROOT / "quarantine_sm_series.csv",
        index=False,
    )

    benchmark_path = (
        CATALOG_ROOT / "benchmark_registry.csv"
    )

    with benchmark_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(BENCHMARK_RULES[0].keys()),
        )
        writer.writeheader()
        writer.writerows(BENCHMARK_RULES)

    summary = (
        inventory.groupby(
            ["collection_id", "pretraining_policy"],
            dropna=False,
        )
        .agg(
            patients=("PatientID", "nunique"),
            studies=("StudyInstanceUID", "nunique"),
            slide_series=("SeriesInstanceUID", "nunique"),
            size_gib=("series_size_MB", lambda x: x.sum() / 1024),
        )
        .reset_index()
        .sort_values(
            ["pretraining_policy", "collection_id"]
        )
    )

    summary.to_csv(
        MANIFEST_ROOT / "collection_summary.csv",
        index=False,
    )

    print("=== IDC SLIDE MICROSCOPY INVENTORY ===")
    print(summary.to_string(index=False))

    print("\n=== STRICT CORPUS ===")
    print(
        "Eligible patients:",
        eligible["PatientID"].nunique(),
    )
    print(
        "Eligible slide series:",
        eligible["SeriesInstanceUID"].nunique(),
    )
    print(
        "Estimated size GiB:",
        round(eligible["series_size_MB"].sum() / 1024, 2),
    )

    print("\n=== RESERVED NSCLC ===")
    print(
        "Reserved patients:",
        reserved["PatientID"].nunique(),
    )
    print(
        "Reserved slide series:",
        reserved["SeriesInstanceUID"].nunique(),
    )

    print("\nInventory:", inventory_path)
    print("Benchmark registry:", benchmark_path)


if __name__ == "__main__":
    main()
