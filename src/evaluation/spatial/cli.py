"""Registry-bound command line for external spatial biology evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", default="hest_biological_conch15_titan")
    parser.add_argument("--variant-id", default="final")
    parser.add_argument("--experiment-registry", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    from .protocol import evaluate, load_config, save_outputs, validate_inputs
    config = load_config(args.config)
    if args.command == "validate":
        data = validate_inputs(config)
        print(json.dumps({"status": "valid_inputs_not_launch_authorization", "spots": len(data["spots"]),
                          "patients": data["spots"].patient_id.nunique(), "programs": data["names"],
                          "signature_audit": data["audit"]}, indent=2))
        return 0
    from src.wsi_pipeline.experiment_registry import load_registry, prepare_experiment_run, repository_root
    from src.wsi_pipeline.experiment_results import publish_run_summary
    registry, _, _ = load_registry(args.experiment_registry)
    entry = registry.get("experiments", {}).get(args.experiment_id, {})
    # Freeze the scientific protocol in the registry, beyond merely recording it.
    if entry.get("status") != "ready":
        raise RuntimeError(f"Experiment {args.experiment_id} is blocked/not ready: {entry.get('blockers', [])}")
    from .data import sha256
    if entry.get("protocol_sha256") != sha256(args.config):
        raise ValueError("Register protocol_sha256 of the frozen config before launch")
    import os
    root_value = args.data_root or os.environ.get("EAF_WSI_ROOT")
    if not root_value:
        raise ValueError("Set EAF_WSI_ROOT or --data-root")
    root = Path(root_value).expanduser().resolve()
    if root == repository_root() or repository_root() in root.parents:
        raise ValueError("Runtime artifacts cannot be stored inside the repository")
    data = validate_inputs(config)
    args.spatial_config = config
    # Protect completed scientific artifacts from accidental reruns.
    result_dir = root / "results/wsi_eaf/evaluation" / args.experiment_id / args.variant_id / f"seed_{args.seed}"
    if (result_dir / "summary.json").exists():
        raise FileExistsError(f"Completed run already exists: {result_dir}")
    run = prepare_experiment_run(args, family="wsi_eaf", stage="evaluation")
    result = evaluate(config, data, seed=args.seed)
    save_outputs(run.result_dir, config, data, result)
    if not args.no_plots:
        from .report import write_report
        write_report(run.result_dir, config, data, result)
    summary = {"protocol": str(run.result_dir / "protocol.json"), "input_hashes": data["input_hashes"],
               "programs": data["names"], "test_patients": result["test_spots"].patient_id.nunique(),
               "test_spots": len(result["test_spots"]), "output_csv": str(run.result_dir / "patient_summary.csv")}
    publish_run_summary(run=run, args=args, summary=summary)
    (run.log_dir / "run.log").write_text(json.dumps(summary, indent=2) + "\n")
    print(run.result_dir)
    return 0
