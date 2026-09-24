"""Frozen EAF cache export; operates on prepared patches / existing WSI caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import read_spots, sha256


def export_tile_pair(student, transform, spots, *, device, batch_size=32):
    """Patch paths are explicit; no ST expression is loaded by the encoder."""
    import torch
    from PIL import Image
    import h5py
    stores = {}

    def load_image(row):
        if getattr(row, "patch_store", None):
            store = str(row.patch_store)
            if store not in stores:
                stores[store] = h5py.File(store, "r")
            index = int(row.patch_index)
            return Image.fromarray(np.asarray(stores[store]["img"][index])).convert("RGB")
        with Image.open(row.patch_path) as image:
            return image.convert("RGB")

    if "patch_path" not in spots and "patch_store" not in spots:
        raise ValueError("Tile extraction requires patch_path or patch_store for every spot")
    full, pruned = [], []
    timings = {"full_seconds": 0.0, "eaf_seconds": 0.0}
    student.eval()
    with torch.inference_mode():
        for start in range(0, len(spots), batch_size):
            images = []
            for row in spots.iloc[start:start + batch_size].itertuples(index=False):
                image = load_image(row)
                images.append(transform(image))
            batch = torch.stack(images).to(device)
            for name, fn, destination in (("full", student.full_teacher_embedding, full), ("eaf", student, pruned)):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                before = time.perf_counter()
                embedding = fn(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings[f"{name}_seconds"] += time.perf_counter() - before
                destination.append(embedding.float().cpu().numpy())
    return np.concatenate(full), np.concatenate(pruned), timings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=("tile", "wsi"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pruned-checkpoint", type=Path, required=True)
    parser.add_argument("--forecaster-checkpoint", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    from src.wsi_pipeline.experiment_registry import add_experiment_arguments, prepare_experiment_run, repository_root
    add_experiment_arguments(parser)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    import os
    root_value = args.data_root or os.environ.get("EAF_WSI_ROOT")
    if not root_value:
        raise ValueError("Set EAF_WSI_ROOT or --data-root")
    root = Path(root_value).expanduser().resolve()
    if root == repository_root() or repository_root() in root.parents:
        raise ValueError("Runtime artifacts cannot be stored inside repository")
    # Export and evaluation have the same scientific identity, but separate
    # manifests: preserve an existing evaluation run.json rather than overwriting.
    from src.wsi_pipeline.experiment_registry import load_registry
    registry, _, _ = load_registry(args.experiment_registry)
    entry = registry.get("experiments", {}).get(args.experiment_id, {})
    if entry.get("status") != "ready":
        raise RuntimeError(f"Experiment {args.experiment_id} is blocked/not ready")
    result_dir = root / "results/wsi_eaf/evaluation" / args.experiment_id / args.variant_id / f"seed_{args.seed}"
    if (result_dir / "summary.json").exists():
        raise FileExistsError("Cannot export into a completed evaluation")
    out = root / "caches/experiments" / args.experiment_id / args.variant_id / f"seed_{args.seed}" / "spatial" / args.level
    if out.exists():
        raise FileExistsError(f"Export directory already exists: {out}")
    run = prepare_experiment_run(args, family="wsi_eaf", stage="evaluation")
    out.mkdir(parents=True)
    model_pair = entry.get("tile_encoder", "conch_v15")
    if args.level == "wsi":
        model_pair += "_" + entry.get("wsi_encoder", "titan")
    teacher_out = root / "caches/spatial" / entry.get("dataset_id", "hest") / model_pair / args.level
    teacher_out.mkdir(parents=True, exist_ok=True)
    import torch
    from src.evaluation.checkpoints import _load_pruned_titan, _load_student
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    provenance = {"manifest_sha256": sha256(args.manifest), "checkpoint_sha256": sha256(args.pruned_checkpoint),
                  "checkpoint": str(args.pruned_checkpoint.resolve()), "level": args.level,
                  "device": str(device), "timing_scope": "encoder forward only; includes first batch; not an end-to-end benchmark"}
    if args.level == "tile":
        spots = read_spots(args.manifest)
        if "patch_path" in spots:
            spots["patch_path"] = spots.patch_path.map(lambda p: str((args.manifest.resolve().parent / p).resolve()))
        student, transform, model_name, checkpoint_config = _load_student(args, device)
        provenance["checkpoint_config"] = checkpoint_config
        provenance["forecaster_sha256"] = sha256(checkpoint_config["resolved_forecaster_checkpoint"])
        full, pruned, timings = export_tile_pair(student, transform, spots, device=device, batch_size=args.batch_size)
        _save_full(teacher_out, spots.spot_id.to_numpy(dtype=str), full, "spot_id")
        np.savez_compressed(out / "eaf.npz", spot_id=spots.spot_id.to_numpy(dtype=str), embeddings=pruned)
        provenance.update(timings, model_name=model_name)
    else:
        from src.wsi_pipeline.numpy_store import read_array
        from src.wsi_pipeline.io import read_wsi_output_record
        table = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
        required = {"slide_id", "source_path", "teacher_path"}
        if required - set(table) or table.empty or table.slide_id.duplicated().any():
            raise ValueError("WSI manifest requires unique slide_id, source_path, teacher_path")
        student, meta = _load_pruned_titan(args.pruned_checkpoint, device=device, hf_token=None)
        provenance["forecaster_sha256"] = sha256(meta["resolved_forecaster_checkpoint"])
        full, pruned, rectangles = [], [], []
        for row in table.itertuples(index=False):
            source = (args.manifest.resolve().parent / row.source_path).resolve()
            teacher = (args.manifest.resolve().parent / row.teacher_path).resolve()
            if source == teacher:
                raise ValueError("Student-source and full teacher caches must be distinct")
            coords = read_array(source, "coords")
            features = read_array(source, "tile_embeddings")
            target = read_wsi_output_record(teacher)
            if target.coords is None:
                raise ValueError("Full teacher cache must include coordinates for grid verification")
            if target.slide_id != row.slide_id:
                raise ValueError("Full teacher slide ID does not match WSI manifest")
            teacher_coords = np.asarray(target.coords)
            if set(map(tuple, coords)) != set(map(tuple, teacher_coords)):
                raise ValueError("Student source / teacher grids differ")
            embedding, indices = student.encode_with_selection(torch.as_tensor(features, device=device).float(),
                                                               torch.as_tensor(coords, device=device).long())
            full.append(np.asarray(target.slide_embedding).reshape(-1))
            pruned.append(embedding.float().cpu().numpy())
            kept = set(indices.cpu().tolist())
            size = student.patch_size_level0
            rectangles.extend(dict(slide_id=row.slide_id, x=int(x), y=int(y), width=size, height=size, kept=int(i in kept))
                              for i, (x, y) in enumerate(np.asarray(coords)[:, :2]))
        _save_full(teacher_out, table.slide_id.to_numpy(dtype=str), np.stack(full), "slide_id")
        np.savez_compressed(out / "eaf.npz", slide_id=table.slide_id.to_numpy(dtype=str), embeddings=np.stack(pruned))
        pd.DataFrame(rectangles).to_csv(out / "selection.csv", index=False)
        provenance.update(meta)
    provenance["full_cache"] = str(teacher_out / "full.npz")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    (out / "run.json").write_bytes((run.result_dir / "run.json").read_bytes())
    print(out)
    return 0


def _save_full(directory, ids, values, key):
    """Full teachers are reusable dataset/model-centric artifacts, never replaced."""
    from .data import aligned_matrix
    path = directory / "full.npz"
    if path.exists():
        previous = aligned_matrix(path, ids, id_key=key)
        if not np.allclose(previous, values, rtol=1e-4, atol=1e-5):
            raise ValueError(f"Existing teacher cache differs: {path}; use a separately registered dataset/model identity")
        return
    np.savez_compressed(path, **{key: ids, "embeddings": values})
