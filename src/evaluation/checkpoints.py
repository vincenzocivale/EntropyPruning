"""Shared checkpoint loaders for frozen downstream evaluation."""
from __future__ import annotations

from pathlib import Path
import torch


def _load_student(args, device: torch.device):
    from thunder.models.pretrained_models import get_model_from_name
    from src.models import AttentionForecaster, ThunderBackboneAdapter
    from src.models.online_tile_eaf import PrunedLoRAEncoder, unwrap_checkpoint_state
    payload = torch.load(args.pruned_checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model_name = args.model_name or config.get("model_name") or payload.get("base_model")
    if not model_name:
        raise ValueError("Could not resolve tile model name from CLI/checkpoint")
    raw, transform, _ = get_model_from_name(model_name, str(device))
    raw = raw.to(device)
    adapter = ThunderBackboneAdapter(raw, transform=transform)

    forecaster_path = args.forecaster_checkpoint or payload.get("forecaster_checkpoint")
    if not forecaster_path:
        raise ValueError("Could not resolve forecaster checkpoint")
    fpayload = torch.load(forecaster_path, map_location="cpu", weights_only=False)
    fcfg = fpayload.get("config", fpayload.get("args", {}))
    forecaster = AttentionForecaster(
        embed_dim=adapter.embed_dim,
        hidden=int(config.get("hidden", fcfg.get("hidden", 256))),
        n_heads=int(config.get("n_heads", fcfg.get("n_heads", 4))),
        n_layers=int(config.get("n_layers", fcfg.get("n_layers", 2))),
        dropout=0.0,
    )
    forecaster.load_state_dict(unwrap_checkpoint_state(fpayload), strict=True)
    forecaster = forecaster.to(device).eval()

    student = PrunedLoRAEncoder(
        raw,
        adapter,
        forecaster,
        prune_layer=int(config["prune_layer"]),
        keep_ratio=float(config["keep_ratio"]),
        lora_r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
    ).to(device)
    student.load_trainable_state_dict(payload["trainable_state_dict"])
    student.eval()
    return student, transform, model_name, dict(config, resolved_forecaster_checkpoint=str(forecaster_path))


def _load_wsi_forecaster(checkpoint_path: Path, *, device: torch.device):
    """Load a frozen WSI forecaster for downstream inference."""
    from src.models.wsi.dense_forecaster import WSIDenseForecaster, WSIDenseForecasterALiBi
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fargs = payload["args"]
    common = dict(embed_dim=768, hidden=fargs["hidden"], n_heads=fargs["n_heads"], n_layers=fargs["n_layers"], dropout=0.0)
    if fargs.get("architecture") == "dense_alibi":
        forecaster = WSIDenseForecasterALiBi(**common)
    else:
        forecaster = WSIDenseForecaster(**common)
    forecaster.load_state_dict(payload["model"])
    forecaster = forecaster.to(device).eval()
    source_layer = fargs.get("titan_hidden_layer")
    if source_layer is None:
        raise ValueError(f"{checkpoint_path} was not trained with --input-source titan_hidden")
    return forecaster, int(source_layer)


def _load_pruned_titan(
    pruned_checkpoint: Path, *, device: torch.device, hf_token: str | None
) -> tuple[PrunedLoRATitanEncoder, dict]:
    from src.models.wsi.pruned_titan import PrunedLoRATitanEncoder
    from src.wsi_pipeline.wsi_models.titan import TitanAdapter
    payload = torch.load(pruned_checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("args", {})
    forecaster_ckpt = config.get("forecaster_checkpoint")
    if not forecaster_ckpt:
        raise ValueError(f"{pruned_checkpoint} does not record a forecaster_checkpoint")
    forecaster, prune_layer = _load_wsi_forecaster(Path(forecaster_ckpt), device=device)
    titan_model = TitanAdapter(token=hf_token).model
    student = PrunedLoRATitanEncoder(
        titan_model,
        forecaster,
        prune_layer=int(config.get("prune_layer") if config.get("prune_layer") is not None else prune_layer),
        keep_ratio=float(config["keep_ratio"]),
        patch_size_level0=int(config.get("patch_size_level0", 512)),
        lora_r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
    ).to(device)
    student.load_trainable_state_dict(payload["model"])
    student.eval()
    meta = {
        "run_name": payload.get("run_name", pruned_checkpoint.parent.name),
        "prune_layer": student.prune_layer,
        "keep_ratio": student.keep_ratio,
        "resolved_forecaster_checkpoint": str(forecaster_ckpt),
    }
    return student, meta
