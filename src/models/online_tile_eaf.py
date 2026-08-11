"""Online teacher extraction and pruning-aware tile-encoder adaptation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from peft.tuners.lora import LoraModel

from .backbone_adapter import ThunderBackboneAdapter


def unwrap_checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    """Return a state dictionary from common checkpoint wrappers."""
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "trainable_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError("Checkpoint does not contain a recognizable state dict")


def load_checkpoint_flexibly(
    module: nn.Module,
    path: str | Path,
    *,
    min_match_fraction: float = 0.25,
) -> tuple[list[str], list[str]]:
    """Load checkpoints saved through common EAF/PEFT wrapper prefixes."""
    payload = torch.load(path, map_location="cpu")
    state = unwrap_checkpoint_state(payload)
    target = module.state_dict()
    prefixes = (
        "module.",
        "model.",
        "backbone.model.",
        "backbone.",
        "raw_backbone.",
    )
    candidates: list[dict[str, torch.Tensor]] = [state]
    for prefix in prefixes:
        stripped = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if stripped:
            candidates.append(stripped)

    def score(candidate: dict[str, torch.Tensor]) -> int:
        return sum(
            key in target and tuple(target[key].shape) == tuple(value.shape)
            for key, value in candidate.items()
        )

    best = max(candidates, key=score)
    matched = score(best)
    fraction = matched / max(len(target), 1)
    if fraction < min_match_fraction:
        raise RuntimeError(
            f"Only {matched}/{len(target)} tensors matched {path}; "
            "the checkpoint likely belongs to another encoder/wrapper"
        )
    missing, unexpected = module.load_state_dict(best, strict=False)
    return list(missing), list(unexpected)


def pooled_embedding(output: Any) -> torch.Tensor:
    """Normalize common timm/Thunder outputs to ``[batch, dimension]``."""
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "cls_token", "embedding", "features", "x"):
            if key in output:
                output = output[key]
                break
        else:
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if not tensors:
                raise TypeError("Backbone output dictionary contains no tensor")
            output = tensors[0]
    if isinstance(output, (tuple, list)):
        output = next((value for value in output if torch.is_tensor(value)), None)
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported backbone output type: {type(output)!r}")
    if output.ndim == 3:
        output = output[:, 0]
    if output.ndim != 2:
        raise ValueError(f"Expected [B,D] or [B,N,D], got {tuple(output.shape)}")
    return output


class OnlineAttentionTeacher:
    """Extract source block outputs and final CLS-to-patch attention in memory.

    The source representation is captured *after* ``source_layer`` because that
    is exactly the tensor scored by the deployed pruning hook. No embeddings or
    attention maps are copied to CPU or written to disk.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ThunderBackboneAdapter,
        source_layer: int,
        target_layer: int,
    ) -> None:
        self.backbone = backbone
        self.adapter = adapter
        self.source_layer = source_layer
        self.target_layer = target_layer
        if not 0 <= source_layer < adapter.n_blocks:
            raise ValueError(f"Invalid source layer {source_layer}")
        if not 0 <= target_layer < adapter.n_blocks:
            raise ValueError(f"Invalid target layer {target_layer}")

    def _target_attention(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = module.qkv(x).reshape(
            batch, tokens, 3, module.num_heads, module.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        q_norm = getattr(module, "q_norm", nn.Identity())
        k_norm = getattr(module, "k_norm", nn.Identity())
        q = q_norm(q)
        k = k_norm(k)
        attention_logits = (q @ k.transpose(-2, -1) * module.scale).float()
        attention = attention_logits.softmax(dim=-1)
        prefix = self.adapter.num_prefix_tokens
        target = attention[:, :, 0, prefix:].mean(dim=1)
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cache: dict[str, torch.Tensor] = {}
        prefix = self.adapter.num_prefix_tokens
        source_block = self.adapter.get_blocks()[self.source_layer]
        target_attention = self.adapter.get_attn_module(self.target_layer)

        def source_hook(
            _: nn.Module,
            __: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Source block returned unsupported type {type(output)!r}"
                )
            cache["source"] = output[:, prefix:].detach()

        def target_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            cache["target"] = self._target_attention(module, inputs[0]).detach()

        handles = [
            source_block.register_forward_hook(source_hook),
            target_attention.register_forward_pre_hook(target_hook),
        ]
        try:
            if hasattr(self.backbone, "forward_features"):
                self.backbone.forward_features(images)
            else:
                self.backbone(images)
        finally:
            for handle in handles:
                handle.remove()
        if "source" not in cache or "target" not in cache:
            raise RuntimeError("Teacher hooks did not observe the requested layers")
        source = cache["source"]
        target = cache["target"]
        if source.shape[:2] != target.shape:
            raise RuntimeError(
                f"Source/target token mismatch: {tuple(source.shape)} vs "
                f"{tuple(target.shape)}"
            )
        return source, target

    @torch.no_grad()
    def extract_early(self, images: torch.Tensor) -> torch.Tensor:
        """Return source-layer patch tokens and abort before later blocks execute."""
        cache: dict[str, torch.Tensor] = {}
        prefix = self.adapter.num_prefix_tokens
        source_block = self.adapter.get_blocks()[self.source_layer]

        class _EarlyExit(Exception):
            pass

        def source_hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Source block returned unsupported type {type(output)!r}"
                )
            cache["source"] = output[:, prefix:].detach()
            raise _EarlyExit()

        handle = source_block.register_forward_hook(source_hook)
        try:
            try:
                if hasattr(self.backbone, "forward_features"):
                    self.backbone.forward_features(images)
                else:
                    self.backbone(images)
            except _EarlyExit:
                pass
        finally:
            handle.remove()
        if "source" not in cache:
            raise RuntimeError("Early-exit teacher hook did not observe source layer")
        return cache["source"]


class PrunedLoRAEncoder(nn.Module):
    """LoRA tile encoder with vectorized forecaster-guided token pruning.

    A single backbone serves both roles during adaptation:

    * ``full_teacher_embedding`` disables LoRA and pruning and runs the frozen
      base encoder in evaluation mode;
    * ``forward`` enables LoRA and applies EAF pruning.

    This avoids keeping a second foundation model in GPU memory.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: ThunderBackboneAdapter,
        forecaster: nn.Module,
        *,
        prune_layer: int,
        keep_ratio: float,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if not 0 <= prune_layer < adapter.n_blocks:
            raise ValueError(f"Invalid prune layer {prune_layer}")
        self.adapter = adapter
        self.forecaster = forecaster
        self.prune_layer = prune_layer
        self.keep_ratio = keep_ratio
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.backbone = LoraModel(backbone, config, adapter_name="default")
        self.forecaster.eval()
        for parameter in self.forecaster.parameters():
            parameter.requires_grad_(False)

    @property
    def raw_backbone(self) -> nn.Module:
        # The adapter resolves THUNDER wrappers (for example TITAN/CONCH v1.5)
        # to the actual timm trunk. PEFT injects LoRA modules in-place, so this
        # remains the same live transformer after wrapping.
        return self.adapter.model

    def train(self, mode: bool = True) -> "PrunedLoRAEncoder":
        super().train(mode)
        # The ranking function is a frozen teacher and must never activate dropout.
        self.forecaster.eval()
        return self

    @contextmanager
    def _pruning_hook(self) -> Iterator[None]:
        block = self.raw_backbone.blocks[self.prune_layer]
        original_forward = block.forward
        prefix_count = self.adapter.num_prefix_tokens

        def pruned_forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            x = original_forward(x, *args, **kwargs)
            prefix = x[:, :prefix_count]
            patches = x[:, prefix_count:]
            n_patches = patches.shape[1]
            keep = max(1, int(round(n_patches * self.keep_ratio)))
            with torch.no_grad():
                scores = self.forecaster(patches)
                indices = scores.topk(keep, dim=-1).indices
            gather_index = indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
            kept = torch.gather(patches, dim=1, index=gather_index)
            return torch.cat((prefix, kept), dim=1)

        block.forward = pruned_forward
        try:
            yield
        finally:
            block.forward = original_forward

    @torch.no_grad()
    def full_teacher_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Run the frozen, unpruned base encoder without allocating another model."""
        was_training = self.backbone.training
        self.backbone.eval()
        self.backbone.disable_adapter_layers()
        try:
            output = self.backbone(images)
        finally:
            self.backbone.enable_adapter_layers()
            self.backbone.train(was_training)
            self.forecaster.eval()
        return pooled_embedding(output).detach()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with self._pruning_hook():
            output = self.backbone(images)
        return pooled_embedding(output)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only LoRA tensors, never the full frozen backbone."""
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        parameters = dict(self.named_parameters())
        unknown = sorted(set(state) - set(parameters))
        if unknown:
            raise KeyError(f"Unknown trainable keys: {unknown[:5]}")
        missing = sorted(
            name
            for name, parameter in parameters.items()
            if parameter.requires_grad and name not in state
        )
        if missing:
            raise KeyError(f"Missing trainable keys: {missing[:5]}")
        with torch.no_grad():
            for name, value in state.items():
                parameter = parameters[name]
                if tuple(parameter.shape) != tuple(value.shape):
                    raise ValueError(
                        f"Shape mismatch for {name}: {tuple(value.shape)} != "
                        f"{tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def embedding_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    cosine_weight: float = 1.0,
    mse_weight: float = 1.0,
    pairwise_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Task-agnostic full-vs-pruned representation distillation objective."""
    student_n = F.normalize(student.float(), dim=-1)
    teacher_n = F.normalize(teacher.float(), dim=-1)
    cosine = 1.0 - (student_n * teacher_n).sum(dim=-1).mean()
    mse = F.mse_loss(student_n, teacher_n)
    student_similarity = student_n @ student_n.transpose(0, 1)
    teacher_similarity = teacher_n @ teacher_n.transpose(0, 1)
    pairwise = F.mse_loss(student_similarity, teacher_similarity)
    total = cosine_weight * cosine + mse_weight * mse + pairwise_weight * pairwise
    return total, {"cosine": cosine, "mse": mse, "pairwise": pairwise}
