from __future__ import annotations

import contextlib
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TitanAttentionCaptureConfig:
    """Runtime capture policy for TITAN's real post-softmax self-attention.

    ``global_to_tokens`` and ``received`` are O(T) per layer/head. ``rollout``
    keeps one head-averaged T x T matrix per layer while processing a slide.
    ``full`` stores the complete H x T x T matrices and is intentionally gated.
    """

    modes: tuple[str, ...] = ("global_to_tokens", "received", "rollout")
    global_token_index: int = 0
    full_layers: tuple[int, ...] = (-1,)
    max_full_attention_tokens: int = 2048
    max_rollout_tokens: int = 4096
    strict: bool = True

    def __post_init__(self) -> None:
        allowed = {"global_to_tokens", "received", "rollout", "full"}
        unknown = set(self.modes) - allowed
        if unknown:
            raise ValueError(f"Unsupported TITAN attention modes: {sorted(unknown)}")
        if not self.modes:
            raise ValueError("At least one TITAN attention mode is required")


@dataclass
class _CapturedAttention:
    module_name: str
    backend: str
    global_to_tokens: torch.Tensor | None = None
    received: torch.Tensor | None = None
    rollout_matrix: torch.Tensor | None = None
    full: torch.Tensor | None = None
    query_tokens: int = 0
    key_tokens: int = 0


@dataclass(frozen=True)
class TitanAttentionResult:
    attention: Mapping[str, torch.Tensor]
    auxiliary: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]


class _RuntimeAttentionCapture:
    """Scoped capture of post-softmax attention used by the live TITAN model.

    The implementation supports both common paths used by ViTs:

    * fused/SDPA attention via ``torch.nn.functional.scaled_dot_product_attention``;
    * explicit attention via ``Tensor.softmax`` / ``torch.softmax`` / ``F.softmax``.

    No model weights are copied or modified. Original functions and hooks are
    restored when the context exits.
    """

    def __init__(self, model: nn.Module, config: TitanAttentionCaptureConfig) -> None:
        self.model = model
        self.config = config
        self.records: list[_CapturedAttention] = []
        self._handles: list[Any] = []
        self._local = threading.local()
        self._orig_sdpa = F.scaled_dot_product_attention
        self._orig_tensor_softmax = torch.Tensor.softmax
        self._orig_torch_softmax = torch.softmax
        self._orig_f_softmax = F.softmax
        self._call_index = 0

    @staticmethod
    def _is_attention_module(name: str, module: nn.Module) -> bool:
        lower_name = name.lower()
        lower_class = module.__class__.__name__.lower()
        return (
            "attention" in lower_name
            or "attn" in lower_name
            or "attention" in lower_class
            or "attn" in lower_class
            or (hasattr(module, "num_heads") and (hasattr(module, "qkv") or hasattr(module, "q_proj")))
        )

    def _stack(self) -> list[str]:
        stack = getattr(self._local, "module_stack", None)
        if stack is None:
            stack = []
            self._local.module_stack = stack
        return stack

    def _module_name(self) -> str:
        stack = self._stack()
        if stack:
            return stack[-1]
        return f"unscoped_attention_{self._call_index:03d}"

    def _pre_hook(self, name: str) -> Callable:
        def hook(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            self._stack().append(name)

        return hook

    def _post_hook(self, name: str) -> Callable:
        def hook(_module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
            stack = self._stack()
            if stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                stack.remove(name)

        return hook

    @staticmethod
    def _canonical_probs(probs: torch.Tensor) -> torch.Tensor | None:
        if probs.ndim == 3:
            # [B,Q,K] -> one head
            probs = probs.unsqueeze(1)
        if probs.ndim != 4:
            return None
        if probs.shape[-1] < 2 or probs.shape[-2] < 1:
            return None
        return probs

    def _record(self, probs: torch.Tensor, *, backend: str) -> None:
        canonical = self._canonical_probs(probs)
        if canonical is None:
            return
        if not torch.isfinite(canonical).all():
            raise RuntimeError("Captured TITAN attention contains NaN/Inf")

        batch, _heads, query_tokens, key_tokens = canonical.shape
        if batch != 1:
            raise RuntimeError(f"TITAN attention capture currently requires batch=1, got {batch}")
        # TITAN also uses attentional pooling after its ViT blocks.  Its
        # cross-attention has a small query sequence and the same tile keys;
        # it is not a self-attention matrix and cannot be mixed with the ViT
        # layers captured below.
        if query_tokens != key_tokens:
            return
        global_index = self.config.global_token_index
        if not (-query_tokens <= global_index < query_tokens):
            raise IndexError(
                f"global_token_index={global_index} is invalid for {query_tokens} query tokens"
            )
        global_index %= query_tokens

        record = _CapturedAttention(
            module_name=self._module_name(),
            backend=backend,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
        )
        detached = canonical.detach()
        if "global_to_tokens" in self.config.modes:
            record.global_to_tokens = detached[0, :, global_index, :].to("cpu", dtype=torch.float32)
        if "received" in self.config.modes:
            record.received = detached[0].mean(dim=-2).to("cpu", dtype=torch.float32)
        if "rollout" in self.config.modes:
            if query_tokens != key_tokens:
                raise RuntimeError("Attention rollout requires square self-attention matrices")
            if key_tokens > self.config.max_rollout_tokens:
                raise RuntimeError(
                    f"Refusing rollout for T={key_tokens}; raise max_rollout_tokens explicitly"
                )
            record.rollout_matrix = detached[0].mean(dim=0).to("cpu", dtype=torch.float32)
        if "full" in self.config.modes:
            if query_tokens != key_tokens:
                raise RuntimeError("Full TITAN self-attention capture requires square matrices")
            if key_tokens > self.config.max_full_attention_tokens:
                raise RuntimeError(
                    f"Refusing full TITAN attention for T={key_tokens}; "
                    f"limit={self.config.max_full_attention_tokens}"
                )
            record.full = detached[0].to("cpu", dtype=torch.float16)
        self.records.append(record)
        self._call_index += 1

    @staticmethod
    def _apply_sdpa_mask(
        scores: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        if is_causal:
            q, k = scores.shape[-2:]
            causal = torch.ones((q, k), dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if attn_mask is None:
            return scores
        if attn_mask.dtype == torch.bool:
            return scores.masked_fill(~attn_mask, float("-inf"))
        return scores + attn_mask.to(dtype=scores.dtype, device=scores.device)

    def _sdpa_wrapper(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if self.config.strict and dropout_p not in (0, 0.0):
            raise RuntimeError(
                "Exact TITAN attention capture requires model.eval() and SDPA dropout_p=0"
            )
        if enable_gqa and key.shape[-3] != query.shape[-3]:
            repeat = query.shape[-3] // key.shape[-3]
            key_for_scores = key.repeat_interleave(repeat, dim=-3)
        else:
            key_for_scores = key
        factor = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
        scores = torch.matmul(query, key_for_scores.transpose(-2, -1)) * factor
        scores = self._apply_sdpa_mask(scores, attn_mask, is_causal=is_causal)
        probs = self._orig_tensor_softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        self._record(probs, backend="sdpa_qk_exact")
        self._local.inside_sdpa = True
        try:
            kwargs = {
                "attn_mask": attn_mask,
                "dropout_p": dropout_p,
                "is_causal": is_causal,
                "scale": scale,
            }
            if enable_gqa:
                kwargs["enable_gqa"] = True
            try:
                return self._orig_sdpa(query, key, value, **kwargs)
            except TypeError:
                # PyTorch 2.2 builds may not expose enable_gqa.
                kwargs.pop("enable_gqa", None)
                return self._orig_sdpa(query, key, value, **kwargs)
        finally:
            self._local.inside_sdpa = False

    def _capture_softmax_output(self, result: torch.Tensor, *, backend: str) -> None:
        if getattr(self._local, "inside_sdpa", False):
            return
        if not self._stack():
            return
        canonical = self._canonical_probs(result)
        if canonical is None:
            return
        # Explicit ViT attention is [B,H,Q,K]. Restrict to square matrices to
        # avoid capturing unrelated classifier or gating softmax operations.
        if canonical.shape[-2] != canonical.shape[-1]:
            return
        self._record(canonical, backend=backend)

    def _tensor_softmax_wrapper(
        self,
        tensor: torch.Tensor,
        dim: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        result = self._orig_tensor_softmax(tensor, dim=dim, dtype=dtype)
        self._capture_softmax_output(result, backend="tensor_softmax_exact")
        return result

    def _torch_softmax_wrapper(
        self,
        tensor: torch.Tensor,
        dim: int,
        dtype: torch.dtype | None = None,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is not None:
            result = self._orig_torch_softmax(tensor, dim=dim, dtype=dtype, out=out)
        else:
            result = self._orig_torch_softmax(tensor, dim=dim, dtype=dtype)
        self._capture_softmax_output(result, backend="torch_softmax_exact")
        return result

    def _f_softmax_wrapper(
        self,
        input: torch.Tensor,
        dim: int | None = None,
        _stacklevel: int = 3,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        result = self._orig_f_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
        self._capture_softmax_output(result, backend="functional_softmax_exact")
        return result

    def __enter__(self) -> "_RuntimeAttentionCapture":
        for name, module in self.model.named_modules():
            if not name or not self._is_attention_module(name, module):
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            try:
                handle = module.register_forward_hook(self._post_hook(name), always_call=True)
            except TypeError:  # PyTorch versions before always_call support.
                handle = module.register_forward_hook(self._post_hook(name))
            self._handles.append(handle)

        capture = self

        def tensor_softmax(
            tensor: torch.Tensor,
            dim: int | None = None,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            return capture._tensor_softmax_wrapper(tensor, dim=dim, dtype=dtype)

        F.scaled_dot_product_attention = self._sdpa_wrapper
        torch.Tensor.softmax = tensor_softmax
        torch.softmax = self._torch_softmax_wrapper
        F.softmax = self._f_softmax_wrapper
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        F.scaled_dot_product_attention = self._orig_sdpa
        torch.Tensor.softmax = self._orig_tensor_softmax
        torch.softmax = self._orig_torch_softmax
        F.softmax = self._orig_f_softmax
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()


def _select_layers(values: Sequence[torch.Tensor], requested: Sequence[int]) -> list[tuple[int, torch.Tensor]]:
    n_layers = len(values)
    selected: list[tuple[int, torch.Tensor]] = []
    seen: set[int] = set()
    for raw in requested:
        index = raw + n_layers if raw < 0 else raw
        if not (0 <= index < n_layers):
            raise IndexError(f"Requested attention layer {raw}, but captured {n_layers} layers")
        if index not in seen:
            selected.append((index, values[index]))
            seen.add(index)
    return selected


def _attention_rollout(matrices: Sequence[torch.Tensor], global_token_index: int) -> torch.Tensor:
    if not matrices:
        raise RuntimeError("No attention matrices available for rollout")
    tokens = matrices[0].shape[-1]
    rollout = torch.eye(tokens, dtype=torch.float32)
    for matrix in matrices:
        if matrix.shape != (tokens, tokens):
            raise RuntimeError("Rollout matrices have inconsistent token counts")
        augmented = matrix.float() + torch.eye(tokens, dtype=torch.float32)
        augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = augmented @ rollout
    index = global_token_index % tokens
    return rollout[index]


def _find_patch_shape(model: nn.Module) -> tuple[int, int] | None:
    for name, module in model.named_modules():
        if "patch_embed" not in name.lower():
            continue
        patch_size = getattr(module, "patch_size", None)
        if patch_size is not None:
            if isinstance(patch_size, int):
                return patch_size, patch_size
            if isinstance(patch_size, Sequence) and len(patch_size) >= 2:
                return int(patch_size[0]), int(patch_size[1])
        projection = getattr(module, "proj", None)
        kernel = getattr(projection, "kernel_size", None)
        if kernel is not None:
            if isinstance(kernel, int):
                return kernel, kernel
            return int(kernel[0]), int(kernel[1])
    return None


def infer_titan_tile_to_token(
    coords: torch.Tensor,
    *,
    token_count: int,
    patch_size_level0: int,
    model: nn.Module,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Infer the mapping from input CONCH tiles to TITAN vision tokens.

    The direct N or N+1 cases are exact. For TITAN's dense-grid path, the
    function uses the live patch-embedding patch size and validates that the
    inferred spatial token count equals the captured attention token count.
    If validation fails, no mapping is emitted; the real token attention is
    still retained for inspection.
    """

    coords_cpu = coords.detach().to("cpu", dtype=torch.int64)
    n_tiles = int(coords_cpu.shape[0])
    if token_count == n_tiles:
        return torch.arange(n_tiles, dtype=torch.int64), {
            "tile_token_mapping": "direct_no_prefix",
            "prefix_tokens": 0,
        }
    if token_count == n_tiles + 1:
        return torch.arange(1, n_tiles + 1, dtype=torch.int64), {
            "tile_token_mapping": "direct_one_prefix",
            "prefix_tokens": 1,
        }

    patch_shape = _find_patch_shape(model)
    if patch_shape is None:
        return None, {"tile_token_mapping": "unavailable_no_patch_shape"}
    ph, pw = patch_shape
    xy = coords_cpu[:, :2]
    origin = xy.min(dim=0).values
    grid_xy = torch.div(xy - origin, patch_size_level0, rounding_mode="floor")
    grid_w = int(grid_xy[:, 0].max().item()) + 1
    grid_h = int(grid_xy[:, 1].max().item()) + 1

    candidates: list[tuple[str, int, int]] = []
    candidates.append(("ceil", math.ceil(grid_h / ph), math.ceil(grid_w / pw)))
    candidates.append(("floor", grid_h // ph, grid_w // pw))
    for method, out_h, out_w in candidates:
        if out_h <= 0 or out_w <= 0:
            continue
        spatial_tokens = out_h * out_w
        prefix = token_count - spatial_tokens
        if prefix < 0 or prefix > 8:
            continue
        token_y = torch.div(grid_xy[:, 1], ph, rounding_mode="floor")
        token_x = torch.div(grid_xy[:, 0], pw, rounding_mode="floor")
        valid = (token_y < out_h) & (token_x < out_w)
        if not bool(valid.all()):
            continue
        mapping = prefix + token_y * out_w + token_x
        if int(mapping.max().item()) >= token_count:
            continue
        return mapping.to(torch.int64), {
            "tile_token_mapping": f"dense_grid_{method}_validated",
            "prefix_tokens": prefix,
            "tile_grid_height": grid_h,
            "tile_grid_width": grid_w,
            "vision_patch_height": ph,
            "vision_patch_width": pw,
            "vision_token_rows": out_h,
            "vision_token_cols": out_w,
            "grid_origin_x": int(origin[0].item()),
            "grid_origin_y": int(origin[1].item()),
        }
    return None, {
        "tile_token_mapping": "unavailable_token_count_mismatch",
        "captured_token_count": token_count,
        "tile_grid_height": grid_h,
        "tile_grid_width": grid_w,
        "vision_patch_height": ph,
        "vision_patch_width": pw,
    }


def _map_token_values_to_tiles(
    values: torch.Tensor,
    tile_to_token: torch.Tensor,
    *,
    mass_share: bool,
) -> torch.Tensor:
    mapped = values[..., tile_to_token]
    if not mass_share:
        return mapped
    token_count = values.shape[-1]
    counts = torch.bincount(tile_to_token, minlength=token_count).clamp_min(1).to(mapped.dtype)
    return mapped / counts[tile_to_token]


def capture_titan_attention(
    model: nn.Module,
    forward: Callable[[], torch.Tensor],
    *,
    coords: torch.Tensor,
    patch_size_level0: int,
    config: TitanAttentionCaptureConfig,
) -> tuple[torch.Tensor, TitanAttentionResult]:
    """Run TITAN once and return its real post-softmax attention tensors."""

    with _RuntimeAttentionCapture(model, config) as capture:
        embedding = forward()

    if not capture.records:
        diagnostic = [
            f"{name}: {module.__class__.__module__}.{module.__class__.__name__}"
            for name, module in model.named_modules()
            if _RuntimeAttentionCapture._is_attention_module(name, module)
        ]
        raise RuntimeError(
            "No TITAN attention matrix was captured. The installed checkpoint may use an unsupported "
            "attention backend. Candidate modules:\n" + "\n".join(diagnostic[:100])
        )

    token_counts = {(record.query_tokens, record.key_tokens) for record in capture.records}
    if len(token_counts) != 1:
        raise RuntimeError(f"TITAN layers returned inconsistent attention shapes: {sorted(token_counts)}")
    query_tokens, key_tokens = next(iter(token_counts))
    if query_tokens != key_tokens:
        raise RuntimeError("Expected TITAN self-attention with Q=K")

    attention: dict[str, torch.Tensor] = {}
    auxiliary: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {
        "attention_is_native": True,
        "attention_source": "real_post_softmax_runtime_capture",
        "attention_layers": len(capture.records),
        "attention_tokens": key_tokens,
        "attention_capture_backends": ",".join(record.backend for record in capture.records),
        "attention_module_names": ",".join(record.module_name for record in capture.records),
        "global_token_index": config.global_token_index,
    }

    if "global_to_tokens" in config.modes:
        matrices = [record.global_to_tokens for record in capture.records]
        if any(value is None for value in matrices):
            raise RuntimeError("Incomplete global-to-token capture")
        attention["global_to_tokens"] = torch.stack([value for value in matrices if value is not None])
    if "received" in config.modes:
        matrices = [record.received for record in capture.records]
        if any(value is None for value in matrices):
            raise RuntimeError("Incomplete received-attention capture")
        attention["received_by_tokens"] = torch.stack([value for value in matrices if value is not None])
    if "rollout" in config.modes:
        matrices = [record.rollout_matrix for record in capture.records]
        if any(value is None for value in matrices):
            raise RuntimeError("Incomplete rollout capture")
        attention["rollout_global_to_tokens"] = _attention_rollout(
            [value for value in matrices if value is not None], config.global_token_index
        )
    if "full" in config.modes:
        matrices = [record.full for record in capture.records]
        if any(value is None for value in matrices):
            raise RuntimeError("Incomplete full-attention capture")
        full_values = [value for value in matrices if value is not None]
        for layer, value in _select_layers(full_values, config.full_layers):
            attention[f"full_layer_{layer:03d}"] = value

    tile_to_token, mapping_metadata = infer_titan_tile_to_token(
        coords,
        token_count=key_tokens,
        patch_size_level0=patch_size_level0,
        model=model,
    )
    metadata.update(mapping_metadata)
    if tile_to_token is not None:
        auxiliary["tile_to_token"] = tile_to_token
        counts = torch.bincount(tile_to_token, minlength=key_tokens)
        auxiliary["tiles_per_token"] = counts
        if "global_to_tokens" in attention:
            token_values = attention["global_to_tokens"]
            attention["global_to_tiles_broadcast"] = _map_token_values_to_tiles(
                token_values, tile_to_token, mass_share=False
            )
            attention["global_to_tiles_mass_share"] = _map_token_values_to_tiles(
                token_values, tile_to_token, mass_share=True
            )
        if "received_by_tokens" in attention:
            token_values = attention["received_by_tokens"]
            attention["received_by_tiles_broadcast"] = _map_token_values_to_tiles(
                token_values, tile_to_token, mass_share=False
            )
        if "rollout_global_to_tokens" in attention:
            token_values = attention["rollout_global_to_tokens"]
            attention["rollout_global_to_tiles_broadcast"] = _map_token_values_to_tiles(
                token_values, tile_to_token, mass_share=False
            )
            attention["rollout_global_to_tiles_mass_share"] = _map_token_values_to_tiles(
                token_values, tile_to_token, mass_share=True
            )

    return embedding, TitanAttentionResult(attention, auxiliary, metadata)
