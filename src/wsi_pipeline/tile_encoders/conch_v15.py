from __future__ import annotations

from typing import Any

import torch

from .base import TileEncoderAdapter, TileEncoderOutput
from .hooks import extract_cls_token, find_transformer_blocks


class ConchV15MultiLayerEncoder(TileEncoderAdapter):
    """CONCH v1.5 adapter extracting block 2 and final embeddings in one forward."""

    name = "conch_v15"
    input_size = 512

    def __init__(
        self,
        *,
        model_id: str = "MahmoodLab/TITAN",
        token: str | None = None,
        intermediate_block_index: int = 1,
    ) -> None:
        from transformers import AutoModel

        titan = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token,
        )
        conch, transform = titan.return_conch()
        self.model = conch
        self._transform = transform
        self.intermediate_block_index = intermediate_block_index
        blocks = find_transformer_blocks(self.model)
        if not 0 <= intermediate_block_index < len(blocks):
            raise IndexError(f"intermediate_block_index={intermediate_block_index}, number of blocks={len(blocks)}")
        self._captured: Any = None
        self._hook = blocks[intermediate_block_index].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self._captured = output

    @property
    def transform(self):
        return self._transform

    def to(self, device: torch.device) -> "ConchV15MultiLayerEncoder":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "ConchV15MultiLayerEncoder":
        self.model.eval()
        return self

    def encode(self, images: torch.Tensor) -> TileEncoderOutput:
        self._captured = None
        with torch.inference_mode():
            final = self.model.encode_image(images, proj_contrast=False, normalize=False)
        if self._captured is None:
            raise RuntimeError("Intermediate block hook did not fire")
        layer2 = extract_cls_token(self._captured, batch_size=images.shape[0])
        return TileEncoderOutput({"layer2": layer2, "final": final})
