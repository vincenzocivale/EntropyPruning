from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import transforms

from .base import TileEncoderAdapter, TileEncoderOutput
from .hooks import extract_cls_token, find_transformer_blocks


@dataclass(frozen=True)
class TimmPreprocess:
    input_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class TimmViTMultiLayerEncoder(TileEncoderAdapter):
    """Generic timm ViT adapter for UNI/UNI2, Virchow, GigaPath tile, H-optimus, etc."""

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool = True,
        intermediate_block_index: int = 1,
        preprocess: TimmPreprocess = TimmPreprocess(),
    ) -> None:
        import timm

        self.name = model_name
        self.input_size = preprocess.input_size
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self._transform = transforms.Compose(
            [
                transforms.Resize((preprocess.input_size, preprocess.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(preprocess.mean, preprocess.std),
            ]
        )
        blocks = find_transformer_blocks(self.model)
        self._captured = None
        self._hook = blocks[intermediate_block_index].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self._captured = output

    @property
    def transform(self):
        return self._transform

    def to(self, device: torch.device) -> "TimmViTMultiLayerEncoder":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "TimmViTMultiLayerEncoder":
        self.model.eval()
        return self

    def encode(self, images: torch.Tensor) -> TileEncoderOutput:
        self._captured = None
        with torch.inference_mode():
            final = self.model(images)
        if isinstance(final, (tuple, list)):
            final = final[0]
        if final.ndim == 3:
            final = final[:, 0, :]
        layer2 = extract_cls_token(self._captured, batch_size=images.shape[0])
        return TileEncoderOutput({"layer2": layer2, "final": final})
