import torch

from src.wsi_pipeline.tile_encoders.hooks import extract_cls_token


def test_extract_cls_batch_first():
    value = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    torch.testing.assert_close(extract_cls_token(value, 2), value[:, 0, :])


def test_extract_cls_sequence_first():
    value = torch.arange(3 * 2 * 4).reshape(3, 2, 4)
    torch.testing.assert_close(extract_cls_token(value, 2), value[0, :, :])
