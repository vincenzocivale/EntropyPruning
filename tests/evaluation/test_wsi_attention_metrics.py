import pytest
import torch

from src.evaluation.wsi_attention_metrics import (
    wsi_attention_ndcg_at_k,
    wsi_attention_spearmanr,
    wsi_attention_topk_overlap,
)


def test_wsi_attention_spearmanr_is_one_for_perfect_ranking() -> None:
    scores = torch.tensor([0.1, 0.2, 0.4, 0.8], dtype=torch.float32)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    rho = wsi_attention_spearmanr(scores, target)

    assert torch.isclose(rho, torch.tensor(1.0))


def test_wsi_attention_spearmanr_is_minus_one_for_reversed_ranking() -> None:
    scores = torch.tensor([0.8, 0.4, 0.2, 0.1], dtype=torch.float32)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    rho = wsi_attention_spearmanr(scores, target)

    assert torch.isclose(rho, torch.tensor(-1.0))


def test_wsi_attention_spearmanr_supports_masked_batched_inputs() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, 99.0],
            [0.3, 0.2, 0.1, 99.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 2.0, 3.0, 1000.0],
            [1.0, 2.0, 3.0, 1000.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
        ]
    )

    rho = wsi_attention_spearmanr(scores, target, mask)

    assert torch.isclose(rho, torch.tensor(0.0))


def test_wsi_attention_topk_overlap_is_one_for_matching_topk() -> None:
    scores = torch.tensor([0.1, 0.9, 0.8, 0.2], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0, 3.0, 2.0], dtype=torch.float32)

    overlap = wsi_attention_topk_overlap(scores, target, k=2)

    assert torch.isclose(overlap, torch.tensor(1.0))


def test_wsi_attention_topk_overlap_handles_k_larger_than_number_of_valid_tiles() -> None:
    scores = torch.tensor([0.1, 0.9, 0.8], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0, 3.0], dtype=torch.float32)

    overlap = wsi_attention_topk_overlap(scores, target, k=10)

    assert torch.isclose(overlap, torch.tensor(1.0))


def test_wsi_attention_topk_overlap_supports_mask() -> None:
    scores = torch.tensor([0.9, 0.1, 100.0], dtype=torch.float32)
    target = torch.tensor([4.0, 1.0, 1000.0], dtype=torch.float32)
    mask = torch.tensor([True, True, False])

    overlap = wsi_attention_topk_overlap(scores, target, k=1, mask=mask)

    assert torch.isclose(overlap, torch.tensor(1.0))


def test_wsi_attention_ndcg_at_k_is_one_for_ideal_ranking() -> None:
    scores = torch.tensor([0.1, 0.9, 0.8, 0.2], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0, 3.0, 2.0], dtype=torch.float32)

    ndcg = wsi_attention_ndcg_at_k(scores, target, k=3)

    assert torch.isclose(ndcg, torch.tensor(1.0))


def test_wsi_attention_ndcg_at_k_penalizes_bad_ranking() -> None:
    scores = torch.tensor([0.9, 0.1, 0.2, 0.3], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0, 3.0, 2.0], dtype=torch.float32)

    ndcg = wsi_attention_ndcg_at_k(scores, target, k=3)

    assert 0.0 <= ndcg.item() < 1.0


def test_wsi_attention_metrics_reject_shape_mismatch() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    target = torch.tensor([1.0, 2.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="same shape"):
        wsi_attention_topk_overlap(scores, target, k=1)


def test_wsi_attention_metrics_reject_non_boolean_mask() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    mask = torch.tensor([1, 1, 0], dtype=torch.long)

    with pytest.raises(TypeError, match="boolean"):
        wsi_attention_topk_overlap(scores, target, k=1, mask=mask)


def test_wsi_attention_metrics_reject_negative_target_attention() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    target = torch.tensor([1.0, -1.0, 3.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="non-negative"):
        wsi_attention_ndcg_at_k(scores, target, k=2)


def test_wsi_attention_metrics_reject_non_positive_k() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="positive"):
        wsi_attention_topk_overlap(scores, target, k=0)

    with pytest.raises(ValueError, match="positive"):
        wsi_attention_ndcg_at_k(scores, target, k=0)
