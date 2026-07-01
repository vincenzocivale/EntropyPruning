import pytest
import torch

from src.data.wsi import WSIRankingStore, WSITileRanking


def _make_ranking(slide_id: str = "slide_001") -> WSITileRanking:
    scores = torch.tensor([0.2, 0.9, 0.5, 0.7], dtype=torch.float32)
    ranks = torch.tensor([4, 1, 3, 2], dtype=torch.int64)
    coords = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=torch.int64,
    )
    selected_indices = {
        "0.25": torch.tensor([1], dtype=torch.int64),
        "0.5": torch.tensor([1, 3], dtype=torch.int64),
    }
    original_order_indices = {
        "0.25": torch.tensor([1], dtype=torch.int64),
        "0.5": torch.tensor([1, 3], dtype=torch.int64),
    }
    metadata = {
        "ranking_checkpoint": "checkpoints/model.pt",
        "ranking_keep_ratios": [0.25, 0.5],
        "ranking_ranks_are_1_based": True,
    }
    return WSITileRanking(
        slide_id=slide_id,
        scores=scores,
        ranks=ranks,
        coords=coords,
        selected_indices=selected_indices,
        original_order_indices=original_order_indices,
        metadata=metadata,
    )


def test_wsi_ranking_store_npz_roundtrip(tmp_path) -> None:
    store = WSIRankingStore(tmp_path, file_format="npz")
    ranking = _make_ranking()

    path = store.write(ranking)
    loaded = store.read("slide_001")

    assert path.name == "slide_001.npz"
    assert store.slide_ids() == ("slide_001",)
    assert loaded.slide_id == ranking.slide_id
    assert torch.equal(loaded.scores, ranking.scores)
    assert torch.equal(loaded.ranks, ranking.ranks)
    assert loaded.coords is not None
    assert torch.equal(loaded.coords, ranking.coords)
    assert loaded.metadata == ranking.metadata
    assert torch.equal(loaded.selected_indices["0.5"], torch.tensor([1, 3]))
    assert torch.equal(loaded.original_order_indices["0.5"], torch.tensor([1, 3]))


def test_wsi_ranking_store_supports_slide_ids_with_slashes(tmp_path) -> None:
    store = WSIRankingStore(tmp_path, file_format="npz")
    ranking = _make_ranking("patient_001/slide_A")

    store.write(ranking)

    assert store.exists("patient_001/slide_A")
    loaded = store.read("patient_001/slide_A")
    assert loaded.slide_id == "patient_001/slide_A"


def test_wsi_tile_ranking_rejects_selected_indices_not_sorted_by_score() -> None:
    with pytest.raises(ValueError, match="ordered by descending score"):
        WSITileRanking(
            slide_id="slide_001",
            scores=torch.tensor([0.2, 0.9, 0.5, 0.7], dtype=torch.float32),
            ranks=torch.tensor([4, 1, 3, 2], dtype=torch.int64),
            selected_indices={"0.5": torch.tensor([3, 1], dtype=torch.int64)},
            original_order_indices={"0.5": torch.tensor([1, 3], dtype=torch.int64)},
            metadata={},
        )


def test_wsi_tile_ranking_rejects_non_json_metadata() -> None:
    with pytest.raises(TypeError, match="JSON-serializable"):
        WSITileRanking(
            slide_id="slide_001",
            scores=torch.tensor([0.1, 0.2], dtype=torch.float32),
            ranks=torch.tensor([2, 1], dtype=torch.int64),
            metadata={"bad": object()},
        )
