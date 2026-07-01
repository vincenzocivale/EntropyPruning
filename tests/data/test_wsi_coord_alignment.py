import pytest
import torch

from src.data.wsi import align_by_coords


def test_align_by_coords_matches_permuted_coords() -> None:
    input_coords = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_coords = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]])

    input_indices, target_indices = align_by_coords(input_coords, target_coords)

    assert torch.equal(input_indices, torch.arange(4))
    assert torch.equal(input_coords[input_indices], target_coords[target_indices])
    assert target_indices.tolist() == [1, 3, 2, 0]


def test_align_by_coords_rejects_non_2d_coords() -> None:
    with pytest.raises(ValueError, match="2D"):
        align_by_coords(torch.zeros(4), torch.zeros(4, 2))


def test_align_by_coords_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="same coordinate dimension"):
        align_by_coords(torch.zeros(4, 2), torch.zeros(4, 4))


def test_align_by_coords_rejects_duplicate_input_coords() -> None:
    input_coords = torch.tensor([[0, 0], [0, 0], [1, 1]])
    target_coords = torch.tensor([[0, 0], [1, 1], [2, 2]])

    with pytest.raises(ValueError, match="input coords contain duplicate"):
        align_by_coords(input_coords, target_coords)


def test_align_by_coords_rejects_duplicate_target_coords() -> None:
    input_coords = torch.tensor([[0, 0], [1, 1]])
    target_coords = torch.tensor([[0, 0], [0, 0], [1, 1]])

    with pytest.raises(ValueError, match="target coords contain duplicate"):
        align_by_coords(input_coords, target_coords)


def test_align_by_coords_rejects_incomplete_coverage_missing_in_target() -> None:
    input_coords = torch.tensor([[0, 0], [1, 1], [2, 2]])
    target_coords = torch.tensor([[0, 0], [1, 1]])

    with pytest.raises(ValueError, match="input tile"):
        align_by_coords(input_coords, target_coords)


def test_align_by_coords_rejects_incomplete_coverage_missing_in_input() -> None:
    input_coords = torch.tensor([[0, 0], [1, 1]])
    target_coords = torch.tensor([[0, 0], [1, 1], [2, 2]])

    with pytest.raises(ValueError, match="target tile"):
        align_by_coords(input_coords, target_coords)
