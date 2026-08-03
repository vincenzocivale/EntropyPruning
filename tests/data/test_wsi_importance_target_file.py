import numpy as np
import pytest
import torch

from src.data.wsi.importance_target_file import (
    read_wsi_importance_coords_tensor,
    read_wsi_importance_target_tensor,
)


def test_read_wsi_importance_target_tensor_from_h5_default_key(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "target.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("attention", data=np.array([0.1, 0.5, 0.3, 0.1], dtype=np.float32))

    tensor = read_wsi_importance_target_tensor(path)

    assert tensor.shape == (4,)
    assert torch.is_floating_point(tensor)


def test_read_wsi_importance_target_tensor_from_h5_custom_key(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "target.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("my_custom_score", data=np.array([0.2, 0.8], dtype=np.float32))

    tensor = read_wsi_importance_target_tensor(path, key="my_custom_score")

    assert tensor.shape == (2,)


def test_read_wsi_importance_target_tensor_from_npy(tmp_path) -> None:
    path = tmp_path / "target.npy"
    np.save(path, np.array([0.3, 0.3, 0.4], dtype=np.float32))

    tensor = read_wsi_importance_target_tensor(path)

    assert torch.allclose(tensor, torch.tensor([0.3, 0.3, 0.4]))


def test_read_wsi_importance_target_tensor_from_npz_with_key(tmp_path) -> None:
    path = tmp_path / "target.npz"
    np.savez(path, importance=np.array([0.5, 0.5]), other=np.array([[1, 2]]))

    tensor = read_wsi_importance_target_tensor(path)

    assert tensor.shape == (2,)


def test_read_wsi_importance_target_tensor_from_pt(tmp_path) -> None:
    path = tmp_path / "target.pt"
    torch.save({"tile_importance": torch.tensor([0.9, 0.1])}, path)

    tensor = read_wsi_importance_target_tensor(path)

    assert torch.allclose(tensor, torch.tensor([0.9, 0.1]))


def test_read_wsi_importance_target_tensor_rejects_negative_values(tmp_path) -> None:
    path = tmp_path / "target.npy"
    np.save(path, np.array([0.5, -0.1], dtype=np.float32))

    with pytest.raises(ValueError, match="non-negative"):
        read_wsi_importance_target_tensor(path)


def test_read_wsi_importance_target_tensor_rejects_nan(tmp_path) -> None:
    path = tmp_path / "target.npy"
    np.save(path, np.array([0.5, np.nan], dtype=np.float32))

    with pytest.raises(ValueError, match="NaN or Inf"):
        read_wsi_importance_target_tensor(path)


def test_read_wsi_importance_target_tensor_rejects_unknown_key(tmp_path) -> None:
    path = tmp_path / "target.npz"
    np.savez(path, importance=np.array([0.5, 0.5]))

    with pytest.raises(KeyError):
        read_wsi_importance_target_tensor(path, key="does_not_exist")


def test_read_wsi_importance_coords_tensor_from_npy(tmp_path) -> None:
    path = tmp_path / "coords.npy"
    np.save(path, np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64))

    tensor = read_wsi_importance_coords_tensor(path)

    assert tensor.shape == (3, 2)
    assert tensor.dtype == torch.long


def test_read_wsi_importance_coords_tensor_rejects_bad_width(tmp_path) -> None:
    path = tmp_path / "coords.npy"
    np.save(path, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64))

    with pytest.raises(ValueError, match=r"\[n_tiles, 2\] or \[n_tiles, 4\]"):
        read_wsi_importance_coords_tensor(path)


def test_read_wsi_importance_target_tensor_rejects_unsupported_suffix(tmp_path) -> None:
    path = tmp_path / "target.txt"
    path.write_text("not a supported format")

    with pytest.raises(ValueError, match="unsupported file suffix"):
        read_wsi_importance_target_tensor(path)
