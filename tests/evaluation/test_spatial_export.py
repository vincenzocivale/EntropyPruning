import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")


def test_tile_export_preserves_ids_and_full_student_distinction(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from src.evaluation.spatial.extract import export_tile_pair, _save_full

    class Student(torch.nn.Module):
        def full_teacher_embedding(self, x):
            return x.mean(dim=(2, 3))

        def forward(self, x):
            return self.full_teacher_embedding(x) + 1

    files = []
    for i in range(3):
        path = tmp_path / f"patch{i}.png"
        Image.new("RGB", (4, 4), (i, i, i)).save(path)
        files.append(str(path))
    transform = lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    full, eaf, timing = export_tile_pair(Student(), transform, pd.DataFrame({"patch_path": files}),
                                          device=torch.device("cpu"), batch_size=2)
    np.testing.assert_allclose(eaf - full, 1)
    np.testing.assert_allclose(full[:, 0], [0, 1, 2])
    assert timing["full_seconds"] >= 0
    ids = np.array(["a", "b", "c"])
    _save_full(tmp_path, ids, full, "spot_id")
    _save_full(tmp_path, ids[::-1], full[::-1], "spot_id")
    with pytest.raises(ValueError, match="differs"):
        _save_full(tmp_path, ids, eaf, "spot_id")


def preprocess_features(features, coords, size):
    """Tiny x-major TITAN preprocessing stand-in, deliberately reorders tiles."""
    grid = torch.div(coords - coords.min(dim=0).values, size, rounding_mode="floor")
    h, w = (grid.max(dim=0).values + 1).tolist()
    image = features.new_zeros((1, features.shape[1], h, w))
    positions = coords.new_zeros((1, 2, h, w))
    mask = torch.zeros((1, h, w), dtype=torch.bool)
    for i, (x, y) in enumerate(grid.tolist()):
        image[0, :, x, y] = features[i]
        positions[0, :, x, y] = coords[i]
        mask[0, x, y] = True
    return image, positions, mask


def test_titan_selection_maps_grid_tokens_back_to_original_input():
    pytest.importorskip("peft")
    from src.models.wsi.pruned_titan import PrunedLoRATitanEncoder

    class Block(torch.nn.Module):
        def forward(self, x, attn_mask, bg_mask=None):
            return x

    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modules_list = torch.nn.ModuleList([Block(), Block()])

        def forward(self, x, mask, bg_mask=None):
            for block in self.modules_list:
                x = block(x, mask, bg_mask)
            return x

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = Blocks()

        def forward(self, features, coords, size):
            image, _, mask = preprocess_features(features, coords, size)
            tokens = image[0].permute(1, 2, 0)[mask[0]].unsqueeze(0)
            tokens = torch.cat([torch.zeros_like(tokens[:, :1]), tokens], dim=1)
            return self.blocks(tokens, None, mask).sum(dim=1)

    class Titan(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_encoder = Vision()

        def encode_slide_from_patch_features(self, features, coords, size):
            return self.vision_encoder(features, coords, size)

    class Forecaster(torch.nn.Module):
        def forward(self, x):
            return x[:, :, 0]

    # Bypass LoRA initialization: test the actual inference/selection code with a
    # deterministic toy backbone; no pretrained weights or network needed.
    student = PrunedLoRATitanEncoder.__new__(PrunedLoRATitanEncoder)
    torch.nn.Module.__init__(student)
    student.titan = torch.nn.Module()
    student.titan.model = Titan()
    student.forecaster = Forecaster()
    student.prune_layer, student.keep_ratio = 0, 0.34
    student.patch_size_level0, student.num_prefix_tokens = 10, 1
    student.eval()
    features = torch.tensor([[9., 1], [2, 1], [5, 1]])
    coords = torch.tensor([[20, 0], [0, 0], [10, 0]])
    ordinary = student(features, coords)
    embedding, indices = student.encode_with_selection(features, coords)
    torch.testing.assert_close(embedding, ordinary)
    assert indices.tolist() == [0]  # last grid token is first input row
    student.keep_ratio = 1
    _, indices = student.encode_with_selection(features, coords)
    assert set(indices.tolist()) == {0, 1, 2}
    with pytest.raises(ValueError, match="same TITAN grid cell"):
        student.encode_with_selection(features, torch.tensor([[0, 0], [1, 0], [10, 0]]))
    with pytest.raises(ValueError, match="nonzero"):
        student.encode_with_selection(features * 0, coords)
