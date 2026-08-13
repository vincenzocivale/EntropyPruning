import csv
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src.data.wsi.manifest import SlideRecord, write_manifest
from src.models.online_tile_eaf import OnlineAttentionTeacher
from src.wsi_pipeline.cache_contracts import TileCacheSpec
from src.wsi_pipeline.cache_index import build_tile_cache_index, read_tile_cache_index
from src.wsi_pipeline.cache_io import TileCacheWriter, validate_cache
from src.wsi_pipeline.patch_dataset import OpenSlideCoordinateDataset, SlideSequentialBatchSampler
from src.wsi_pipeline.tile_cache_pipeline import (
    TileCacheItem,
    TileCacheRunConfig,
    autotune_loader,
    cache_many_slides,
)


def _write_coords(path: Path, coords: np.ndarray) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=coords)


def test_preallocated_writer_requires_exact_count(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    spec = TileCacheSpec(tile_encoder="test")
    with pytest.raises(RuntimeError, match="wrote 1, expected 2"):
        with TileCacheWriter(
            path, spec, slide_id="s1", case_id="c1", expected_n=2
        ) as writer:
            writer.append(
                coords=np.asarray([[0, 0]]),
                final_attention=np.ones((1, 4), dtype=np.float16) / 4,
                tile_embeddings=np.ones((1, 3), dtype=np.float16),
            )
    assert not path.exists()


def test_build_tile_cache_index_validates_coords(tmp_path: Path) -> None:
    coords = np.asarray([[0, 0], [8, 16]], dtype=np.int32)
    coords_path = tmp_path / "coords.h5"
    raw_path = tmp_path / "slide.svs"
    raw_path.write_bytes(b"placeholder")
    _write_coords(coords_path, coords)
    slides_path = write_manifest(
        tmp_path / "slides.csv",
        [
            SlideRecord(
                slide_id="s1", case_id="c1", source="histai",
                raw_path=str(raw_path), coords_path=str(coords_path), split="train",
            )
        ],
    )
    cache_root = tmp_path / "caches"
    cache_root.mkdir()
    spec = TileCacheSpec(tile_encoder="test")
    with TileCacheWriter(
        cache_root / "s1.h5", spec, slide_id="s1", case_id="c1", expected_n=2
    ) as writer:
        writer.append(
            coords=coords,
            final_attention=np.ones((2, 4), dtype=np.float16) / 4,
            tile_embeddings=np.ones((2, 3), dtype=np.float16),
        )

    output = tmp_path / "index.csv"
    rows = build_tile_cache_index(
        slides_path, [cache_root], output, expected_cache_id=spec.cache_id
    )
    assert len(rows) == 1
    assert read_tile_cache_index(output)["s1"] == cache_root / "s1.h5"
    assert validate_cache(cache_root / "s1.h5")["n_tiles"] == 2


def test_read_tile_cache_index_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("slide_id", "cache_path", "n_tiles", "cache_id")
        )
        writer.writeheader()
        writer.writerows(
            [
                {"slide_id": "s1", "cache_path": "a.h5", "n_tiles": "1", "cache_id": "x"},
                {"slide_id": "s1", "cache_path": "b.h5", "n_tiles": "1", "cache_id": "x"},
            ]
        )
    with pytest.raises(ValueError, match="duplicate"):
        read_tile_cache_index(path)


def test_online_teacher_extract_early_skips_later_blocks() -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value + 1

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block(), Block()])

        def forward_features(self, value):
            for block in self.blocks:
                value = block(value)
            return value

    class Adapter:
        num_prefix_tokens = 1
        n_blocks = 3

        def __init__(self, model):
            self.model = model

        def get_blocks(self):
            return self.model.blocks

    backbone = Backbone()
    teacher = OnlineAttentionTeacher(backbone, Adapter(backbone), 1, 2)
    source = teacher.extract_early(torch.zeros(2, 5, 3))
    assert source.shape == (2, 4, 3)
    assert [block.calls for block in backbone.blocks] == [1, 1, 0]


def test_target_attention_matches_full_matrix_reference() -> None:
    """CLS-row optimization must preserve today's full-[B,H,N,N]-matrix result."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    class FakeAttention(torch.nn.Module):
        def __init__(self, dim: int = 8, num_heads: int = 2) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = torch.nn.Linear(dim, dim * 3, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

    class Adapter:
        num_prefix_tokens = 1
        n_blocks = 2

    teacher = OnlineAttentionTeacher(
        torch.nn.Module(), Adapter(), source_layer=0, target_layer=1
    )
    module = FakeAttention()
    x = torch.randn(3, 6, 8)

    # Reference: the full-[B,H,N,N]-matrix computation this optimizes away,
    # inlined so the test keeps proving equivalence even after the fix lands.
    batch, tokens, channels = x.shape
    qkv = module.qkv(x).reshape(
        batch, tokens, 3, module.num_heads, module.head_dim
    ).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)
    full = (q @ k.transpose(-2, -1) * module.scale).float().softmax(dim=-1)
    reference = full[:, :, 0, Adapter.num_prefix_tokens :].mean(dim=1)
    reference = reference / reference.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    result = teacher._target_attention(module, x)
    torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)


def test_slide_sequential_batch_sampler_never_crosses_boundaries() -> None:
    batches = list(SlideSequentialBatchSampler((3, 5), batch_size=2))
    assert batches == [[0, 1], [2], [3, 4], [5, 6], [7]]


def test_openslide_cache_is_installed_lazily(tmp_path: Path, monkeypatch) -> None:
    installed = []

    class FakeCache:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity

    class FakeSlide:
        def __init__(self, _path: str) -> None:
            pass

        def set_cache(self, cache) -> None:
            installed.append(cache.capacity)

    monkeypatch.setitem(
        sys.modules, "openslide",
        SimpleNamespace(OpenSlide=FakeSlide, OpenSlideCache=FakeCache),
    )
    coords_path = tmp_path / "coords.h5"
    _write_coords(coords_path, np.asarray([[0, 0]], dtype=np.int32))
    dataset = OpenSlideCoordinateDataset(
        tmp_path / "slide.tiff", coords_path, lambda image: image,
        output_size=512, openslide_cache_bytes=256 * 2**20,
    )

    dataset._get_slide()

    assert installed == [256 * 2**20]


def test_autotune_loader_threads_openslide_cache_bytes(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image

    installed = []

    class FakeCache:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity

    class FakeSlide:
        def __init__(self, _path: str) -> None:
            pass

        def read_region(self, location, _level, size):
            return Image.new("RGBA", size, (0, 0, 0, 255))

        def set_cache(self, cache) -> None:
            installed.append(cache.capacity)

        def close(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules, "openslide",
        SimpleNamespace(OpenSlide=FakeSlide, OpenSlideCache=FakeCache),
    )

    raw_path = tmp_path / "slide.svs"
    raw_path.touch()
    coords_path = tmp_path / "coords.h5"
    _write_coords(coords_path, np.asarray([[0, 0], [4, 4]], dtype=np.int32))
    item = TileCacheItem(slide_id="s0", case_id="c0", wsi_path=raw_path, coords_path=coords_path)

    class Adapter:
        input_size = 4

        @staticmethod
        def transform(image):
            array = np.asarray(image, dtype=np.float32).copy()
            return torch.from_numpy(array).permute(2, 0, 1) / 255

        def extract_final(self, images):
            batch = int(images.shape[0])
            return SimpleNamespace(
                final_attention=torch.full((batch, 4), 0.25, dtype=torch.float16),
                tile_embeddings=torch.ones((batch, 3), dtype=torch.float16),
            )

    # Regression guard for a NameError: autotune_loader used to reference an
    # undefined `config.openslide_cache_bytes` instead of taking the value as
    # a parameter -- this call would raise before this fix.
    autotune_loader(
        item, Adapter(), device=torch.device("cpu"),
        batch_size_candidates=(1,), worker_candidates=(0,),
        prefetch_candidates=(2,), benchmark_batches=1,
        openslide_cache_bytes=123 * 2**20,
    )

    assert installed == [123 * 2**20]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_multi_slide_cache_pipeline_reuses_one_loader(
    tmp_path: Path, monkeypatch, num_workers: int
) -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image

    class FakeSlide:
        def __init__(self, _path: str) -> None:
            pass

        def read_region(self, location, _level, size):
            value = int(location[0]) % 255
            return Image.new("RGBA", size, (value, value, value, 255))

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace(OpenSlide=FakeSlide))

    items = []
    for slide_index, count in enumerate((3, 5)):
        raw_path = tmp_path / f"slide_{slide_index}.svs"
        raw_path.touch()
        coords_path = tmp_path / f"coords_{slide_index}.h5"
        coords = np.arange(count * 2, dtype=np.int32).reshape(count, 2)
        _write_coords(coords_path, coords)
        items.append(
            TileCacheItem(
                slide_id=f"s{slide_index}", case_id=f"c{slide_index}",
                wsi_path=raw_path, coords_path=coords_path,
            )
        )

    class Adapter:
        name = "test"
        revision = "test"
        input_size = 4

        def __init__(self) -> None:
            self.batch_sizes = []

        @staticmethod
        def transform(image):
            array = np.asarray(image, dtype=np.float32).copy()
            return torch.from_numpy(array).permute(2, 0, 1) / 255

        def extract_final(self, images):
            batch = int(images.shape[0])
            self.batch_sizes.append(batch)
            return SimpleNamespace(
                final_attention=torch.full((batch, 4), 0.25, dtype=torch.float16),
                tile_embeddings=torch.ones((batch, 3), dtype=torch.float16),
            )

    adapter = Adapter()
    spec = TileCacheSpec(tile_encoder="test", model_revision="test")
    rows = cache_many_slides(
        items,
        adapter=adapter,
        spec=spec,
        config=TileCacheRunConfig(
            output_dir=tmp_path / "cache",
            batch_size=2,
            num_workers=num_workers,
            device="cpu",
            compression=None,
            slide_loader_chunk_size=2,
            persistent_workers=True,
        ),
    )
    assert [row["status"] for row in rows] == ["built", "built"]
    assert adapter.batch_sizes == [2, 1, 2, 2, 1]
    assert validate_cache(tmp_path / "cache" / "s0.h5")["n_tiles"] == 3
    assert validate_cache(tmp_path / "cache" / "s1.h5")["n_tiles"] == 5


def test_shared_loader_failure_falls_back_per_slide(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image

    class FakeSlide:
        def __init__(self, _path: str) -> None:
            pass

        def read_region(self, _location, _level, size):
            return Image.new("RGBA", size, (10, 20, 30, 255))

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace(OpenSlide=FakeSlide))
    items = []
    for index in range(2):
        raw_path = tmp_path / f"slide_{index}.svs"
        raw_path.touch()
        coords_path = tmp_path / f"coords_{index}.h5"
        _write_coords(coords_path, np.asarray([[0, 0], [1, 1]], dtype=np.int32))
        items.append(TileCacheItem(f"s{index}", f"c{index}", raw_path, coords_path))

    class FailOnceAdapter:
        name = "test"
        revision = "test"
        input_size = 4

        def __init__(self) -> None:
            self.calls = 0

        @staticmethod
        def transform(image):
            return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)

        def extract_final(self, images):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic shared-loader failure")
            batch = len(images)
            return SimpleNamespace(
                final_attention=torch.full((batch, 4), 0.25, dtype=torch.float16),
                tile_embeddings=torch.ones((batch, 3), dtype=torch.float16),
            )

    rows = cache_many_slides(
        items,
        adapter=FailOnceAdapter(),
        spec=TileCacheSpec(tile_encoder="test", model_revision="test"),
        config=TileCacheRunConfig(
            output_dir=tmp_path / "cache", batch_size=2, num_workers=0,
            device="cpu", compression=None, slide_loader_chunk_size=2,
        ),
    )
    assert [row["status"] for row in rows] == ["built", "built"]
    assert validate_cache(tmp_path / "cache" / "s0.h5")["n_tiles"] == 2
    assert validate_cache(tmp_path / "cache" / "s1.h5")["n_tiles"] == 2
