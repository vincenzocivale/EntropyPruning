from src.wsi_pipeline.cache_contracts import TileCacheSpec, WSICacheSpec


def test_tile_cache_id_is_stable() -> None:
    a = TileCacheSpec(tile_encoder="conch", model_revision="r1", early_layer=2)
    b = TileCacheSpec(tile_encoder="conch", model_revision="r1", early_layer=2)
    assert a.cache_id == b.cache_id


def test_tile_cache_id_ignores_early_layer() -> None:
    # early_layer no longer determines any stored bytes (early_tokens is not cached,
    # see CACHE_SCHEMA_VERSION v2) -- it is a pure online-recompute training
    # parameter, so it must not force an unrelated, expensive cache rebuild.
    a = TileCacheSpec(tile_encoder="conch", model_revision="r1", early_layer=2)
    c = TileCacheSpec(tile_encoder="conch", model_revision="r1", early_layer=3)
    assert a.cache_id == c.cache_id


def test_tile_cache_id_is_sensitive_to_stored_content_fields() -> None:
    a = TileCacheSpec(tile_encoder="conch", model_revision="r1", dtype="float16")
    b = TileCacheSpec(tile_encoder="conch", model_revision="r1", dtype="float32")
    assert a.cache_id != b.cache_id


def test_wsi_cache_id_tracks_teacher_pair() -> None:
    titan = WSICacheSpec(tile_encoder="conch", wsi_encoder="titan")
    other = WSICacheSpec(tile_encoder="conch", wsi_encoder="other")
    assert titan.cache_id != other.cache_id
