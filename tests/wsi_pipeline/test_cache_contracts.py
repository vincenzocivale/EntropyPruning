from src.wsi_pipeline.cache_contracts import TileCacheSpec, WSICacheSpec


def test_tile_cache_id_is_stable_and_layer_sensitive() -> None:
    a = TileCacheSpec(tile_encoder="conch", model_revision="r1", early_layer=2)
    b = TileCacheSpec(tile_encoder="conch", model_revision="r1", early_layer=2)
    c = TileCacheSpec(tile_encoder="conch", model_revision="r1", early_layer=3)
    assert a.cache_id == b.cache_id
    assert a.cache_id != c.cache_id


def test_wsi_cache_id_tracks_teacher_pair() -> None:
    titan = WSICacheSpec(tile_encoder="conch", wsi_encoder="titan")
    other = WSICacheSpec(tile_encoder="conch", wsi_encoder="other")
    assert titan.cache_id != other.cache_id
