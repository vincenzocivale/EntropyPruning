"""WSI preprocessing and attention/embedding analysis utilities."""

from src.data.wsi.attention import (
    EmbeddedAttentionSource,
    FeatureStoreAttentionSource,
    ManifestAttentionSource,
    WSIAttention,
    WSIAttentionSource,
    align_attention_to_bag,
)
from src.data.wsi.attention_file import (
    read_attention_coords,
    read_attention_tensor,
    reduce_attention_tensor,
)
from src.data.wsi.bag import WSIBag
from src.data.wsi.coord_alignment import align_by_coords
from src.data.wsi.feature_store import InMemoryWSIFeatureStore, WSIFeatureStore
from src.data.wsi.generic_features import (
    GenericFeatureSlideRecord,
    load_generic_feature_slide_record,
    read_generic_coords_tensor,
    read_generic_feature_tensor,
)
from src.data.wsi.h5_feature_store import H5WSIFeatureStore
from src.data.wsi.trident import (
    TridentSlideRecord,
    load_trident_slide_record,
    read_trident_coords,
    read_trident_features,
)

__all__ = [
    "EmbeddedAttentionSource",
    "FeatureStoreAttentionSource",
    "GenericFeatureSlideRecord",
    "H5WSIFeatureStore",
    "InMemoryWSIFeatureStore",
    "ManifestAttentionSource",
    "TridentSlideRecord",
    "WSIAttention",
    "WSIAttentionSource",
    "WSIBag",
    "WSIFeatureStore",
    "align_attention_to_bag",
    "align_by_coords",
    "load_generic_feature_slide_record",
    "load_trident_slide_record",
    "read_attention_coords",
    "read_attention_tensor",
    "read_generic_coords_tensor",
    "read_generic_feature_tensor",
    "read_trident_coords",
    "read_trident_features",
    "reduce_attention_tensor",
]
