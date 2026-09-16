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
from src.data.wsi.feature_store import (
    FeatureStoreWSIBagDataset,
    InMemoryWSIFeatureStore,
    WSIFeatureStore,
)
from src.data.wsi.generic_features import (
    GenericFeatureSlideRecord,
    load_generic_feature_slide_record,
    read_generic_coords_tensor,
    read_generic_feature_tensor,
)
from src.data.wsi.h5_feature_store import H5WSIFeatureStore
from src.data.wsi.numpy_feature_store import NumpyWSIFeatureStore
from pathlib import Path
from src.data.wsi.feature_store import WSIFeatureStore


def open_feature_store(path: str | Path, *, read_only: bool = False) -> WSIFeatureStore:
    """Open a NumPy feature store or an explicitly requested legacy HDF5 store."""
    path = Path(path)
    if path.suffix == ".npyd":
        return NumpyWSIFeatureStore(path, read_only=read_only)
    return H5WSIFeatureStore(path, read_only=read_only)
from src.data.wsi.trident import (
    TridentSlideRecord,
    load_trident_slide_record,
    read_trident_coords,
    read_trident_features,
)

__all__ = [
    "EmbeddedAttentionSource",
    "FeatureStoreAttentionSource",
    "FeatureStoreWSIBagDataset",
    "GenericFeatureSlideRecord",
    "H5WSIFeatureStore",
    "NumpyWSIFeatureStore",
    "open_feature_store",
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
