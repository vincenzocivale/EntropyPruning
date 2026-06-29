"""WSI-level data utilities."""

from src.data.wsi.bag import WSIBag
from src.data.wsi.batch import PaddedWSIBatch, pad_wsi_bags
from src.data.wsi.collate import collate_padded_wsi_bags, collate_wsi_bags
from src.data.wsi.dataset import (
    FeatureStoreWSIBagDataset,
    InMemoryWSIBagDataset,
    WSIBagDataset,
)
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
    "FeatureStoreWSIBagDataset",
    "GenericFeatureSlideRecord",
    "H5WSIFeatureStore",
    "InMemoryWSIBagDataset",
    "InMemoryWSIFeatureStore",
    "PaddedWSIBatch",
    "TridentSlideRecord",
    "WSIBag",
    "WSIBagDataset",
    "WSIFeatureStore",
    "collate_padded_wsi_bags",
    "collate_wsi_bags",
    "load_generic_feature_slide_record",
    "load_trident_slide_record",
    "pad_wsi_bags",
    "read_generic_coords_tensor",
    "read_generic_feature_tensor",
    "read_trident_coords",
    "read_trident_features",
]
