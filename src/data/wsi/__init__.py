"""WSI-level data contracts and loaders."""

from src.data.wsi.bag import WSIBag
from src.data.wsi.batch import PaddedWSIBatch, pad_wsi_bags
from src.data.wsi.collate import collate_padded_wsi_bags, collate_wsi_bags
from src.data.wsi.dataset import (
    FeatureStoreWSIBagDataset,
    InMemoryWSIBagDataset,
    WSIBagDataset,
)
from src.data.wsi.feature_store import InMemoryWSIFeatureStore, WSIFeatureStore
from src.data.wsi.h5_feature_store import H5WSIFeatureStore
from src.data.wsi.trident import (
    TridentSlideRecord,
    load_trident_slide_record,
    read_trident_coords,
    read_trident_features,
)

__all__ = [
    "read_trident_features",
    "read_trident_coords",
    "load_trident_slide_record",
    "TridentSlideRecord",
    "FeatureStoreWSIBagDataset",
    "H5WSIFeatureStore",
    "InMemoryWSIBagDataset",
    "InMemoryWSIFeatureStore",
    "PaddedWSIBatch",
    "WSIBag",
    "WSIBagDataset",
    "WSIFeatureStore",
    "collate_padded_wsi_bags",
    "collate_wsi_bags",
    "pad_wsi_bags",
]
