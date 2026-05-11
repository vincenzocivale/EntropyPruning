from .h5_dataset import H5ForecastDataset
from .wsi_tile_dataset import WSITileDataset

try:
    from .thunder_loaders import build_thunder_loaders
except ModuleNotFoundError:
    # `thunder` is only required by legacy H5-based scripts.
    build_thunder_loaders = None
