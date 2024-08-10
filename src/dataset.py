# %%

from typing import Literal
from monai.data import PersistentDataset, CacheDataset
from monai.transforms import (
    CenterSpatialCrop,
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Resize,
    ScaleIntensityRange,
)

from pydantic.dataclasses import dataclass
from pathlib import Path

CachingType = Literal["memory", "disk"]


@dataclass
class DatasetConfig:
    path: str
    caching: CachingType = "disk"
    image_size: int = 256  # image height and width
    num_slices: int = 32  # image depth
    win_wid: int = 400  # window width for converting to HO scale
    win_lev: int = 60  # window level for converting to HO scale
    cache_dir = "./data/preprocessed"

    # batch_size = 4
    # lambda_gp = 10 # controls how much of gradient penalty will be added to critic loss
    # learning_rate = 1e-5
    # latent_size = 100


def train_transforms(ds_config: DatasetConfig):
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            CenterSpatialCrop((380, 380, 0)),
            Resize((ds_config.image_size, ds_config.image_size, ds_config.num_slices)),
            ScaleIntensityRange(
                a_min=ds_config.win_lev - (ds_config.win_wid / 2),
                a_max=ds_config.win_lev + (ds_config.win_wid / 2),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureType(),
        ]
    )


# train_transforms = Compose(
#     [
#         LoadImage(image_only=True),
#         EnsureChannelFirst(),
#         CenterSpatialCrop((390, 390, 0)),
#         Resize((cube_len, cube_len, depth)),
#         ScaleIntensity(),
#         AdjustContrast(1.5),
#         RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
#         RandFlip(spatial_axis=0, prob=0.5),
#         RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
#         EnsureType(),
#     ]
# )
# %%
def load_dataset(ds_config: DatasetConfig) -> CacheDataset | PersistentDataset:
    files = list(Path(ds_config.path).glob("*.nii.gz"))
    match ds_config.caching:
        case "disk":
            cache_dir = Path(ds_config.cache_dir) / Path(ds_config.path).name
            cache_dir.mkdir(parents=True, exist_ok=True)
            return PersistentDataset(
                files, train_transforms(ds_config), cache_dir=cache_dir
            )
        case "memory":
            return CacheDataset(files, train_transforms(ds_config))

    raise ValueError(f"Unknown dataset type: {ds_config.caching}")
