from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd, RandSpatialCropd


def RandForegrundCropd(
    keys: list[str], crop_size: int, allow_smaller: bool = True, k_divisible: int = 8
):
    source_key = "img"
    assert source_key in keys, f"source_key {source_key} not in keys {keys}"

    return Compose(
        [
            CropForegroundd(
                keys=keys,
                source_key=source_key,
                margin=0,
                k_divisible=k_divisible,
                allow_smaller=allow_smaller,
                start_coord_key=None,
                end_coord_key=None,
                # combine with subsequent crop
                # lazy=True,
            ),
            RandSpatialCropd(
                keys=keys,
                roi_size=[
                    crop_size,
                ]
                * 3,
                random_size=False,
                random_center=True,
                # combine with prev crop
                # lazy=True,
            ),
        ]
    )
