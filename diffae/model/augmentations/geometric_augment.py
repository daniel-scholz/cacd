from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from monai.transforms.compose import Compose
from monai.transforms.spatial.array import (
    Rand2DElastic,
    Rand3DElastic,
    RandAffine,
    RandFlip,
)
from monai.transforms.spatial.dictionary import (
    Rand2DElasticd,
    Rand3DElasticd,
    RandAffined,
    RandFlipd,
)


class GeometricAugment(nn.Module):
    def __init__(
        self,
        image_size: float | tuple[float, float],
        spatial_dims: Literal[2, 3],
        keys: Optional[list[str]] = None,
        dict_mode: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert keys is not None and dict_mode, "keys must be provided in dict_mode"
        self.keys = keys
        self.dict_mode = dict_mode
        self.spatial_dims = spatial_dims
        if dict_mode:
            self.affine = Compose(
                [
                    RandFlipd(keys, prob=0.5),
                    RandAffined(
                        keys,
                        prob=0.5,
                        scale_range=0.2,
                        translate_range=0.2,
                        shear_range=0.2,
                        rotate_range=0.2,
                    ),
                    # RandRotate90d(keys, prob=0.5),
                ]
            )
        else:
            self.affine = Compose(
                [
                    RandFlip(prob=0.5),
                    RandAffine(
                        prob=0.5,
                        scale_range=0.2,
                        translate_range=0.2,
                        shear_range=0.2,
                        rotate_range=0.2,
                    ),
                ]
            )

        self._setup_elastic(image_size, spatial_dims)

        self.augment = Compose([self.affine, self.elastic])

    def _setup_elastic(self, image_size, spatial_dims):
        elastic_spacing = self._calc_spacing_from_image_size(
            image_size, n_control_points=4
        )
        magnitude_range = (0, 1)

        match spatial_dims:
            case 2:
                if self.dict_mode:
                    self.elastic = Rand2DElasticd(
                        keys=self.keys,
                        spacing=elastic_spacing,  # type:ignore
                        magnitude_range=magnitude_range,
                        prob=0.5,
                    )
                else:
                    self.elastic = Rand2DElastic(
                        spacing=elastic_spacing,  # type:ignore
                        magnitude_range=magnitude_range,
                        prob=0.5,
                    )
            case 3:
                init_fn = Rand3DElasticd if self.dict_mode else Rand3DElastic
                params = {
                    "sigma_range": (1, 8),
                    "spacing": elastic_spacing,  # type:ignore
                    "magnitude_range": magnitude_range,
                    "prob": 0.5,
                }
                if self.dict_mode:
                    self.elastic = init_fn(keys=self.keys, **params)  # type: ignore
                else:
                    self.elastic = init_fn(**params)

            case _:
                raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")

    def forward(self, img: torch.Tensor):
        return self.augment(img)

    def _calc_spacing_from_image_size(
        self, image_size, n_control_points=10
    ) -> tuple[float, float] | float | tuple[float, float, float]:
        image_size_np = np.array(image_size)
        spacing = tuple(np.round(image_size_np / n_control_points).tolist())
        if len(spacing) == 1:
            spacing = spacing[0]
        return spacing
