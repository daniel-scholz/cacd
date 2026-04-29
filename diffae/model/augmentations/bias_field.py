from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class RandBiasFieldCorruptionAugmentation(nn.Module):
    """
    Custom version of BiasField corruption to match synthseg parameters.
    (MONAI's RandBiasField uses a different parameter set).
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        dims: int,
        channels: int = 1,
        bias_field_std=0.5,
        bias_scale=0.025,
    ):
        super().__init__()
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.img_size = img_size
        self.dims = dims
        self.channels = channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        bias_field_transform = self.sample_rand_bias_field(
            device=image.device, dtype=image.dtype
        )
        return bias_field_transform(image)

    def sample_rand_bias_field(self, device: torch.device, dtype: torch.dtype):
        return _RandBiasField(
            bias_field_std=self.bias_field_std,
            channels=self.channels,
            img_size=self.img_size,
            dims=self.dims,
            bias_scale=self.bias_scale,
            device=device,
            dtype=dtype,
        )

    def sample_n_transforms(self, n: int, device: torch.device, dtype: torch.dtype):
        return [
            self.sample_rand_bias_field(device=device, dtype=dtype) for _ in range(n)
        ]


class _RandBiasField(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, ...],
        dims: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        channels: int = 1,
        bias_field_std=0.5,
        bias_scale=0.025,
    ):
        super().__init__()
        self.dims = dims
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        bias_field_small_spatial_shape = tuple(
            [int(np.ceil(s * bias_scale)) for s in img_size]
        )

        small_bias_field_shape = (
            channels,
            *bias_field_small_spatial_shape,
        )

        self.bias_field_small = torch.normal(
            mean=torch.zeros(small_bias_field_shape),
            std=torch.rand(small_bias_field_shape) * bias_field_std,
        ).to(device=device, dtype=dtype)

        interp_mode = "trilinear" if dims == 3 else "bilinear"

        self.interpolate = partial(torch.nn.functional.interpolate, mode=interp_mode)

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        # resize to match image size
        target_spatial_size = image.shape[-self.dims :]

        bias_field = self.interpolate(
            self.bias_field_small.unsqueeze(0),  # add batch dim
            size=target_spatial_size,
        )

        bias_field = torch.exp(bias_field)

        vmin = image.min()
        vmax = image.max()

        # Normalize image to [0, 1]
        image = (image - vmin) / (vmax - vmin)

        image = image * bias_field

        # scale to 0,1

        # Denormalize image
        image = image * (vmax - vmin) + vmin

        # clamp max and min
        image = torch.clamp(image, vmin, vmax)

        return image


if __name__ == "__main__":

    def main():
        import os
        from pathlib import Path

        import nibabel as nib
        import torchvision.utils as vutils

        # Set IXI_DATA_DIR to the directory holding T1 NIfTIs to run this
        # visual-check entry point.
        dataset_dir = Path(os.environ["IXI_DATA_DIR"]) / "T1"
        fn = next(iter(sorted(p.name for p in dataset_dir.iterdir())))
        fp = dataset_dir / fn

        image = torch.from_numpy(nib.load(str(fp)).get_fdata(dtype=np.float32))

        # 0,1 normalize
        image = (image - image.min()) / (image.max() - image.min())
        # scale to -1,1
        image = 2 * image - 1
        spatial_dims = image.shape[-3:]
        print(f"{spatial_dims=}")

        corruptor = RandBiasFieldCorruptionAugmentation(
            img_size=spatial_dims,
            dims=3,
            bias_field_std=0.5,
            bias_scale=0.025,
        )
        image_as_batch = image.unsqueeze(0).unsqueeze(0)
        for i in range(10):
            # Test
            corrupted_image = corruptor(image_as_batch)
            image_mid_slice = corrupted_image[..., image.shape[-1] // 2]
            vutils.save_image(
                image_mid_slice,
                "test_bias_corrupted_image.png",
                value_range=(-1, 1),
                normalize=True,
            )

    main()
