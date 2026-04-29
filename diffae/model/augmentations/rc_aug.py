from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms.spatial.array import Resize

from diffae.model.augmentations.gin import GlassOfGIN
from diffae.model.augmentations.linear_rc import LinearRandConv

torch.manual_seed(0)


class RandConvAugmentation(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        rc_type: Literal["linear", "gin"],
        rotationally_symmetric: bool = False,
        n_layers: int = 4,
        n_hidden_chans: int = 2,
        alpha_range: tuple[float, float] = (0.0, 1.0),
        normalization: Literal["fro", "minmax"] = "minmax",
        do_updownsampling: bool = True,
        resize_mode: str = "bilinear",
        target_size: int = 2048,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_dims = spatial_dims
        self.n_layers = n_layers
        self.n_hidden_chans = n_hidden_chans
        self.alpha_range = alpha_range
        self.normalization = normalization
        self.rc_type = rc_type
        self.rotationally_symmetric = rotationally_symmetric
        self.rng_map = {}

        self._init_updownsampling(do_updownsampling, resize_mode, target_size)

    def _init_updownsampling(
        self, do_updownsampling: bool, resize_mode: str, target_size: int
    ):
        if do_updownsampling:
            self.upsampling = Resize(
                mode=resize_mode,
                spatial_size=target_size,
                size_mode="longest",
                anti_aliasing=True,
                align_corners=resize_mode != "nearest",
            )
            self.downsampling = partial(
                F.interpolate,
                mode=resize_mode,
                align_corners=resize_mode != "nearest",
            )
        else:

            def _identity_with_args(x: Any, *args, **kwargs) -> Any:
                return x

            self.upsampling = _identity_with_args
            self.downsampling = _identity_with_args

    def forward(self, img: torch.Tensor):
        """Implementation of global intensity non-linear (GIN) augmentation."""
        in_channels = img.size(1)
        out_channels = in_channels

        # sample newly in each forward pass
        gin_transform = self._init_transform(in_channels, out_channels)
        gin_img = gin_transform(img)

        return gin_img

    def get_rng(self, device: torch.device) -> torch.Generator:
        if device not in self.rng_map:
            rng = torch.Generator(device=device).manual_seed(device.index or 0)
            self.rng_map[device] = rng

        return self.rng_map[device]

    def sample_n_transforms(
        self, in_channels: int, out_channels: int, n_transforms: int, **conv_kwargs
    ) -> list[GlassOfGIN]:
        return [
            self._init_transform(in_channels, out_channels, **conv_kwargs)
            for _ in range(n_transforms)
        ]

    def _init_transform(
        self, in_channels: int, out_channels, **conv_kwargs
    ) -> GlassOfGIN | LinearRandConv:
        match self.rc_type:
            case "linear":
                return LinearRandConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_hidden_chans=self.n_hidden_chans,
                    spatial_dims=self.spatial_dims,
                    n_layers=self.n_layers,
                    alpha_range=self.alpha_range,
                    **conv_kwargs,
                )
            case "gin":
                return GlassOfGIN(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_hidden_chans=self.n_hidden_chans,
                    spatial_dims=self.spatial_dims,
                    n_layers=self.n_layers,
                    rotationally_symmetric=self.rotationally_symmetric,
                    normalization=self.normalization,
                    alpha_range=self.alpha_range,
                    upsampling_fn=self.upsampling,
                    downsampling_fn=self.downsampling,
                    rng=self.get_rng(conv_kwargs.get("device", torch.device("cpu"))),
                    **conv_kwargs,
                )
            case _:
                raise ValueError(f"Invalid rc_type: {self.rc_type}")
