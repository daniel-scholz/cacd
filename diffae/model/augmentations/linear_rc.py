from typing import Literal

import torch.nn as nn

from diffae.model.augmentations.gin import GlassOfGIN

type ConvNd = nn.Conv2d | nn.Conv3d


class LinearRandConv(GlassOfGIN):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_hidden_chans,
        spatial_dims,
        n_layers,
        normalization: Literal["fro", "minmax"] = "minmax",
        alpha_range: tuple[float, float] = (0, 1),
        **conv_kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            n_hidden_chans,
            spatial_dims,
            n_layers,
            normalization="minmax",
            alpha_range=alpha_range,
            **conv_kwargs,
        )

    def init_rand_convs(self, rand_conv: ConvNd, rotationally_symmetric: bool):
        nn.init.kaiming_normal_(rand_conv.weight)

    def non_linearity(self):
        return nn.Identity()
