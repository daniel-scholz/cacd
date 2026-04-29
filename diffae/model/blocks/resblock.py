from dataclasses import dataclass
from typing import Optional, TypedDict

import torch
from torch import nn

from diffae.config_base import BaseConfig
from diffae.model.blocks.updownsample import Downsample, Upsample
from diffae.model.nn import conv_nd, normalization, torch_checkpoint


class ConstResblockKwargs(TypedDict):
    """
    Contains part of the arguments for the ResBlock constructor.
    These should be consistent across all ResBlocks in a model.
    """

    use_zero_module: bool
    dropout_rate: float
    dims: int
    use_checkpoint: bool


@dataclass(kw_only=True)
class ResBlockConfig(BaseConfig):
    # number of input channels
    in_channels: int

    # dropout rate
    dropout_rate: float
    # number of output channels
    out_channels: int

    # whether to use 3x3 conv for skip path when the channels aren't matched
    use_skip_conv3x3: bool = False

    # dimension of conv
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False

    # suggest: False
    has_lateral: bool = False
    lateral_channels: Optional[int] = None

    # whether to init the convolution with zero weights
    # this is default from BeatGANs and seems to help learning
    use_zero_module: bool = True

    def make_model(self):
        return ResBlock(self)


class ResBlock(nn.Module):

    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf

        #############################
        # IN LAYERS
        #############################

        layers = [
            normalization(conf.in_channels),
            nn.SiLU(),
            conv_nd(conf.dims, conf.in_channels, conf.out_channels, 3, padding=1),
        ]
        self.in_layers = nn.Sequential(*layers)

        self.updown = conf.up or conf.down

        if conf.up:
            self.h_upd = Upsample(conf.in_channels, False, conf.dims)
            self.x_upd = Upsample(conf.in_channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.in_channels, False, conf.dims)
            self.x_upd = Downsample(conf.in_channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # SKIP LAYERS
        #############################
        if conf.out_channels == conf.in_channels:
            # cannot be used with gatedconv, also gatedconv is alsways used as the first block
            self.skip_connection = nn.Identity()
        else:
            if conf.use_skip_conv3x3:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(
                conf.dims,
                conf.in_channels,
                conf.out_channels,
                kernel_size,
                padding=padding,
            )

    def forward(self, x: torch.Tensor, lateral: Optional[torch.Tensor] = None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return torch_checkpoint(self._forward, (x, lateral), self.conf.use_checkpoint)

    def _forward(
        self,
        x: torch.Tensor,
        lateral: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            lateral: required if "has_lateral" and non-gated, with gated,
            it can be supplied optionally
        """
        # pass through main connection (analogy to residual connectino)
        x, h = self.main_connection(x, lateral)

        return self.skip_connection(x) + h

    def main_connection(self, x: torch.Tensor, lateral: Optional[torch.Tensor] = None):

        if self.conf.has_lateral:
            # lateral may be supplied even if it doesn't require
            # the model will take the lateral only if "has_lateral"
            assert lateral is not None
            x = torch.cat([x, lateral], dim=1)

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # return x because potentially with lateral
        return x, h
