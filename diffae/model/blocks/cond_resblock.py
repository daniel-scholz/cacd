from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from diffae.model.blocks.resblock import ConstResblockKwargs, ResBlock, ResBlockConfig
from diffae.model.nn import conv_nd, linear, normalization, torch_checkpoint, zero_module


@dataclass(kw_only=True)
class TimeStyleCondResBlockConfig(ResBlockConfig):
    # time embedding channels
    t_emb_channels: int

    # number of encoders' output channels
    cond_channels: int

    def make_model(self):
        return TimeStyleCondResBlock(self)


class TimeStyleCondResBlock(ResBlock):
    """
    A residual block that can optionally change the number of channels.
    is used for denoising defusion

    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """

    def __init__(self, conf: TimeStyleCondResBlockConfig):
        super().__init__(conf=conf)
        # condition layers for the out_layers
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(conf.t_emb_channels, 2 * conf.out_channels),
        )

        self.cond_emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(conf.cond_channels, conf.out_channels),
        )

        conv = conv_nd(conf.dims, conf.out_channels, conf.out_channels, 3, padding=1)
        if conf.use_zero_module:
            # zere out the weights
            # it seems to help training
            conv = zero_module(conv)

        # construct the layers
        # - norm
        # - (modulation)
        # - act
        # - dropout
        # - conv
        layers = []
        layers += [
            normalization(conf.out_channels),
            nn.SiLU(),
            nn.Dropout(p=conf.dropout_rate),
            conv,
        ]
        self.out_layers = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
        lateral: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch_checkpoint(self._forward, (x, t_emb, cond, lateral), self.conf.use_checkpoint)

    def _forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
        lateral: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, h = self.main_connection(x, t_emb, cond, lateral)
        return self.skip_connection(x) + h

    def main_connection(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
        lateral: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # call normal resblock main connection (without the skip connection)
        x, h = super().main_connection(x, lateral)

        t_emb_out = self.t_emb_layers(t_emb)

        cond_emb_out = self.cond_emb_layers(cond)

        # this is the new refactored code
        h = self.apply_conditions(
            h=h,
            t_emb=t_emb_out,
            cond_emb=cond_emb_out,
            layers=self.out_layers,
            scale_bias=1.0,
            in_channels=self.conf.out_channels,
            up_down_layer=None,
        )
        # skip connection is applied in _forward

        return x, h

    def apply_conditions(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor,
        cond_emb: torch.Tensor,
        layers: nn.Sequential,
        scale_bias: float,
        in_channels: int,
        up_down_layer: Optional[nn.Module] = None,
    ):
        """
        apply conditions on the feature maps

        Args:
            emb: time conditional (ready to scale + shift)
            cond: encoder's conditional (read to scale + shift)
        """

        # expand the scale and shift to match the feature map shape
        t_emb = t_emb[..., *(None,) * (h.dim() - 2)]
        cond_emb = cond_emb[..., *(None,) * (h.dim() - 2)]

        # time first
        scale_shifts = [t_emb, cond_emb]
        # support scale, shift or shift only
        for i, each in enumerate(scale_shifts):
            if each.shape[1] == in_channels * 2:
                a, b = torch.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
            scale_shifts[i] = (a, b)

        # condition scale bias could be a list
        biases = [scale_bias] * len(scale_shifts)

        # default, the scale & shift are applied after the group norm but BEFORE SiLU
        pre_layers, post_layers = layers[0], layers[1:]

        # spilt the post layer to be able to scale up or down before conv
        # post layers will contain only the conv
        mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

        h = pre_layers(h)
        # scale and shift for each condition
        for i, (scale, shift) in enumerate(scale_shifts):
            # if scale is None, it indicates that the condition is not provided
            if scale is not None:
                h = h * (biases[i] + scale)
                if shift is not None:
                    h = h + shift
        h = mid_layers(h)

        # upscale or downscale if any just before the last conv
        if up_down_layer is not None:
            h = up_down_layer(h)
        h = post_layers(h)
        return h


class ConstCondResblockKwargs(ConstResblockKwargs):
    cond_channels: int
    t_emb_channels: int
