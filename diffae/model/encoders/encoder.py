from dataclasses import dataclass
from typing import Any, Literal, Sequence, Tuple

import torch
from torch import nn

from diffae.config_base import BaseConfig
from diffae.model.blocks.attention import AttentionBlock
from diffae.model.blocks.resblock import ConstResblockKwargs, ResBlockConfig
from diffae.model.blocks.timestep_blocks import TimestepEmbedSequential
from diffae.model.blocks.updownsample import Downsample
from diffae.model.nn import adaptive_avg_pool_nd, conv_nd, normalization


@dataclass
class EncoderConfig(BaseConfig):
    image_size: int
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int]
    use_attention: bool
    dropout: float = 0
    channel_mult: Sequence[int] = (1, 2, 4, 8)
    use_time_condition: bool = True
    conv_resample: bool = True
    dims: Literal[2, 3] = 2
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = "adaptivenonzero"

    def make_model(self):
        return BeatGANsEncoderModel(self)


class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(self, conf: EncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = torch.float32

        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = (
            conf.image_size
            if isinstance(conf.image_size, int)
            else min(conf.image_size)
        )
        const_resblock_kwargs = ConstResblockKwargs(
            use_checkpoint=conf.use_checkpoint,
            dropout_rate=conf.dropout,
            dims=conf.dims,
            use_zero_module=True,
        )
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers: list[Any] = [
                    ResBlockConfig(
                        in_channels=ch,
                        out_channels=int(mult * conf.model_channels),
                        **const_resblock_kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if conf.use_attention and resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            in_channels=ch,
                            out_channels=out_ch,
                            down=True,
                            **const_resblock_kwargs,
                        ).make_model()
                        if (conf.resblock_updown)
                        else Downsample(
                            ch, conf.conv_resample, dims=conf.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                in_channels=ch,
                out_channels=ch,
                **const_resblock_kwargs,
            ).make_model(),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=conf.use_checkpoint,
                    num_heads=conf.num_heads,
                    num_head_channels=conf.num_head_channels,
                    use_new_attention_order=conf.use_new_attention_order,
                )
                if conf.use_attention
                else nn.Identity()
            ),
            ResBlockConfig(
                in_channels=ch,
                out_channels=ch,
                **const_resblock_kwargs,
            ).make_model(),
        )
        self._feature_size += ch
        if conf.pool == "adaptivenonzero":
            self._init_out_layer(conf, ch)
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")

    def _init_out_layer(self, conf: EncoderConfig, ch: int):
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            adaptive_avg_pool_nd(conf.dims, output_size=1),
            conv_nd(conf.dims, ch, conf.out_channels, 1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor, return_2d_feature=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """

        results = []

        for module in self.input_blocks:
            x = module(x)
            if self.conf.pool.startswith("spatial"):
                results.append(x.mean(dim=(2, 3)))
        x = self.middle_block(x)

        h_2d = x
        # B x 512 x [4x4(x4)]
        x = self.out(x)
        # B x 512

        if return_2d_feature:
            return x, h_2d
        else:
            return x
