from typing import TYPE_CHECKING, Callable

import torch
from torch import nn

from diffae.model.nn import adaptive_avg_pool_nd, conv_nd, normalization

if TYPE_CHECKING:
    from diffae.model.encoders.encoder_id_preserve import EncoderIDConfig


class SemLayer(nn.Module):
    """A class only generating the semantic feature vector."""

    def __init__(self, z_dim: int, **layer_kwargs):
        super().__init__()
        self._init_layers(z_dim, **layer_kwargs)

    def _init_layers(self, z_dim: int, **common_kwargs):
        self.layer: Callable[[torch.Tensor], torch.Tensor] = conv_nd(
            out_channels=z_dim, **common_kwargs
        )

    def forward(self, x) -> torch.Tensor:
        return self.layer(x).flatten(start_dim=1)


class IDLayer(SemLayer):
    """A class only generating the identity feature vector."""


def NonLinearHead(out_dims: int, dims: int, ch: int):
    return nn.Sequential(
        normalization(ch),
        nn.SiLU(),
        # input is of size 4x4x4
        conv_nd(
            in_channels=ch,
            out_channels=out_dims,
            kernel_size=3,
            padding=1,
            dims=dims,
        ),
        normalization(out_dims),
        nn.SiLU(),
        adaptive_avg_pool_nd(dims, output_size=1),
        conv_nd(
            in_channels=out_dims,
            out_channels=out_dims,
            kernel_size=1,
            dims=dims,
        ),
    )


class SemIDLayer(nn.Module):
    def __init__(self, z_sem_dim: int, z_id_dim: int, **layer_kwargs):
        super().__init__()
        self._init_layers(z_sem_dim, z_id_dim, **layer_kwargs)

    def _init_layers(
        self,
        z_sem_dim: int,
        z_id_dim: int,
        **common_kwargs,
    ):
        self.sem: Callable[[torch.Tensor], torch.Tensor] = conv_nd(
            out_channels=z_sem_dim, **common_kwargs
        )
        self.id: Callable[[torch.Tensor], torch.Tensor] = conv_nd(
            out_channels=z_id_dim, **common_kwargs
        )

    def forward(self, x) -> torch.Tensor:

        z_sem = self.sem(x).flatten(start_dim=1)
        z_id = self.id(x).flatten(start_dim=1)
        cond = torch.cat([z_sem, z_id], dim=1)

        return cond


class SemIDLayerNonLinear(SemIDLayer):

    def _init_layers(self, z_sem_dim: int, z_id_dim: int, conf: "EncoderIDConfig", ch):
        self.sem: Callable[[torch.Tensor], torch.Tensor] = NonLinearHead(
            out_dims=z_sem_dim, dims=conf.dims, ch=ch
        )
        self.id: Callable[[torch.Tensor], torch.Tensor] = NonLinearHead(
            out_dims=z_id_dim, dims=conf.dims, ch=ch
        )
