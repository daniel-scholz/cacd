from enum import Enum

from torch import nn

from diffae.model.blocks.cond_resblock import TimeStyleCondResBlock
from diffae.model.blocks.resblock import ResBlock


class ScaleAt(Enum):
    after_norm = "afternorm"


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, t_emb=None, cond=None, lateral=None):
        for layer in self:
            match layer:
                case TimeStyleCondResBlock():
                    x = layer(x, t_emb=t_emb, cond=cond, lateral=lateral)
                case ResBlock():
                    x = layer(x, lateral=lateral)
                case _:
                    x = layer(x)

        return x
