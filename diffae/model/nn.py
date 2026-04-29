"""
Various utilities for neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    # @torch.jit.script
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs) -> nn.Conv1d | nn.Conv2d | nn.Conv3d:
    """
    Create a 1D, 2D, or 3D convolution module.
    """

    return getattr(nn, f"Conv{dims}d")(*args, **kwargs)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    return getattr(nn, f"AvgPool{dims}d")(*args, **kwargs)


def adaptive_avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    return getattr(nn, f"AdaptiveAvgPool{dims}d")(*args, **kwargs)


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)
    # return GroupNorm32(channels, channels)


def timestep_embedding(t: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def torch_checkpoint(func, args, flag, preserve_rng_state=False) -> torch.Tensor:
    # torch's gradient checkpoint works with automatic mixed precision, given torch >= 1.8
    if flag:
        return torch.utils.checkpoint.checkpoint(
            func, *args, preserve_rng_state=preserve_rng_state
        )  # type: ignore
    else:
        return func(*args)
