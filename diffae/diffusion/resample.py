from abc import ABC, abstractmethod

import torch
from torch import nn


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(nn.Module, ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self) -> torch.Tensor:
        """
        Get a torch tensor of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / torch.sum(w)
        indices = torch.multinomial(p, batch_size, replacement=True)
        weights = 1 / (len(p) * p[indices])
        return indices, weights

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:

        return self.sample(batch_size)


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps: int):
        super().__init__()
        # register buffer
        self.register_buffer("_weights", torch.ones(num_timesteps), persistent=False)

    def weights(self) -> torch.Tensor:
        return self._weights
