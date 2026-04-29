from typing import Any

import torch
from torch import nn


class IdentityAugmentation(nn.Identity):

    def __init__(self, n_views: int = 1, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.n_views = n_views

    def sample_n_transforms(self, n_transforms: int, *args, **kwargs):
        return [self for _ in range(n_transforms)]

    def forward(self, input, *args, **kwargs):

        # return n_views copies of the input
        return torch.cat([input for _ in range(self.n_views)], dim=0)
