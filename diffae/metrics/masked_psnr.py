from typing import Optional, Tuple, Union

import torch
import torchmetrics.image


class MaskedPSNR(torchmetrics.image.PeakSignalNoiseRatio):
    def __init__(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs):

        super().__init__(dim=0, **kwargs)

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):

        outputs = []
        for p, t, m in zip(preds, target, mask):
            p = p[m]
            t = t[m]
            outputs.append(super().forward(p, t))

        return torch.stack(outputs)
