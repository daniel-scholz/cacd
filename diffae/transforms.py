from typing import Hashable, Sequence

import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms.intensity.array import AdjustContrast
from torch import nn


class ScaleIntensityRangePercentilesForegroundd:
    def __init__(
        self,
        keys: Sequence[Hashable],
        channel_wise: bool = False,
        lower: float = 0.0,
        upper: float = 1.0,
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
    ):
        assert 0.0 <= lower < upper <= 1.0, "percentiles should be in [0, 1] and lower < upper"
        self.keys = keys
        self.channel_wise = channel_wise
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def _scale_intensity_range_percentiles(
        self, img: torch.Tensor, brainmask: torch.Tensor
    ) -> torch.Tensor:
        """Scale intensity range of image."""
        # compute percentiles
        assert brainmask.dtype == torch.bool, "brainmask should be boolean"
        img_flat = img.flatten(start_dim=1)
        bm_flat = brainmask.flatten(start_dim=1)

        if self.channel_wise:  # channel dim is 0th dim
            p_lower = torch.zeros(img_flat.shape[0], device=img.device).view(
                -1, *([1] * (img.dim() - 1))
            )
            p_upper = torch.zeros(img_flat.shape[0], device=img.device).view(
                -1, *([1] * (img.dim() - 1))
            )
            for i_channel, (i_flat, b_flat) in enumerate(zip(img_flat, bm_flat)):
                # insert percentile values
                p_lower[i_channel], p_upper[i_channel] = self._get_fg_quantiles(i_flat, b_flat)
        else:
            p_lower, p_upper = self._get_fg_quantiles(img_flat, bm_flat)

        # scale image to be in range [b_min, b_max]
        img = (img - p_lower) / (p_upper - p_lower) * (self.b_max - self.b_min) + self.b_min

        if self.clip:
            img = torch.clamp(img, self.b_min, self.b_max)

        return img

    def _get_fg_quantiles(
        self, img_flat: torch.Tensor, brainmask_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # brainmask is boolean tensor
        img_flat_fg = img_flat[brainmask_flat]
        p_lower = torch.quantile(img_flat_fg, self.lower)
        p_upper = torch.quantile(img_flat_fg, self.upper)
        return p_lower, p_upper

    def __call__(self, data_dict):
        # get brainmask from data_dict
        brainmask = data_dict["brainmask"]
        # remove brainmask from keys
        img_keys = [k for k in self.keys if k != "brainmask"]

        for k in img_keys:
            img = data_dict[k]
            data_dict[k] = self._scale_intensity_range_percentiles(img, brainmask)
        return data_dict


def correct_bias_field(img: torch.Tensor) -> torch.Tensor:
    img_ndim = img.ndim
    img = img.squeeze()
    needs_unsqueeze = img_ndim != img.ndim

    # normalize image
    img_dtype = img.dtype
    img_min = img.min()
    img_max = img.max()
    img = img - img_min
    img = img / (img_max - img_min)
    mask = img > 0

    raw_img_sitk = sitk.GetImageFromArray(img.numpy().astype(np.float32))
    head_mask = sitk.GetImageFromArray(mask.numpy().astype(np.uint8))

    shrinkFactor = 1

    # convert torch tensor to SimpleITK image
    inputImage = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()

    corrected_image_shrink = bias_corrector.Execute(inputImage, maskImage)

    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
    corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)

    # convert back to torch tensor
    corrected_image = torch.from_numpy(sitk.GetArrayFromImage(corrected_image_full_resolution))
    if needs_unsqueeze:
        while corrected_image.ndim < img_ndim:
            corrected_image = corrected_image.unsqueeze(0)

    # clamp values to [0, 1]

    # rescale image to original range
    corrected_image = corrected_image * (img_max - img_min) + img_min
    return corrected_image.to(dtype=img_dtype)


class GammaAugmentation(nn.Module):

    def __init__(self, gamma_std=0.5):
        super().__init__()
        self.gamma_std = gamma_std
        self.transform = self.sample_gamma_transform()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    def sample_gamma_transform(
        self,
    ):
        rng = np.random.default_rng()
        log_gamma = rng.normal(loc=0, scale=self.gamma_std)
        gamma = np.exp(log_gamma)

        return AdjustContrast(gamma=gamma)

    def sample_n_transforms(self, n: int):
        return [type(self)(self.gamma_std) for _ in range(n)]
