from typing import Literal

import torch
import torch.nn as nn

from diffae.model.nn import conv_nd


class GlassOfGIN(nn.Module):
    """Single GIN transformation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        n_hidden_chans,
        spatial_dims,
        n_layers,
        upsampling_fn,
        downsampling_fn,
        rng: torch.Generator,
        rotationally_symmetric: bool = False,
        normalization: Literal["fro", "minmax"] = "minmax",
        alpha_range: tuple[float, float] = (0.0, 1.0),
        kernel_size=3,
        padding=1,
        **conv_kwargs,
    ):
        super().__init__()
        self.rng = rng

        self.upsampling_fn = upsampling_fn
        self.downsampling_fn = downsampling_fn
        layers = []
        for i_iter in range(n_layers):
            # intialize conv layer
            rand_conv = conv_nd(
                spatial_dims,
                in_channels=in_channels if i_iter == 0 else n_hidden_chans,
                out_channels=n_hidden_chans if i_iter != n_layers - 1 else out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=0,
                padding_mode="replicate",
                **conv_kwargs,
            )
            # init weights with N(0,1)
            self.init_rand_convs(rand_conv, rotationally_symmetric)

            layers.append(rand_conv)
            if i_iter != n_layers - 1:
                layers.append(self.non_linearity())  # default value: 0.01
                # layers.append(nn.SiLU())
        # print(f"Layers in GIN: {layers}")
        self.transform = nn.Sequential(*layers)
        self.rand_interpolate = RandInterpolate(alpha_range=alpha_range, rng=self.rng)

        match normalization:
            case "fro":
                self.normalize_image = self.normalize_image_fro
            case "minmax":
                self.normalize_image = self.normalize_image_minmax

    def non_linearity(self):
        return nn.LeakyReLU(negative_slope=1e-2)

    def init_rand_convs(
        self, rand_conv: nn.Conv1d | nn.Conv2d | nn.Conv3d, rotationally_symmetric: bool
    ):
        in_channels, out_channels = (
            rand_conv.weight.size(1),
            rand_conv.weight.size(0),
        )

        if not rotationally_symmetric:
            nn.init.normal_(
                rand_conv.weight,
                mean=0,
                std=1,
                generator=self.rng,
            )

        if rotationally_symmetric:
            with torch.no_grad():
                if isinstance(rand_conv, nn.Conv1d):
                    raise NotImplementedError("1D not implemented")
                if isinstance(rand_conv, nn.Conv2d):

                    # Define the size of the image
                    rand_conv_symmetric = sample_symmetric_rand_conv_2d(
                        rand_conv, in_channels, out_channels, self.rng
                    )

                    # assign weight to rand_conv
                    rand_conv.weight.data = rand_conv_symmetric

    def forward(self, og_img: torch.Tensor) -> torch.Tensor:
        """Augment, interpolate, and normalize the original image."""
        # GIN augmented the original image
        img_size = og_img.shape[2:]
        # remove channel dimension -> allows to upsample a batch of images
        gin_img = self.upsampling_fn(og_img.squeeze(1)).unsqueeze(1)
        with torch.no_grad():
            gin_img = self.transform(gin_img)
        gin_img = self.downsampling_fn(gin_img, img_size)

        # interpolate between original and augmented image
        # alpha is constant for all images in the batch to mimic scanner properties
        gin_img = self.rand_interpolate(og_img, gin_img)

        # normalize image to have same frobenius norm as original image
        gin_img = self.normalize_image(og_img, gin_img)
        return gin_img

    def normalize_image_fro(self, og_img: torch.Tensor, gin_img: torch.Tensor) -> torch.Tensor:
        gin_img = gin_img / torch.norm(gin_img, p="fro") * torch.norm(og_img, p="fro")
        return gin_img

    def normalize_image_minmax(self, og_img: torch.Tensor, gin_img: torch.Tensor) -> torch.Tensor:
        """minmax normalize the batch to [0, 1] and scale to [-1, 1]"""

        gin_img = (gin_img - gin_img.min()) / (gin_img.max() - gin_img.min())

        # scale to -1 and 1
        gin_img = gin_img * 2 - 1
        return gin_img


class RandInterpolate(nn.Module):
    def __init__(self, rng: torch.Generator, alpha_range: tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.alpha: torch.Tensor

        self.alpha_range = alpha_range
        self.rng = rng
        self.register_buffer("alpha", self.sample_alpha(1))

    def sample_alpha(self, n_samples):
        """uniform randomly sample batch size many alphas between 0 and 1"""
        if self.alpha_range[0] != self.alpha_range[1]:
            # sample alpha from uniform distribution in range [alpha_range[0], alpha_range[1]]
            alpha = (
                torch.rand(n_samples, device=self.rng.device, generator=self.rng)
                * (self.alpha_range[1] - self.alpha_range[0])
                + self.alpha_range[0]
            )
        else:
            # set alpha to alpha_range[0]
            alpha = torch.ones(n_samples, device=self.rng.device) * self.alpha_range[0]

        return alpha

    def forward(self, og_img: torch.Tensor, gin_img: torch.Tensor) -> torch.Tensor:
        # mix the original image with the augmented image
        broadcast_ones = [1] * (og_img.ndim - 1)
        alpha = self.alpha.view(-1, *broadcast_ones).to(dtype=og_img.dtype)

        # interpolate between original and augmented image
        gin_img = alpha * gin_img + (1 - alpha) * og_img
        return gin_img


@torch.no_grad()
def sample_symmetric_rand_conv_2d(
    rand_conv: nn.Conv2d, in_channels: int, out_channels: int, rng: torch.Generator
) -> torch.Tensor:
    width, height = (
        rand_conv.weight.shape[-2],
        rand_conv.weight.shape[-1],
    )

    center_x, center_y = width / 2, height / 2

    # Create a meshgrid
    x = torch.linspace(0, width, width)
    y = torch.linspace(0, height, height)
    c_in = torch.linspace(0, in_channels, in_channels)
    c_out = torch.linspace(0, out_channels, out_channels)
    C_out, C_in, X, Y = torch.meshgrid(c_out, c_in, x, y, indexing="ij")

    distance = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    dist_bins = torch.linspace(0, torch.max(distance), width)
    digitized_dist_map = torch.bucketize(distance, dist_bins, right=True) - 1
    num_chans = torch.tensor(in_channels * out_channels, device=rand_conv.weight.device)

    rand_conv_symmetric = torch.zeros_like(rand_conv.weight.data)
    for k in range(width):
        # sample from normal distribution zero mean and std 1
        cur_dist_mask = digitized_dist_map == k
        n_cur = cur_dist_mask.sum()
        rand_conv_symmetric[cur_dist_mask] = torch.randn(  # type: ignore
            num_chans,
            device=rand_conv.weight.device,
            dtype=rand_conv.weight.dtype,
            generator=rng,
        ).repeat_interleave(n_cur // num_chans)

    return rand_conv_symmetric
