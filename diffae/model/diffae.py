from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional

import torch
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import RandSpatialCrop
from monai.transforms.spatial.array import Resize
from monai.transforms.utility.array import Identity
from torch import nn

from diffae.model.augmentations.anat_reg_augs import AnatomicalRegionsAugmentation
from diffae.model.augmentations.bias_field import RandBiasFieldCorruptionAugmentation
from diffae.model.augmentations.identity import IdentityAugmentation
from diffae.model.augmentations.rc_aug import RandConvAugmentation
from diffae.model.blocks.resblock import ResBlock
from diffae.model.encoders.encoder import EncoderConfig
from diffae.model.latentnet import MLPSkipNetConfig
from diffae.model.nn import linear, timestep_embedding
from diffae.model.unet import DiffConfig, DiffModel
from diffae.transforms import GammaAugmentation

AugName = Literal["rc", "bf", "gamma", "anat_reg"]


@dataclass(kw_only=True)
class DiffAEConfig(DiffConfig):

    enc_attn_resolutions: tuple[int] = None
    enc_pool: str = "depthconv"
    enc_num_res_block: int = 2
    enc_channel_mult: tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    # range to interpolate between the original and augmented image
    gin_alpha_range: tuple[float, float] = (0.0, 1.0)

    gin_n_views: int = 2
    gin_n_layers: int = 4
    gin_n_hidden_dims: int = 2

    rc_type: Literal["linear", "gin"] = "gin"

    # augmentations to use to create views
    augmentor_type: Literal["gin", "classic", "synth"] = "gin"

    rc_normalization: Literal["fro", "minmax"] = "minmax"
    rc_rotationally_symmetric: bool = False

    intensity_augs_names: tuple[AugName, ...] = ("rc", "bf", "gamma")
    rand_resize_aug: bool = False

    apply_aug_to_fg_only: bool = True

    enc_learnable_downsampling: bool = False

    rc_target_size: int = 2048
    rc_do_updownsampling: bool = True
    rc_resize_mode: str = "bilinear"

    def make_model(self):
        return DiffAEModel(self)


class DiffAEModel(DiffModel):
    def __init__(self, conf: DiffAEConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.t_embed_channels,
        )

        self._encoder = self._init_sem_enc_conf(conf).make_model()

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

        self.intensity_augs_names = conf.intensity_augs_names

        if "rc" in self.intensity_augs_names and "anat_reg" in self.intensity_augs_names:
            raise ValueError(
                "RandConvAugmentation and AnatomicalRegionsAugmentation cannot be used together."
            )

        if "rc" in self.intensity_augs_names:
            self.rcaug = RandConvAugmentation(
                rc_type=conf.rc_type,
                spatial_dims=conf.dims,
                n_layers=conf.gin_n_layers,
                n_hidden_chans=conf.gin_n_hidden_dims,
                alpha_range=conf.gin_alpha_range,
                normalization=conf.rc_normalization,
                rotationally_symmetric=conf.rc_rotationally_symmetric,
                do_updownsampling=conf.rc_do_updownsampling,
                target_size=conf.rc_target_size,
                resize_mode=conf.rc_resize_mode,
            )
        else:
            self.rcaug = IdentityAugmentation()

        use_anat_reg_aug = "anat_reg" in self.intensity_augs_names

        self.anat_reg_aug_init = (
            AnatomicalRegionsAugmentation if use_anat_reg_aug else IdentityAugmentation
        )

        if "bf" in self.intensity_augs_names:
            self.bias_field_corruption = RandBiasFieldCorruptionAugmentation(
                img_size=conf.image_size, dims=conf.dims
            )
        else:
            self.bias_field_corruption = IdentityAugmentation()

        if "gamma" in self.intensity_augs_names:
            self.gamma_transform = GammaAugmentation()
        else:
            self.gamma_transform = IdentityAugmentation()

        # register as buffer to move to device
        self.register_buffer(
            "apply_aug_to_fg_only",
            torch.tensor(conf.apply_aug_to_fg_only, dtype=torch.bool),
        )

        # random resize crop
        if conf.rand_resize_aug:
            scaling_factor_min = 0.5
            scaling_factor_max = 1.0
            roi_size_min, roi_size_max = self._calculate_roi_sizes(
                conf, scaling_factor_min, scaling_factor_max
            )
            self.random_resize_crop = Compose(
                [
                    RandSpatialCrop(
                        roi_size=roi_size_min,
                        max_roi_size=roi_size_max,
                        random_size=True,
                        lazy=True,
                    ),
                    Resize(conf.image_size),
                ]
            )
        else:
            self.random_resize_crop = Identity()

        if "rc" in self.intensity_augs_names:
            self.create_contrast_views = self.create_contrast_views_with_rc
        elif "anat_reg" in self.intensity_augs_names:
            raise NotImplementedError()
        else:
            self.create_contrast_views = IdentityAugmentation(self.gin_n_views)

    @property
    def gin_n_views(self):
        return 1

    def _init_sem_enc_conf(self, conf: DiffAEConfig) -> EncoderConfig:
        """"""
        return EncoderConfig(
            image_size=conf.image_size,
            in_channels=1,  # conf.in_channels,
            model_channels=conf.model_channels,
            out_channels=conf.cond_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions or conf.attention_resolutions),
            use_attention=conf.use_attention,
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        )

    def intensity_augment(
        self,
        imgs: torch.Tensor,
        anat_label_map: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self.apply_aug_to_fg_only: torch.BoolTensor
        sample_view_aug_individually = self.gin_n_views == 1

        gin_n_views = imgs.size(0) if self.gin_n_views == 1 else self.gin_n_views

        bias_field_corruptions = self.bias_field_corruption.sample_n_transforms(
            gin_n_views, device=imgs.device, dtype=imgs.dtype
        )

        gamma_transforms = self.gamma_transform.sample_n_transforms(gin_n_views)

        # compose transforms
        transforms = [Compose([bf, g]) for bf, g in zip(bias_field_corruptions, gamma_transforms)]

        imgs = self.create_contrast_views(
            imgs,
            # anat_label_map,
            sample_view_aug_individually=sample_view_aug_individually,
        )

        # apply transforms to chunks of the image
        # each chunk corresponds to a different view
        img_views_chunked = list(imgs.chunk(gin_n_views, dim=0))

        for i, (img_chunk, transform) in enumerate(zip(img_views_chunked, transforms)):
            img_views_chunked[i] = transform(img_chunk)  # type: ignore

        imgs = torch.cat(img_views_chunked, dim=0)

        # random resize crop
        imgs = torch.stack([self.random_resize_crop(img) for img in imgs])
        return imgs

    def noise_to_cond(self, noise: torch.Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self._encoder(x)
        return {"cond": cond}

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0=None,
        imgs=None,
        cond=None,
        noise=None,
    ):
        """
        Apply the model to an input batch.

        Args:
            x0: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if cond is None:
            cond = self.encode(x0)["cond"]

        t_emb_sin = timestep_embedding(t, self.conf.model_channels)

        t_emb = self.time_embed.forward(t_emb_sin)

        # where in the model to supply time conditions
        enc_time_emb = t_emb
        mid_time_emb = t_emb
        dec_time_emb = t_emb
        # where in the model to supply style conditions
        enc_cond_emb = cond
        mid_cond_emb = cond
        dec_cond_emb = cond

        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if imgs is not None:
            # repeat imgs to match batch size
            imgs = imgs.repeat(x_t.size(0) // imgs.size(0), *[1] * len(imgs.shape[1:]))
            x_t = torch.cat([x_t, imgs], dim=1)

        # input blocks
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                x_t = self.input_blocks[k](x_t, t_emb=enc_time_emb, cond=enc_cond_emb)

                # print(i, j, h.shape)
                hs[i].append(x_t)
                k += 1
        assert k == len(self.input_blocks)

        # middle blocks
        x_t = self.middle_block(x_t, t_emb=mid_time_emb, cond=mid_cond_emb)

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                x_t = self.output_blocks[k](
                    x_t, t_emb=dec_time_emb, cond=dec_cond_emb, lateral=lateral
                )
                k += 1

        pred = self.out(x_t)
        return AutoencReturn(pred=pred, cond=cond)

    @property
    def all_conditioned_modules(self) -> dict[str, nn.Module]:
        modules_dict = {}

        def extract_res_blocks(blocks: nn.Sequential | nn.ModuleList, prefix: str):
            for i, block in enumerate(blocks):
                for j, module in enumerate(block):
                    if isinstance(module, ResBlock):
                        modules_dict[f"{prefix}_{i}_{j}"] = module

        extract_res_blocks(self.input_blocks, "input")
        extract_res_blocks(nn.ModuleList([self.middle_block]), "middle")
        extract_res_blocks(self.output_blocks, "output")

        return modules_dict

    def create_contrast_views_with_rc(
        self,
        img: torch.Tensor,
        sample_view_aug_individually: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        in_channels = img.shape[1]

        if sample_view_aug_individually:
            repeats = 1
            rc_transforms = self.rcaug.sample_n_transforms(
                in_channels=in_channels,
                out_channels=in_channels,
                n_transforms=img.size(0),
                device=img.device,
                dtype=img.dtype,
            )
            # apply a new augmentation to each view

            img_views = torch.cat(
                [rcaug(img_[None]) for rcaug, img_ in zip(rc_transforms, img)], dim=0
            )
        else:
            repeats = self.gin_n_views
            rc_transforms = self.rcaug.sample_n_transforms(
                in_channels=in_channels,
                out_channels=in_channels,
                n_transforms=self.gin_n_views,
                device=img.device,
                dtype=img.dtype,
            )
            img_views = torch.cat([rc(img) for rc in rc_transforms], dim=0)

        # mask to apply augmentation only to the foreground if desired
        fg_mask = (img > img.min()).repeat(
            repeats, *[1] * (img.dim() - 1)
        ) + ~self.apply_aug_to_fg_only
        img_views[~fg_mask] = img.min()  # type: ignore
        return img_views


class AutoencReturn(NamedTuple):
    pred: torch.Tensor
    cond: torch.Tensor


class TimeEmbed(nn.Module):

    def __init__(self, time_channels: int, time_out_channels: int):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

    def forward(self, t_emb: torch.Tensor):

        return self.time_embed(t_emb)
