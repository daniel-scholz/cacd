from dataclasses import dataclass
from typing import Sequence

from diffae.model.diffae import DiffAEConfig, DiffAEModel
from diffae.model.encoders.encoder_id_preserve import EncoderIDConfig
from diffae.model.encoders.encoder_id_preserve_separate import EncoderIDSeparateConfig


@dataclass
class DiffAEIDConfig(DiffAEConfig):
    # define additional parameters here
    # dimension of the semantic feature vector
    enc_z_sem_dim: int = 256
    # dimension of the identity feature vector
    enc_z_id_dim: int = 256

    # encoder use non-linear head
    enc_use_non_linear_head: bool = False

    enc_separate_encoders: bool = False

    def make_model(self):
        return DiffAEIDModel(self)


class DiffAEIDModel(DiffAEModel):
    """Encoder yielding two different feature vectors: z_sem and z_id."""

    def __init__(self, conf: DiffAEIDConfig):
        super().__init__(conf)

        if (conf.enc_z_id_dim + conf.enc_z_sem_dim) != conf.cond_channels:
            raise ValueError(
                "Sum of z_id_dim and z_sem_dim must be equal to cond_channels",
                f"got {conf.enc_z_id_dim} + {conf.enc_z_sem_dim} != {conf.cond_channels}",
            )
        if len(self.intensity_augs_names) == 0:
            raise ValueError("At least one intensity augmentation must be enabled.")

        self.target_ema_updater = None
        self.conf = conf

    @property
    def gin_n_views(self):
        return self.conf.gin_n_views

    def _calculate_roi_sizes(
        self, conf: DiffAEIDConfig, scaling_factor_min: float, scaling_factor_max: float
    ) -> tuple[int, int] | tuple[tuple[int, ...], tuple[int, ...]]:
        if isinstance(conf.image_size, Sequence):
            roi_size_min = tuple([int(s * scaling_factor_min) for s in conf.image_size])
            roi_size_max = tuple([int(s * scaling_factor_max) for s in conf.image_size])
        elif isinstance(conf.image_size, int):
            roi_size_min = int(conf.image_size * scaling_factor_min)
            roi_size_max = int(conf.image_size * scaling_factor_max)
        else:
            raise TypeError(f"image_size must be tuple or int, got {type(conf.image_size)}")

        return roi_size_min, roi_size_max

    def _init_sem_enc_conf(self, conf: DiffAEIDConfig) -> EncoderIDConfig | EncoderIDSeparateConfig:

        base_conf = super()._init_sem_enc_conf(conf)

        if conf.enc_separate_encoders:
            new_conf_init_fn = EncoderIDSeparateConfig
        else:
            new_conf_init_fn = EncoderIDConfig

        new_conf = new_conf_init_fn(
            z_sem_dim=conf.enc_z_sem_dim,
            z_id_dim=conf.enc_z_id_dim,
            use_non_linear_head=conf.enc_use_non_linear_head,
            learnable_downsampling=conf.enc_learnable_downsampling,
            **base_conf.__dict__,
        )

        return new_conf
