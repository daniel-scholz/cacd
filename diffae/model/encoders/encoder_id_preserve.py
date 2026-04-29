from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

from diffae.model.encoders.encoder import BeatGANsEncoderModel, EncoderConfig
from diffae.model.encoders.out_layers import IDLayer, SemIDLayer, SemLayer
from diffae.model.nn import adaptive_avg_pool_nd, conv_nd, normalization

if TYPE_CHECKING:
    from diffae.model.encoders.encoder_id_preserve_nonlinear import EncoderIDPreserveModelNonLinear


@dataclass
class EncoderIDConfig(EncoderConfig):
    # dimension of the semantic feature vector
    z_sem_dim: int = 256
    # dimension of the identity feature vector
    z_id_dim: int = 256

    use_non_linear_head: bool = False

    learnable_downsampling: bool = False

    def make_model(self):
        if self.use_non_linear_head:

            print("Using non-linear head for encoder")
            return EncoderIDPreserveModelNonLinear(self)
        return EncoderIDPreserveModel(self)


class EncoderIDPreserveModel(BeatGANsEncoderModel):
    def _init_out_layer(self, conf: EncoderIDConfig, ch: int):
        if conf.z_sem_dim > 0 and conf.z_id_dim > 0:
            last_layer = SemIDLayer(
                z_sem_dim=conf.z_sem_dim,
                z_id_dim=conf.z_id_dim,
                dims=conf.dims,
                in_channels=ch,
                kernel_size=1,
            )
        elif conf.z_sem_dim > 0:
            last_layer = SemLayer(
                z_dim=conf.z_sem_dim,
                dims=conf.dims,
                in_channels=ch,
                kernel_size=1,
            )
        elif conf.z_id_dim > 0:
            last_layer = IDLayer(
                z_dim=conf.z_id_dim,
                dims=conf.dims,
                in_channels=ch,
                kernel_size=1,
            )

        smallest_feat_map_size = 4 if 64 == conf.image_size or 64 in conf.image_size else 8
        if not conf.learnable_downsampling:
            smallest_feat_map_size = 1

        out_layers = [
            normalization(ch),
            nn.SiLU(),
            adaptive_avg_pool_nd(conf.dims, output_size=smallest_feat_map_size),
        ]

        # interpolate feature map to 4x4
        # 4x4 conf layer
        if conf.learnable_downsampling:
            out_layers.append(conv_nd(conf.dims, ch, ch, smallest_feat_map_size, 1, 0))
        else:
            out_layers.append(nn.Identity())
        out_layers.append(last_layer)

        self.out = nn.Sequential(*out_layers)
