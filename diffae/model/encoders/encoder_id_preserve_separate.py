from dataclasses import dataclass

import torch
from lightning.pytorch.utilities.model_summary.model_summary import LayerSummary
from torch import nn

from diffae.model.encoders.encoder import BeatGANsEncoderModel
from diffae.model.encoders.encoder_id_preserve import EncoderIDConfig


@dataclass
class EncoderIDSeparateConfig(EncoderIDConfig):
    def make_model(self):
        return EncoderIDSeparateModel(self)


class EncoderIDSeparateModel(nn.Module):
    def __init__(self, conf: EncoderIDSeparateConfig):
        super().__init__()
        # half the number of channels (ie parameters) to end up with the same number of parameters

        _full_channels_encoder = BeatGANsEncoderModel(conf)
        full_num_params = LayerSummary(_full_channels_encoder).num_parameters

        new_conf = conf.clone()
        new_conf.out_channels = conf.out_channels // 2
        new_conf.model_channels = conf.model_channels // 2

        self.conf = new_conf

        # half the model channels

        self.sem_encoder = BeatGANsEncoderModel(new_conf)
        self.id_encoder = BeatGANsEncoderModel(new_conf)

        half_num_params = LayerSummary(self.sem_encoder).num_parameters

        print(
            f"Full model has {full_num_params} parameters,",
            f"separate encoder has {half_num_params} parameters",
        )
        print(f"Param ratio: {half_num_params / full_num_params}")

    def forward(self, x: torch.Tensor, return_2d_feature=False, *args, **kwargs):
        sem = self.sem_encoder(x, return_2d_feature, *args, **kwargs)
        id = self.id_encoder(x, return_2d_feature, *args, **kwargs)

        if return_2d_feature:
            sem, sem_2d = sem
            id, id_2d = id

            h = torch.cat([sem, id], dim=1)
            h_2d = torch.cat([sem_2d, id_2d], dim=1)
            return h, h_2d

        h = torch.cat([sem, id], dim=1)
        return h
