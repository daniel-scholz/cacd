from diffae.model.encoders.encoder import BeatGANsEncoderModel
from diffae.model.encoders.out_layers import SemIDLayerNonLinear


class EncoderIDPreserveModelNonLinear(BeatGANsEncoderModel):
    from diffae.model.encoders.encoder_id_preserve import EncoderIDConfig

    def _init_out_layer(self, conf: EncoderIDConfig, ch: int):
        self.out = SemIDLayerNonLinear(
            z_sem_dim=conf.z_sem_dim,
            z_id_dim=conf.z_id_dim,
            conf=conf,
            ch=ch,
        )
