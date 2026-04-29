from typing import Optional

from diffae.choices import GenerativeType, ModelName
from diffae.config import TrainConfig


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = "linear"
    conf.data_names = ("ffhq",)

    conf.metrics_ema_every_steps = 200_000
    conf.metrics_every_steps = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn_resolutions = (16,)
    conf.net_beatgans_attn_head = 1

    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = "linear"
    conf.data_names = ("ffhq",)

    conf.metrics_ema_every_steps = 200_000
    conf.metrics_every_steps = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn_resolutions = (16,)
    conf.net_beatgans_attn_head = 1

    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)

    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.dims = 2
    conf.make_model_conf()
    return conf


def autoenc_base_3d():
    """
    base configuration for all Diff-AE models.
    """
    conf = autoenc_base()
    conf.dims = 3
    conf.make_model_conf()
    return conf


def mri_base_autoenc():
    conf = autoenc_base_3d()
    conf.in_channels = 1
    conf.model_out_channels = 1
    conf.img_size = (192, 224, 192)
    conf.net_ch = 32
    conf.ema_decay = 0.999

    # how often to generate samples to tensorboard/disk
    conf.vis_every_steps = 50000

    # how often to evaluate image quality metrics
    conf.metrics_every_steps = 50000
    # save checkpoint frequency
    conf.save_every_steps = 25000
    # eval twice as often as saving the model
    conf.eval_every_steps = 10000

    conf.data_split_mode = "stratified"

    # how often to save checkpoints
    conf.save_every_steps = conf.metrics_every_steps
    conf.make_model_conf()
    return conf


def sequence_contrastive(name: str):
    conf = mri_base_autoenc()
    conf.name = name
    conf.data_names = ("glioma_public",)
    conf.in_channels = 4
    conf.model_out_channels = 4

    conf.make_model_conf()
    return conf


def scanner_harm(name: str):
    conf = mri_base_autoenc()

    conf.fit_sites = {
        "ixi": ("Guys", "HH"),
        "oasis3": tuple(),
    }
    conf.test_sites = {
        "ixi": ("IOP",),
        "oasis3": tuple(),
    }

    conf.name = name
    conf.data_names = ("ixi",)
    conf.data_split_mode = "loaded"

    conf.net_use_attn = False

    conf.model_name = ModelName.beatgans_autoenc_id

    conf.total_steps = 2_500_000

    conf.make_model_conf()

    return conf


def multi_contrast_representation(): ...


templates_dict = {
    "glioma_public": sequence_contrastive,
    "scanner_harm": scanner_harm,
}
