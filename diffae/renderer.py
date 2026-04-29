import torch

from diffae.choices import TrainMode
from diffae.config import TrainConfig
from diffae.diffusion import Sampler
from diffae.model.diffae import DiffAEModel


def render_uncondition(
    conf: TrainConfig,
    model: DiffAEModel,
    x_T,
    sampler: Sampler,
    latent_sampler: Sampler,
    clip_latent_noise: bool = False,
):
    device = x_T.device
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.can_sample()
        return sampler.sample(model=model, noise=x_T)
    elif conf.train_mode.is_latent_diffusion():
        if conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = torch.randn(len(x_T), conf.net_cond_channels, device=device)
        else:
            raise NotImplementedError()

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        cond = latent_sampler.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=conf.latent_clip_sample,
        )

        # the diffusion on the model
        return sampler.sample(model=model, noise=x_T, cond=cond)
    else:
        raise NotImplementedError()


def render_condition(
    conf: TrainConfig,
    model: DiffAEModel,
    x_T,
    sampler: Sampler,
    x0=None,
    cond=None,
    with_grad=False,
    T_offset=0,
    imgs=None,
):
    if conf.in_channels == 2:
        if imgs is None:
            imgs = x0

    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        # returns {'cond', 'cond2'}
        if cond is None:
            cond = model.encode(x0)
        return sampler.sample(
            model=model,
            noise=x_T,
            model_kwargs=cond,
            with_grad=with_grad,
            T_offset=T_offset,
            imgs=imgs,
        )
    else:
        raise NotImplementedError()
