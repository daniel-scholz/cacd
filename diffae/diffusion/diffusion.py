from dataclasses import dataclass

import torch

from diffae.diffusion.base import (
    DummyModel,
    GaussianDiffusionBeatGans,
    GaussianDiffusionBeatGansConfig,
)
from diffae.model import Model
from diffae.model.diffae import DiffAEModel
from diffae.model.diffae_id_preserve import DiffAEIDModel


def space_timesteps(num_timesteps: int, section_counts: str | list[int]) -> set[int]:
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


@dataclass
class SpacedDiffusionBeatGansConfig(GaussianDiffusionBeatGansConfig):
    use_timesteps: set[int]

    def make_sampler(self):
        return SpacedDiffusionBeatGans(self)


class SpacedDiffusionBeatGans(GaussianDiffusionBeatGans):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, conf: SpacedDiffusionBeatGansConfig):
        self.conf = conf
        self.use_timesteps = set(conf.use_timesteps)
        # how the new t's mapped to the old t's
        self.timestep_map = []
        self.original_num_steps = len(conf.betas)

        base_diffusion = GaussianDiffusionBeatGans(conf)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                # getting the new betas of the new timesteps
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        conf.betas = torch.stack(new_betas)
        super().__init__(conf)

    def p_mean_variance(self, model: Model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model: Model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model: Model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.do_rescale_timesteps, self.original_num_steps
        )

    def _rescale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    """
    converting the supplied t's to the old t's scales.
    """

    def __init__(
        self,
        model: DiffAEModel | DiffAEIDModel,
        timestep_map: list[int],
        rescale_timesteps: bool,
        original_num_steps: int,
    ):
        self.model = model
        if not isinstance(model, DummyModel):
            device = next(iter(model.parameters())).device
        else:
            device = model.pred.device

        self.timestep_map = torch.tensor(timestep_map, device=device, dtype=torch.long)
        self._rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        Args:
            t:
                - t's with differrent ranges (can be << T due to smaller eval T)
                - need to be converted to the original t's
            t_cond: the same as t but can be of different values
        """
        t = self.rescale_timesteps(t)

        return self.model(x_t=x_t, t=t, **kwargs)

    def rescale_timesteps(self, t):
        new_ts = self.timestep_map[t]
        if self._rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return new_ts

    def __getattr__(self, name):
        # allow for calling the model's methods
        if hasattr(self.model, name):
            func = getattr(self.model, name)
            return func
        raise AttributeError(name)
