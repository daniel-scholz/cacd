"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from diffae.choices import GenerativeType, LossType, ModelMeanType, ModelType, ModelVarType
from diffae.config_base import BaseConfig
from diffae.loss import InfoNCELoss, MaskedNegativesInfoNCELoss
from diffae.model import Model
from diffae.model.diffae_id_preserve import DiffAEIDModel

if TYPE_CHECKING:
    from diffae.diffusion.diffusion import _WrappedModel


@dataclass
class GaussianDiffusionBeatGansConfig(BaseConfig):
    gen_type: GenerativeType
    betas: torch.Tensor
    model_type: ModelType
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType

    # double GIN only
    scanner_loss_weight: float
    content_loss_weight: float
    ortho_loss_weight: float
    cross_correlation_loss_weight: float

    info_nce_loss_agg_fn: Literal["sum", "mean"]
    double_gin_scanner_contrasting: Literal["same", "all", "same_patient_negative"]
    z_sem_dim: int
    z_id_dim: int
    do_rescale_timesteps: bool
    fp16: bool
    train_pred_xstart_detach: bool

    def make_sampler(self):
        return GaussianDiffusionBeatGans(self)


class GaussianDiffusionBeatGans(nn.Module):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(self, conf: GaussianDiffusionBeatGansConfig):
        super().__init__()
        self.conf = conf
        self.model_mean_type = conf.model_mean_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type

        if self.loss_type == LossType.mse:
            if self.model_mean_type == ModelMeanType.eps:
                # (n, c, h, w) => (n, )
                self.mse_loss = nn.MSELoss(reduction="none")
            else:
                raise NotImplementedError()
        elif self.loss_type == LossType.l1:
            # (n, c, h, w) => (n, )
            self.mse_loss = nn.L1Loss(reduction="none")
        else:
            raise NotImplementedError()

        self.do_rescale_timesteps = conf.do_rescale_timesteps
        if self.do_rescale_timesteps:
            self.rescale_timesteps = self._rescale_timesteps
        else:
            self.rescale_timesteps = nn.Identity()

        self.register_buffer("betas", conf.betas, persistent=False)
        self.betas: torch.Tensor
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        self.num_timesteps = self.betas.size(0)

        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0), persistent=False)
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat(
                [
                    torch.ones(1, device=self.alphas_cumprod.device),
                    self.alphas_cumprod[:-1],
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "alphas_cumprod_next",
            torch.cat(
                [
                    self.alphas_cumprod[1:],
                    torch.zeros(1, device=self.alphas_cumprod.device),
                ]
            ),
            persistent=False,
        )
        assert self.alphas_cumprod_prev.size() == torch.Size((self.num_timesteps,))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            torch.log(1.0 - self.alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / self.alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / self.alphas_cumprod - 1),
            persistent=False,
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_variance",
            (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)),
            persistent=False,
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.cat([self.posterior_variance[1][None], self.posterior_variance[1:]])),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)),
            persistent=False,
        )
        # initialize double GIN loss function
        agg_fn = getattr(torch, conf.info_nce_loss_agg_fn)
        match conf.double_gin_scanner_contrasting:
            case "same" | "all":
                self.info_nce_loss = InfoNCELoss(
                    # sum or mean
                    aggregation_fn=agg_fn
                )
            case "same_patient_negative":
                self.info_nce_loss = MaskedNegativesInfoNCELoss(aggregation_fn=agg_fn)

    def training_losses(
        self,
        model: "_WrappedModel",
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        batch_cond: Optional[torch.Tensor] = None,
        **model_kwargs,
    ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x0: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        noise = torch.randn_like(x_0)

        # DEBUG: original image is noised
        x_t = self.q_sample(x_0, t, noise=noise)

        terms = {"x_t": x_t}

        # x_t is static wrt. to the diffusion process
        model_forward = model.forward(
            x_t=x_t.detach(),
            t=self.rescale_timesteps(t),
            x0=x_0.detach(),
            **model_kwargs,
        )
        model_output = model_forward.pred

        _model_output = model_output
        if self.conf.train_pred_xstart_detach:
            _model_output = _model_output.detach()
        # get the pred xstart
        p_mean_var = self.p_mean_variance(
            model=DummyModel(pred=_model_output),
            # gradient goes through x_t
            x_t=x_t,
            t=t,
            clip_denoised=False,
        )
        terms["pred_xstart"] = p_mean_var["pred_xstart"]

        # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        target_types = {
            ModelMeanType.eps: noise,
        }
        target = target_types[self.model_mean_type]

        terms["mse_loss"] = self.mse_loss(model_output.flatten(1), target.flatten(1)).mean(dim=1)
        terms["loss"] = terms["mse_loss"].mean()

        # add identity loss on z_ids from two different views
        if isinstance(model.model, DiffAEIDModel):

            cond = model_forward.cond

            z_sem, z_id = torch.split(cond, [self.conf.z_sem_dim, self.conf.z_id_dim], dim=1)

            n_views = model.model.gin_n_views
            batch_size = x_0.size(0) // n_views

            l_content_loss = self.conf.content_loss_weight
            l_scanner_loss = self.conf.scanner_loss_weight
            l_ortho_loss = self.conf.ortho_loss_weight
            l_cross_correlation = self.conf.cross_correlation_loss_weight

            if l_content_loss > 0:
                # content loss is regular SimCLR to enforce consistent representations for the same patient
                # but different representations for different patients
                labels_content = torch.arange(batch_size, device=z_id.device).repeat(n_views)

                content_attract_mask = labels_content.unsqueeze(0) == labels_content.unsqueeze(1)
                match self.conf.double_gin_scanner_contrasting:
                    case "same_patient_negative":

                        labels_scanner = torch.arange(
                            n_views, device=z_id.device
                        ).repeat_interleave(batch_size)
                        same_scanner_mask = labels_scanner.unsqueeze(0) == labels_scanner.unsqueeze(
                            1
                        )
                        content_loss = self.info_nce_loss(
                            z_id, content_attract_mask, same_scanner_mask
                        )

                    case _:
                        content_loss = self.info_nce_loss(z_id, content_attract_mask)

                terms["content_loss"] = content_loss
                z_id_var = torch.var(z_id.detach(), dim=0).mean()
                terms["z_id_var"] = z_id_var

                terms["loss"] = terms["loss"] + l_content_loss * terms["content_loss"].mean()

            if l_scanner_loss > 0:
                # scanner is somewhat inverse SimCLR
                # images that have been augmented with the same transformation
                # in the same domain should yield the same representation
                # -> indicates which views should be positive
                labels_scanner = torch.arange(n_views, device=z_sem.device).repeat_interleave(
                    batch_size
                )
                unique_scanners: torch.Tensor = torch.unique(batch_cond)
                unique_scanners.sort()

                # repeat scanners
                batch_cond = batch_cond.repeat(n_views)

                match self.conf.double_gin_scanner_contrasting:
                    case "same_patient_negative" | "same":
                        # create labels for patients and scanners
                        scanner_wise_loss = torch.zeros(
                            batch_cond.shape[0],
                            device=z_sem.device,
                        )
                        labels_patient = torch.arange(batch_size, device=z_id.device).repeat(
                            n_views
                        )

                        for i_scanner, cur_scanner in enumerate(unique_scanners):
                            cur_scanner_mask = batch_cond == cur_scanner

                            # negative pairs only come from within the same scanner
                            # ->filter out the current scanner
                            cur_labels_scanner = labels_scanner[cur_scanner_mask]
                            cur_z_sem = z_sem[cur_scanner_mask]
                            cur_labels_patient = labels_patient[cur_scanner_mask]

                            scanner_attract_mask = cur_labels_scanner.unsqueeze(
                                0
                            ) == cur_labels_scanner.unsqueeze(1)
                            # determine whether there is at least one positive pair.
                            # if there is only one subject from the scanner,
                            # ->there are no positive pairs hence the loss should be 0
                            has_one_positive_pair = (
                                scanner_attract_mask[
                                    ~torch.eye(
                                        scanner_attract_mask.size(0),
                                        device=scanner_attract_mask.device,
                                        dtype=torch.bool,
                                    )
                                ]
                                .sum()
                                .sign()
                            )
                            if self.conf.double_gin_scanner_contrasting == "same_patient_negative":
                                # draw negatives only from different scanners
                                # and the same subject
                                same_patient_mask = cur_labels_patient.unsqueeze(
                                    0
                                ) == cur_labels_patient.unsqueeze(1)
                                # negatives come from the same patient
                                cur_scanner_loss = self.info_nce_loss(
                                    cur_z_sem,
                                    scanner_attract_mask,
                                    same_patient_mask,
                                )
                            else:
                                cur_scanner_loss = self.info_nce_loss(
                                    cur_z_sem,
                                    scanner_attract_mask,
                                )

                            cur_scanner_loss = (
                                cur_scanner_loss
                                * has_one_positive_pair  # map to 0 if no positive pairs, else multiply with 1
                            )
                            scanner_wise_loss[cur_scanner_mask] = cur_scanner_loss

                            terms[f"scanner_{unique_scanners[i_scanner]}_loss"] = cur_scanner_loss

                    case "all":
                        raise NotImplementedError(
                            "double_gin_scanner_contrasting 'all' not implemented"
                        )
                        # negative pairs come from all scanners with different augmentations
                        scanners_map = batch_cond.unsqueeze(0) == batch_cond.unsqueeze(1)

                        same_aug_map = labels_scanner.unsqueeze(0) == labels_scanner.unsqueeze(1)
                        pos_indices = []
                        neg_indices = []
                        for i_sample in torch.arange(
                            labels_scanner.shape[0],
                            device=scanners_map.device,
                        ):
                            # positives are
                            #   the same scanner and same augmentation
                            pos_indices.append(
                                torch.where(same_aug_map[i_sample] & scanners_map[i_sample])[0]
                            )
                            # negatives are
                            #   different scanners and different augmentations and
                            #   same scanner and different augmentations
                            neg_indices.append(
                                torch.where(
                                    (~scanners_map[i_sample] & ~same_aug_map[i_sample])
                                    | (scanners_map[i_sample] & ~same_aug_map[i_sample])
                                )[0]
                            )
                        # make 2d indices with row and column indices
                        row_idx_pos = torch.tensor(
                            [j for j, p in enumerate(pos_indices) for _ in range(len(p))]
                        )
                        pos_indices = torch.stack(
                            [
                                row_idx_pos,
                                torch.tensor([i for p in pos_indices for i in p]),
                            ],
                            dim=1,
                        )
                        # filter diagonals from positive indices
                        pos_indices = pos_indices[pos_indices[:, 0] != pos_indices[:, 1]]
                        row_idx_neg = torch.tensor(
                            [j for j, p in enumerate(neg_indices) for _ in range(len(p))]
                        )
                        neg_indices = torch.stack(
                            [
                                row_idx_neg,
                                torch.tensor([i for p in neg_indices for i in p]),
                            ],
                            dim=1,
                        )
                        scanner_wise_loss = info_nce_loss_with_indices(
                            z_sem,
                            pos_indices,
                            neg_indices,
                        )
                terms["scanner_loss"] = scanner_wise_loss

                z_sem_var = torch.var(z_sem.detach(), dim=0).mean()

                terms["z_sem_var"] = z_sem_var

                terms["loss"] = terms["loss"] + l_scanner_loss * terms["scanner_loss"].mean()

            if l_ortho_loss > 0:
                # calculate cosine similarity between z_sem and z_id
                z_sem_norm = z_sem / torch.norm(z_sem, dim=1, keepdim=True)
                z_id_norm = z_id / torch.norm(z_id, dim=1, keepdim=True)

                # calculate cosine similarity between z_sem and z_id for each sample
                cosine_similarity = nn.functional.cosine_similarity(z_sem_norm, z_id_norm, dim=1)

                # calculate the loss: minimize similarity between z_sem and z_id
                ortho_loss = cosine_similarity.mean()
                terms["ortho_loss"] = ortho_loss
                terms["loss"] = terms["loss"] + l_ortho_loss * ortho_loss

            if l_cross_correlation > 0:
                cross_correlation_loss = self.calc_cross_correlation_loss(z_sem, z_id)

                terms["cross_correlation_loss"] = cross_correlation_loss
                terms["loss"] = terms["loss"] + l_cross_correlation * cross_correlation_loss

        return terms

    def calc_cross_correlation_loss(self, z_sem, z_id):
        # normalise each feature to have zero mean and unit variance
        z_sem_norm = self.zero_mean_unit_variance_norm(z_sem)
        z_id_norm = self.zero_mean_unit_variance_norm(z_id)
        # calculate the cross-correlation matrix per sample
        cross_correlation_matrix_samples = z_sem_norm[..., None] @ z_id_norm[..., None, :]
        # take the expectation over the samples
        cross_correlation_matrix = cross_correlation_matrix_samples.mean(dim=0)
        # calculate the loss as the Frobenius norm of the cross-correlation matrix
        cross_correlation_loss = torch.norm(torch.triu(cross_correlation_matrix), p="fro")
        # print(cross_correlation_loss)
        return cross_correlation_loss

    def zero_mean_unit_variance_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor to have zero mean and unit variance for each feature.

        :param x: the input tensor.
        :return: the normalized tensor.
        """

        return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)

    def sample(
        self,
        model: Model,
        shape=None,
        noise=None,
        cond=None,
        x0=None,
        imgs=None,
        clip_denoised=True,
        model_kwargs=None,
        progress=False,
        with_grad=False,
        T_offset=0,
    ):
        """
        Args:
            x0: given for the autoencoder
        """
        if model_kwargs is None:
            model_kwargs = {}
            if self.conf.model_type.has_autoenc():
                model_kwargs["x0"] = x0
                model_kwargs["cond"] = cond
        if imgs is not None:
            model_kwargs["imgs"] = imgs

        if self.conf.gen_type == GenerativeType.ddpm:
            return self.p_sample_loop(
                model,
                shape=shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                progress=progress,
            )
        elif self.conf.gen_type == GenerativeType.ddim:
            return self.ddim_sample_loop(
                model,
                shape=shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                progress=progress,
                with_grad=with_grad,
                T_offset=T_offset,
            )
        else:
            raise NotImplementedError()

    def q_mean_variance(self, x0, t):
        """
        Get the distribution q(x_t | x_0).

        :param x0: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x0's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x0.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x0.shape)
        return mean, variance, log_variance

    def q_sample(self, x0, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x0: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        assert noise.shape == x0.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def q_posterior_mean_variance(self, x0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x0.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x0
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x0.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model: Model,
        x_t,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x0 prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        model_forward = model.forward(x_t=x_t, t=self.rescale_timesteps(t), **model_kwargs)
        model_output = model_forward.pred

        if self.model_var_type in [ModelVarType.fixed_large, ModelVarType.fixed_small]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    torch.cat([self.posterior_variance[1][None], self.betas[1:]]),
                    torch.log(torch.cat([self.posterior_variance[1][None], self.betas[1:]])),
                ),
                ModelVarType.fixed_small: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x_t.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type in [
            ModelMeanType.eps,
        ]:
            if self.model_mean_type == ModelMeanType.eps:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
                )
            else:
                raise NotImplementedError()
            model_mean, _, _ = self.q_posterior_mean_variance(x0=pred_xstart, x_t=x_t, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "model_forward": model_forward,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_xstart_from_scaled_xstart(self, t, scaled_xstart):
        return scaled_xstart * _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, scaled_xstart.shape
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_scaled_xstart(self, x_t, t, scaled_xstart):
        """
        Args:
            scaled_xstart: is supposed to be sqrt(alphacum) * x_0
        """
        # 1 / sqrt(1-alphabar) * (x_t - scaled xstart)
        return (x_t - scaled_xstart) / _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

    def _rescale_timesteps(self, t: torch.Tensor) -> torch.Tensor:

        return t.float() * (1000.0 / self.num_timesteps)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self.rescale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self.rescale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x0=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x0 prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x0 prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x0 predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x0 prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            # t = torch.tensor([i] * shape[0], device=device)
            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x0 or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        NOTE: never used ?
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x0 or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed  (DDIM paper)  (torch.sqrt == torch.sqrt)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model: Model,
        x: torch.Tensor,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = x.device
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.num_timesteps))
        sample = x
        for i in indices:
            t = torch.tensor([i] * len(sample), device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(
                    model,
                    sample,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                sample = out["sample"]
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out["pred_xstart"])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            "sample": sample,
            # (1, ..., T)
            "sample_t": sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            "xstart_t": xstart_t,
            "T": T,
        }

    def ddim_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        with_grad=False,
        T_offset=0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            with_grad=with_grad,
            T_offset=T_offset,
        ):
            final = sample
        return final["sample"].float()

    def ddim_sample_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        with_grad=False,
        # possibility to start at a specific timestep instead of T-1
        T_offset=0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = torch.randn(*shape, device=device)
        indices = torch.arange(self.num_timesteps - 1 - T_offset, end=-1, step=-1, device=device)
        if progress:
            indices = tqdm(indices)

        for t in indices:
            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[t.item()]
            else:
                _kwargs = model_kwargs

            t = t.repeat(img.size(0))
            with torch.set_grad_enabled(with_grad):
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out["t"] = t
                yield out
                img = out["sample"]


def _extract_into_tensor(
    arr: torch.Tensor, timesteps: torch.LongTensor, broadcast_shape: torch.Size
):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr[timesteps].view(-1, *([1] * (len(broadcast_shape) - 1)))
    return res.expand(broadcast_shape)


# define type for ScheduleName
ScheduleName = Literal[
    "linear",
    "cosine",
    "const0.01",
    "const0.015",
    "const0.008",
    "const0.0065",
    "const0.0055",
    "const0.0045",
    "const0.0035",
    "const0.0025",
    "const0.0015",
]


def get_named_beta_schedule(
    schedule_name: ScheduleName, num_diffusion_timesteps: int
) -> torch.Tensor:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    match schedule_name:
        case "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(
                beta_start,
                beta_end,
                num_diffusion_timesteps,
                dtype=torch.float32,
            )
        case "cosine":
            return betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: torch.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
            )
        case str(schedule_name) if "const" in schedule_name:
            const = float(schedule_name.replace("const", ""))
            scale = 1000 / num_diffusion_timesteps
            return torch.tensor([scale * const], dtype=torch.float32).repeat(
                num_diffusion_timesteps
            )
        case _ as unreachable:
            raise ValueError(f"Unknown schedule name: {unreachable}")


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    max_beta=0.999,
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in torch.arange(num_diffusion_timesteps, device=device):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(torch.clamp_max(1 - alpha_bar(t2) / alpha_bar(t1), max=max_beta))
    return torch.stack(betas)


class DummyModel(torch.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, *args, **kwargs):
        return DummyReturn(pred=self.pred)


class DummyReturn(NamedTuple):
    pred: torch.Tensor
