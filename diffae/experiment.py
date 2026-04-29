import copy
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import wandb
import wandb.util
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.utils import make_grid
from tqdm import tqdm

from diffae.callbacks import MetricsModelCheckpoint
from diffae.config import OptimizerType, TrainConfig, TrainMode
from diffae.model.diffae_id_preserve import DiffAEIDModel
from diffae.renderer import render_condition, render_uncondition
from diffae.vis_utils import grid_and_log_img, plot_latents, plt_to_np, project_latents

os.environ["WANDB__SERVICE_WAIT"] = "300"

torch.set_float32_matmul_precision("high")


type ZSem = torch.Tensor
type ZId = torch.Tensor


class LitModel(L.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            L.seed_everything(conf.seed, workers=True)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        self.model = conf.model_conf.make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print("Model params: %.2f M" % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = (
                conf.make_latent_eval_diffusion_conf().make_sampler()
            )
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # initial variables for consistent sampling
        self.noise = None

        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",  # default
            # => input in [-1, 1]
            normalize=False,  # default
        )

        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=(-1, 1))
        self.ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=(-1, 1), kernel_size=7
        )
        self.mse_fn = MeanSquaredError()
        self._fid_fn = FrechetInceptionDistance(normalize=True)  # => input in [0, 1]
        self.psnr_fn = PeakSignalNoiseRatio(data_range=(-1, 1))

    def render(
        self,
        noise: torch.Tensor,
        cond: Optional[dict[str, torch.Tensor]] = None,
        T: Optional[int] = None,
        with_grad: bool = False,
        T_offset: int = 0,
        imgs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler().to(self.device)

        if cond is not None:
            pred_img = render_condition(
                self.conf,
                self.ema_model,
                noise,
                sampler=sampler,
                cond=cond,
                with_grad=with_grad,
                T_offset=T_offset,
                imgs=imgs,
            )
        else:
            pred_img = render_uncondition(
                self.conf, self.ema_model, noise, sampler=sampler, latent_sampler=None
            )
        return pred_img

    def render_differentiable(self, *args, **kwargs):
        return self.render(*args, with_grad=True, **kwargs)

    def encode(self, x) -> dict[str, torch.Tensor]:
        return self._encode(x, self.model)

    def encode_ema(self, x) -> dict[str, torch.Tensor]:
        return self._encode(x, self.ema_model)

    def _encode(self, x: torch.Tensor, model: DiffAEIDModel) -> dict[str, torch.Tensor]:
        assert self.conf.model_type.has_autoenc()
        cond = model.encode(x)
        return cond

    def _encode_stochastic(
        self,
        x: torch.Tensor,
        model: DiffAEIDModel,
        cond: torch.Tensor,
        T=None,
        noise_steps=None,
        imgs=None,
    ) -> torch.Tensor:
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler().to(x.device)
        if noise_steps is None:
            noise_steps = T or self.conf.T_eval

        out = sampler.ddim_reverse_sample_loop(
            model, x, model_kwargs={"cond": cond, "imgs": imgs}
        )
        # -1 to convert to 0-indexed
        # out["sample_t"][T-1] == out["sample"] holds
        return out["sample_t"][noise_steps - 1]

    def encode_stochastic(
        self, x: torch.Tensor, cond: torch.Tensor, T=None, noise_steps=None, imgs=None
    ) -> torch.Tensor:
        return self._encode_stochastic(x, self.model, cond, T, noise_steps, imgs)

    def encode_stochastic_ema(
        self, x: torch.Tensor, cond: torch.Tensor, T=None, noise_steps=None, imgs=None
    ) -> torch.Tensor:
        return self._encode_stochastic(x, self.ema_model, cond, T, noise_steps, imgs)

    def reconstruct(self, img: torch.Tensor, T=None) -> torch.Tensor:
        cond = self.encode_ema(img)["cond"]
        xT = self.encode_stochastic_ema(img, cond, T)
        return self.render(xT, {"cond": cond}, T=T)

    def forward(self, noise=None, x0=None, ema_model: bool = False):
        if ema_model:
            model = self.ema_model
        else:
            model = self.model
        raise NotImplementedError(f"forward not implemented for {model}")
        gen = self.eval_sampler.sample(model=model, noise=noise, x0=x0)
        return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """

        train_datasets = self.conf.make_datasets(
            self.conf.data_paths,
            split="train",  # type: ignore
        )
        val_datasets = self.conf.make_datasets(self.conf.data_paths, split="val")  # type: ignore
        test_datasets = self.conf.make_datasets(self.conf.data_paths, split="test")

        self.train_data = torch.utils.data.ConcatDataset(train_datasets)
        self.val_data = torch.utils.data.ConcatDataset(val_datasets)
        self.test_data = torch.utils.data.ConcatDataset(test_datasets)

        print("train data:", len(self.train_data))
        print("val data:", len(self.val_data))
        print("test data:", len(self.test_data))

    def shared_loader(self, split: Literal["train", "val", "test"], drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        dataset = getattr(self, f"{split}_data")
        dataloader = conf.make_loader(
            dataset, shuffle=split == "train", drop_last=drop_last and split == "train"
        )
        return dataloader

    def train_dataloader(self):
        return self.shared_loader(split="train")

    def val_dataloader(self):
        return self.shared_loader(split="val")

    def test_dataloader(self):
        return self.shared_loader(split="test")

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """

        return self.conf.batch_size

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop?
        used with gradient_accum > 1
        and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def is_first_accum(self, batch_idx):
        """
        is it the first gradient accumulation loop?
        """
        return batch_idx % self.conf.accum_batches == 0

    def training_step(self, batch, batch_idx) -> dict:
        # log global step scaled by world size, i.e. multiple GPUs
        global_step_scaled = self.global_step * self.trainer.world_size
        self.log("trainer/global_step_scaled", global_step_scaled, rank_zero_only=True)

        return self.shared_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch, batch_idx) -> dict:
        shared_out = self.shared_step(batch, batch_idx, step_name="val")

        # imgs
        batch_imgs: torch.Tensor = batch["img"]
        aug_imgs = self.model.intensity_augment(batch_imgs)

        #  cond
        cond = self.encode_ema(batch_imgs)
        cond_aug = self.encode_ema(aug_imgs)
        # spatial cond aka noise
        noise = torch.randn_like(batch_imgs)
        x_T = self.encode_stochastic_ema(
            batch_imgs,
            cond=cond["cond"],
            imgs=batch_imgs if self.conf.in_channels == 2 else None,
        )
        x_T_aug = self.encode_stochastic_ema(
            x=aug_imgs,
            cond=cond_aug["cond"],
            imgs=batch_imgs if self.conf.in_channels == 2 else None,
        )

        # render images
        pred_imgs_stochastic_enc = self.render(
            x_T,
            cond=cond,
            imgs=batch_imgs if self.conf.in_channels == 2 else None,
        )
        pred_imgs_noise = self.render(
            noise,
            cond=cond,
            imgs=batch_imgs if self.conf.in_channels == 2 else None,
        )
        pred_imgs_stochastic_enc_aug = self.render(
            x_T_aug,
            cond=cond_aug,
            imgs=batch_imgs if self.conf.in_channels == 2 else None,
        )

        img_qual_metrics = {}

        # this evaluation has the following purposes
        # stochastic encoded: check reconstruction quality of regular T1w images
        # noise: check quality of "unconditional" generation
        # stochastic encoded aug: check reconstruction quality of out of distribution data
        pred_img_names_list = ["stochastic_encoded", "noise", "stochastic_encoded_aug"]
        pred_imgs_list = [
            pred_imgs_stochastic_enc,
            pred_imgs_noise,
            pred_imgs_stochastic_enc_aug,
        ]
        imgs_list = [batch_imgs, batch_imgs, aug_imgs]

        for pred_imgs, pred_imgs_name, imgs in zip(
            pred_imgs_list, pred_img_names_list, imgs_list
        ):
            cur_img_qual_metrics = self._compute_image_quality_metrics(
                pred_imgs, pred_imgs_name, imgs
            )

            img_qual_metrics.update(cur_img_qual_metrics)

        self.log_dict(img_qual_metrics, sync_dist=True, batch_size=batch_imgs.size(0))

        return shared_out | img_qual_metrics

    def _compute_image_quality_metrics(
        self,
        pred_imgs: torch.Tensor,
        pred_imgs_name: str,
        imgs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cur_img_qual_metrics = {}

        if imgs.size(1) == 1:  # if grayscale, make rgb
            imgs = imgs.repeat(1, 3, 1, 1)
            pred_imgs = pred_imgs.repeat(1, 3, 1, 1)
        imgs_01 = (imgs + 1) / 2

        # resize image and pred_img to 224 for lpips
        imgs_224 = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        pred_imgs_224 = F.interpolate(
            pred_imgs, size=224, mode="bilinear", align_corners=False
        )
        pred_imgs_01 = (pred_imgs + 1) / 2

        lpips = self.lpips_fn(imgs_224, pred_imgs_224)
        ssim = self.ssim_fn(imgs, pred_imgs)
        ms_ssim = self.ms_ssim_fn(imgs, pred_imgs)
        mse = self.mse_fn(pred_imgs, imgs)
        psnr = self.psnr_fn(pred_imgs, imgs)

        fid = self.fid_fn(pred_imgs_01, imgs_01)

        cur_img_qual_metrics[f"img/val_lpips_{pred_imgs_name}"] = lpips
        cur_img_qual_metrics[f"img/val_ssim_{pred_imgs_name}"] = ssim
        cur_img_qual_metrics[f"img/val_ms_ssim_{pred_imgs_name}"] = ms_ssim
        cur_img_qual_metrics[f"img/val_mse_{pred_imgs_name}"] = mse
        cur_img_qual_metrics[f"img/val_fid_{pred_imgs_name}"] = fid
        cur_img_qual_metrics[f"img/val_psnr_{pred_imgs_name}"] = psnr
        return cur_img_qual_metrics

    def fid_fn(self, preds: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        self._fid_fn.update(real, real=True)
        self._fid_fn.update(preds, real=False)
        return self._fid_fn.compute()

    def log_cond_weights(self):
        # check multi gpu
        if self.trainer.accelerator.auto_device_count() > 1:
            conditoned_modules = self.trainer.model.module.model.all_conditioned_modules
        else:
            conditoned_modules = self.trainer.model.model.all_conditioned_modules
        cond_weight_dict = {}
        for name, module in conditoned_modules.items():
            if not hasattr(module, "cond_emb_layers"):
                continue
            weight_per_cond_entry = (
                module.cond_emb_layers[1].weight.norm(dim=1).detach()
            )

            cond_weight_dict[f"cond_weight/{name}"] = weight_per_cond_entry.mean()
            cond_weight_dict[f"cond_weight/{name}_std"] = weight_per_cond_entry.std()
        # average the weights over all prefixes
        for prefix in ["input", "middle", "output"]:
            weights = [
                v
                for k, v in cond_weight_dict.items()
                if k.startswith(f"cond_weight/{prefix}")
            ]
            if weights:
                cond_weight_dict[f"cond_weight/{prefix}"] = torch.stack(weights).mean()
                cond_weight_dict[f"cond_weight/{prefix}_std"] = torch.stack(
                    weights
                ).std()

        self.log_dict(cond_weight_dict, sync_dist=False)

    def shared_step(
        self, batch: dict[str, torch.Tensor], batch_idx, step_name: str
    ) -> dict:
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """

        imgs = batch["img"]

        x_0 = imgs

        """
        main training mode!!!
        """

        imgs_ref = None
        batch_cond = None

        # create unqiue condition for each scanner based off of its name

        batch_cond = np.unique(batch["scanner"], return_inverse=True)[1]
        batch_cond = torch.tensor(batch_cond, device=x_0.device)

        n_views = self.model.gin_n_views

        # augment for each scanner separately
        x_0_aug = x_0.repeat(n_views, *[1] * len(x_0.shape[1:]))
        for cond in batch_cond.unique():
            cond_mask = batch_cond == cond
            anat_label_map = batch.get("anat_label_map", None)
            if anat_label_map is not None:
                anat_label_map = anat_label_map[cond_mask]
            x_0_aug_cond = self.model.intensity_augment(x_0[cond_mask], anat_label_map)

            cond_mask = cond_mask.repeat(n_views)
            x_0_aug[cond_mask] = x_0_aug_cond

        x_0 = x_0_aug
        imgs_ref = imgs if self.conf.in_channels == 2 else None
        # augment reference images

        if self.conf.net_augment_refs:
            imgs_ref_aug = imgs_ref.repeat(n_views, *[1] * len(imgs_ref.shape[1:]))
            for cond in batch_cond.unique():
                cond_mask = batch_cond == cond
                anat_label_map = batch.get("anat_label_map", None)
                if anat_label_map is not None:
                    anat_label_map = anat_label_map[cond_mask]
                imgs_ref_aug_cond = self.model.intensity_augment(
                    imgs_ref[cond_mask], anat_label_map
                )

                cond_mask = cond_mask.repeat(n_views)
                imgs_ref_aug[cond_mask] = imgs_ref_aug_cond

            imgs_ref = imgs_ref_aug

        # with numpy seed we have the problem that the sample t's are related!
        t, weight = self.T_sampler.sample(len(x_0))
        terms = self.sampler.training_losses(
            model=self.model,
            x_0=x_0,
            t=t,
            batch_cond=batch_cond,
            imgs=imgs_ref,
        )
        # pop items from loss dictionary if they end with "_var"
        variances: dict[str, torch.Tensor] = {
            k: terms.pop(k) for k in list(terms) if k.endswith("_var")
        }
        for var_name, var in variances.items():
            self.log(
                f"var/{step_name}_{var_name}",
                var.detach(),
                batch_size=batch["img"].size(0),
                sync_dist=step_name == "val",
            )

        pred_x_0 = terms.pop("pred_xstart")
        x_t = terms.pop("x_t")
        if (
            self.conf.vis_every_steps > 0  # enable vis
            and (
                (self.global_step) % (self.conf.vis_every_steps * 10) == 0
            )  # frequent vis
            and self.conf.dims == 2  # only for 2D images
        ):
            self.log_generated(step_name, x_0, pred_x_0, x_t)

        losses = {k: terms.pop(k) for k in list(terms) if "loss" in k}

        # assert not terms, f"unhandled terms: {terms.keys()}"

        for loss_key in losses:
            if loss_key != "loss":
                losses[loss_key] = losses[loss_key].mean()

        for loss_key, loss_val in losses.items():
            self.log(
                f"loss/{step_name}_{loss_key}",
                loss_val.detach(),
                batch_size=batch["img"].size(0),
                sync_dist=step_name == "val",
            )

        return {"loss": losses["loss"]}

    def log_generated(
        self,
        step_name: str,
        x_0: torch.Tensor,
        pred_x_0: torch.Tensor,
        x_t: torch.Tensor,
    ):
        all_imgs = torch.stack([x_0], dim=1).flatten(end_dim=1)

        all_imgs_vis = all_imgs
        img_grid = make_grid(
            all_imgs_vis,
            normalize=True,
            value_range=(-1, 1),
            nrow=4,
        )
        wandb_image = wandb.Image(img_grid.permute(1, 2, 0).detach().cpu().numpy())
        self.logger.log_image(
            f"generated/{step_name}_comparison",
            images=[wandb_image],
            step=self.global_step,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:

        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()

            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(
                    self.model.latent_net,
                    self.ema_model.latent_net,
                    self.conf.ema_decay,
                )
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

        imgs = batch["img"]
        # need tofirst accum step because on last accum step, the global step is already incremented
        if self.is_first_accum(batch_idx):
            if self.conf.vis_every_steps > 0 and (
                (self.global_step) % self.conf.vis_every_steps == 0
            ):
                if self.global_rank == 0:
                    self.log_cond_weights()

                self.trainer.strategy.barrier("log_cond_weights")
                self.log_latents()

                self.log_samples(x0=imgs, step_name="train")

    def on_validation_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # only render first sample of validation batch
        if batch_idx == 0 and self.conf.vis_every_steps > 0:
            imgs = batch["img"]
            self.log_samples(x0=imgs, step_name="val")
            aug_imgs = self.model.intensity_augment(imgs)
            self.log_samples(x0=aug_imgs, step_name="val")

    def log_latents(self):
        self.logger: WandbLogger

        # infer all latents
        full_dataset = torch.utils.data.ConcatDataset(
            [self.train_data, self.val_data, self.test_data]
        )

        full_loader = DataLoader(
            full_dataset,
            batch_size=self.conf.batch_size,
        )
        z_ids = []
        z_sems = []
        scanners = []
        for i_batch, batch in tqdm(enumerate(full_loader), total=len(full_loader)):
            # vis_3d(batch["img"][0], "test_img_fg.png")
            with torch.no_grad():
                batch = self.on_before_batch_transfer(batch, i_batch)
                cond = self.encode_ema(batch["img"].to(self.device))["cond"]
                z_sem, z_id = self.split_sem_id(cond)
                z_sems.append(z_sem)
                z_ids.append(z_id)
                scanners.extend(batch["scanner"])

        scanners = np.array(scanners)
        z_ids = torch.cat(z_ids).detach().cpu()
        z_sems = torch.cat(z_sems).detach().cpu()
        # map conds to unique condition_names
        condition_names_set: np.ndarray = np.unique(scanners)
        condition_names_set.sort()
        # map condition names to integers
        conds = torch.from_numpy(
            np.array([np.where(condition_names_set == c)[0][0] for c in scanners])
        )

        split_list = (
            ["train"] * len(self.train_data)
            + ["val"] * len(self.val_data)
            + ["test"] * len(self.test_data)
        )
        split_list = np.array(split_list)
        if z_ids.size(1):
            z_id_proj, _ = project_latents(
                z_ids.detach(),
                fit_latents=z_ids.detach()[split_list == "train"],
            )

            latents_fig_z_id = plot_latents(
                z_id_proj,
                conds.numpy(),
                scanners,
                cond_type="cat",
                split_list=split_list,
                projection_fp=None,
            )
            if latents_fig_z_id is not None:
                plt_img = plt_to_np(latents_fig_z_id)

                self.logger.log_image(
                    "latents/z_id", [wandb.Image(plt_img)], step=self.global_step
                )
        if z_sems.size(1):
            z_sem_proj, _ = project_latents(
                z_sems.detach(),
                fit_latents=z_sems.detach()[split_list == "train"],
            )

            latents_fig_z_sem = plot_latents(
                z_sem_proj,
                conds.numpy(),
                scanners,
                cond_type="cat",
                split_list=split_list,
                projection_fp=None,
            )
            if latents_fig_z_sem is not None:
                plt_img = plt_to_np(latents_fig_z_sem)
                self.logger.log_image(
                    "latents/z_sem", [wandb.Image(plt_img)], step=self.global_step
                )

    def split_sem_id(self, cond: torch.Tensor) -> tuple[ZSem, ZId]:
        z_sem, z_id = torch.split(
            cond,
            [self.conf.net_enc_z_sem_dim, self.conf.net_enc_z_id_dim],
            dim=1,
        )

        return z_sem, z_id

    def combine_sem_id(self, z_sem: ZSem, z_id: ZId) -> torch.Tensor:
        return torch.cat([z_sem, z_id], dim=1)

    def log_samples(self, x0, step_name: Literal["train", "val"]):
        """
        put images to the tensorboard
        """

        model = self.ema_model
        postfix = "_ema"
        models = [self.ema_model, self.model]
        model_postfix = ["_ema", ""]

        for i_log, (model, postfix) in enumerate(zip(models, model_postfix)):
            self.log_sample_with_model(x0, step_name, model, postfix)

    def log_sample_with_model(
        self,
        x0: torch.Tensor,
        step_name: Literal["train", "val"],
        model: torch.nn.Module,
        postfix: str,
    ):

        imgs = x0 if self.conf.in_channels == 2 else None
        with torch.no_grad():
            # render with stochastic encoder
            # only return embeddings for the encoder for logging
            cond = self._encode(x0, model=model)["cond"]
            x_T = self._encode_stochastic(
                x0,
                model,
                cond=cond,
                imgs=imgs,
            )

            # reconstruct the images (compared to generation from random x_T below.)
            rec = self.eval_sampler.sample(model=model, noise=x_T, cond=cond, imgs=imgs)
            # rec = self.reconstruct(x0)

            if self.noise is None:
                # only sampled during the first evaluation
                self.noise = torch.randn_like(x0)

            noise = self.noise
            if self.noise.size(0) != x0.size(0):
                # i.e. when using augmented images
                # repeat noise
                noise = self.noise.repeat(
                    x0.size(0) // self.noise.size(0), *(1,) * (self.noise.dim() - 1)
                )

            gen = self.eval_sampler.sample(
                model=model, noise=noise, cond=cond, x0=x0, imgs=imgs
            )

            gen = self.all_gather(gen).view(-1, *gen.shape[1:])

            grid_and_log_img(
                gen,
                self.logger,
                f"sample{postfix}/synthetic_{step_name}",
                step=self.global_step,
            )

            rec = self.all_gather(rec).view(-1, *rec.shape[1:])
            grid_and_log_img(
                rec,
                self.logger,
                f"sample{postfix}/reconstruction_{step_name}",
                step=self.global_step,
            )

            x0 = self.all_gather(x0).view(-1, *x0.shape[1:])
            grid_and_log_img(
                x0,
                self.logger,
                f"sample{postfix}/real_{step_name}",
                step=self.global_step,
            )

    def configure_optimizers(self):
        # scale learning rate by number of GPUs
        lr = self.conf.lr * self.trainer.world_size

        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.conf.weight_decay,
            )
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.conf.weight_decay,
            )
        else:
            raise NotImplementedError()
        out = {"optimizer": optim}
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(
                optim, lr_lambda=WarmupLR(self.conf.warmup)
            )
            out["lr_scheduler"] = {
                "scheduler": sched,
                "interval": "step",
            }
        return out


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def get_ckpt_dir(logdir: Path, wandb_id: str) -> Path:
    return logdir.parent.parent / f"checkpoints-{wandb_id}"


def train(conf: TrainConfig, fast_dev_run: bool = False):
    if fast_dev_run:
        conf.batch_size = 2
        conf.net_gin_n_views = 2
        if conf.slices_around_middle != 0:
            conf.slices_around_middle = 2
        print(
            f"set batch size to {conf.batch_size} ,{conf.net_gin_n_views} views,",
            f"{conf.slices_around_middle} slices around middle for debugging",
        )

    # enable continue training after making a change to the lightning module
    LitModel.strict_loading = False

    model = LitModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    if not fast_dev_run:
        resume = conf.wandb_id is not None
        conf.wandb_id = conf.wandb_id or wandb.util.generate_id()
        wandb_logger = WandbLogger(
            project="cacd",
            group=conf.name,
            save_dir=conf.logdir,
            log_model=False,
            id=conf.wandb_id,  # for resuming, is None if not resuming
            resume="must" if resume else None,
        )

        cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        conf.logdir = (
            Path(conf.logdir) / "wandb" / f"run-{cur_time}-{conf.wandb_id}" / "files"
        )

    else:
        wandb_logger = None
        if not conf.wandb_id:
            conf.wandb_id = "debug"
            conf.logdir = Path(conf.logdir) / "debug"
        else:
            cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            conf.logdir = (
                Path(conf.logdir)
                / "wandb"
                / f"run-{cur_time}-{conf.wandb_id}"
                / "files"
            )
            conf.logdir.mkdir(parents=True, exist_ok=True)

    print(f"Saving all logged files to {conf.logdir}")
    # create separate dir for checkpoints because wandb always creates a new directory

    ckpt_dir = get_ckpt_dir(conf.logdir, conf.wandb_id)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving all checkpoints to {ckpt_dir}")

    def create_checkpoint(
        monitor: Optional[str], save_last: Optional[bool] = None, **kwargs
    ):
        # define defaults
        save_top_k = 1

        checkpoint = MetricsModelCheckpoint(
            monitor=monitor,
            # insert metric name into filename
            filename="{epoch}-{step}-" + f"{{{monitor}}}",
            dirpath=ckpt_dir,
            save_top_k=save_top_k,
            # is based self.global_step instead of batches (val frequency)
            every_n_train_steps=conf.save_every_steps if not save_last else None,
            # save last approximately as often as the other metrics, but at the end of an epoch
            every_n_epochs=1 if save_last else None,
            auto_insert_metric_name=True,
            verbose=True,
            save_last=save_last,
            **kwargs,
        )
        print(f"Monitoring checkpoints on metric: {monitor}")
        return checkpoint

    monitoring_metrics = [
        # overall loss
        # {"monitor": "loss/val_loss", "mode": "min"},
        # # reconstruction quality in terms of MSE (correlates well with LPIPS)
        # {"monitor": "img/val_mse_stochastic_encoded_aug", "mode": "min"},
        # # # noise loss
        # {"monitor": "loss/val_mse", "mode": "min"},
        # save last checkpoint every epoch (for resuming)
        {"monitor": None, "save_last": True},
    ]

    checkpoint_callbacks = [
        create_checkpoint(**metric_kwargs) for metric_kwargs in monitoring_metrics
    ]

    last_ckpt_fps = list(ckpt_dir.glob("last*.ckpt"))
    # sort by v\d+ to get the latest version, last one should be last.ckpt without version
    last_ckpt_fps = sorted(
        last_ckpt_fps,
        key=lambda x: int(
            re.search(r"v(\d+)", x.name).group(1) if re.search(r"v(\d+)", x.name) else 0
        ),
    )
    last_ckpt_fp = None
    if last_ckpt_fps:
        last_ckpt_fp = last_ckpt_fps[-1]
        print(f"loading from {last_ckpt_fp}")

    precision = (
        "16-mixed"  # "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        if conf.fp16
        else 32
    )
    print(f"Using precision: {precision}")

    trainer = L.Trainer(
        profiler=(
            # PyTorchProfiler("pytorch_profiles")
            "simple" if conf.name == "profile" else None
        ),
        # use cpu if on macos else "auto"
        accelerator=(
            "cpu"
            if torch.backends.mps.is_available() or conf.device == "cpu"
            else "auto"
        ),
        max_steps=conf.total_steps,
        val_check_interval=conf.eval_every_batch,
        #  seems to be required to have val_check_interval > len(train_loader)
        check_val_every_n_epoch=None,
        precision=precision,
        callbacks=[
            *checkpoint_callbacks,
            LearningRateMonitor(),
            *([RichProgressBar(leave=True)] if sys.stdout.isatty() else []),
            RichModelSummary(2),
        ],
        gradient_clip_val=conf.grad_clip,
        logger=wandb_logger,
        accumulate_grad_batches=conf.accum_batches,
        # log every n updates steps, hence we need to multiply by accum_batches
        log_every_n_steps=25 if not fast_dev_run else 1,
        overfit_batches=0 if not conf.overfit else 1,
        fast_dev_run=4 if fast_dev_run else False,
        limit_val_batches=2,
    )

    trainer.fit(model, ckpt_path=last_ckpt_fp)

    wandb.finish()
