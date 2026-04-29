import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import wandb
from monai.data.dataset import CacheDataset
from torch.utils.data import DataLoader

from diffae.choices import Activation, ModelName, ModelType, OptimizerType, TrainMode
from diffae.config_base import BaseConfig
from diffae.data.dataset_cached import IXIDatasetCached
from diffae.data.glioma import PublicGliomaDataset, PublicGliomaTranslateDataset
from diffae.data.ixi import IXIDataset
from diffae.data.ms import MSPublicDataset
from diffae.data.oasis3 import OASIS3Dataset
from diffae.data.wmh import WMHDataset
from diffae.diffusion.base import (
    GenerativeType,
    LossType,
    ModelMeanType,
    ModelVarType,
    ScheduleName,
    get_named_beta_schedule,
)
from diffae.diffusion.diffusion import SpacedDiffusionBeatGansConfig, space_timesteps
from diffae.diffusion.resample import UniformSampler
from diffae.model import ModelConfig
from diffae.model.augmentations.geometric_augment import GeometricAugment
from diffae.model.blocks.timestep_blocks import ScaleAt
from diffae.model.diffae import AugName, DiffAEConfig
from diffae.model.diffae_id_preserve import DiffAEIDConfig
from diffae.model.latentnet import LatentNetType, MLPSkipNetConfig
from diffae.model.unet import DiffConfig
from diffae.uniform_sampler import UniformStratifiedSampler

_datasets_dir = os.environ.get("DATASETS_DIR", os.path.expanduser("~/datasets"))

data_paths = {
    "glioma_public": os.path.join(_datasets_dir, "glioma_public"),
    "glioma_public_translate": os.path.join(_datasets_dir, "glioma_public"),
    "wmh": os.path.join(_datasets_dir, "wmh"),
    "ms_public": os.path.join(_datasets_dir, "ms_public"),
    "ixi": os.path.join(_datasets_dir, "ixi_reg_skullstrip"),
    "ixi_cached": os.path.join(_datasets_dir, "ixi_reg_skullstrip"),
    "oasis3": os.path.join(_datasets_dir, "OASIS/OASIS3/MR_Scans_reg_skullstrip"),
}


@dataclass
class TrainConfig(BaseConfig):
    # random seed
    seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False

    # losses for double gin
    scanner_loss_weight: float = 0.5
    content_loss_weight: float = 0.5
    ortho_loss_weight: float = 0.0
    cross_correlation_loss_weight: float = 0.0
    info_nce_loss_agg_fn: Literal["sum", "mean"] = "sum"

    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False

    beta_scheduler: ScheduleName = "linear"
    data_names: tuple[str, ...] = ("",)
    fit_sites: dict = field(default_factory=dict)
    test_sites: dict = field(default_factory=dict)

    data_split_mode: Literal["random", "stratified", "loaded"] = "random"
    data_biasfield_corrected: bool = True

    dropout: float = 0.1
    ema_decay: float = 0.9999

    # based on trainer.global_step
    metrics_every_steps: int = 200_000
    # based on batch count
    eval_every_steps: int = 200_000
    # based on batch count, is used in trainer. initialized in make_model_conf
    eval_every_batch: Optional[int] = None
    metrics_ema_every_steps: int = 200_000

    fp16: bool = False
    grad_clip: float = 1
    img_size: tuple[int, ...] = (64, 64, 64)
    slices_around_middle: int = 0  # default: only middle slice
    norm_range: tuple[float, float] = (-1.0, 1.0)

    # can be set to override the mri sequnces defined in each dataset
    mri_sequences: Optional[tuple[str, ...]] = None
    target_prop: Literal["scanner", "age", "sex"] = "scanner"
    dims: Literal[2, 3] = 2  # whether to use 2D or 3D diffusion
    in_channels: int = 3  # number of input channels
    model_out_channels: int = 3  # number of output channels
    lr: float = 0.0001
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    net_attn_resolutions: tuple[int] = (16,)
    net_use_attn: bool = False
    net_beatgans_attn_head: int = 1

    # embedding dimensions for the time steps
    net_t_embed_channels: int = 512
    # embedding dimensions for the latent space (condition)
    net_cond_channels: int = 512
    net_enc_separate_encoders: bool = False
    net_enc_z_sem_dim: int = 256
    net_enc_z_id_dim: int = 256

    net_enc_use_non_linear_head: bool = False
    net_resblock_updown: bool = True

    net_enc_pool: str = "adaptivenonzero"
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_ch_mult: Tuple[int, ...] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int, ...] = None
    net_enc_k: int = None

    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Sequence[int] = None
    net_enc_grad_checkpoint: bool = False
    net_enc_learnable_downsampling: bool = False
    net_autoenc_stochastic: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_channel_mult: Tuple[int] = (1, 2, 4)
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type: LatentNetType = LatentNetType.none
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_gin_alpha_range: tuple[float, float] = (0.0, 1.0)

    net_augment_refs: bool = False
    net_gin_n_views: int = 9
    net_rc_gin_n_layers: int = 2
    net_rc_gin_n_hidden_dims: int = 2
    net_rc_rotationally_symmetric: bool = False
    net_double_gin_scanner_contrasting: Literal["all", "same", "same_patient_negative"] = "same"
    net_rc_type: Literal["linear", "gin"] = "gin"
    net_rc_normalization: Literal["fro", "minmax"] = "minmax"
    net_rc_do_updownsampling: bool = False
    net_rc_resize_mode: str = "bilinear"
    net_rc_target_size: int = 2048
    net_intensity_augs_names: tuple[AugName, ...] = (
        "rc",
        "bf",
        "gamma",
    )
    net_rand_resize_aug: bool = False
    net_use_geometric_augs: bool = False
    net_augmentor_type: Literal["gin", "classic", "synth"] = "gin"
    net_apply_aug_to_fg_only: bool = True
    net_enc_num_cls: int = None
    num_workers: int = 2
    parallel: bool = False
    postfix: str = ""
    sample_size: int = 64

    vis_every_steps: int = 20_000
    # based on global step
    save_every_steps: int = 100_000

    T_eval: int = 1_000
    T_sampler: str = "uniform"
    T: int = 1_000
    total_steps: int = 10_000_000
    warmup: int = 0

    _logdir: Optional[Path] = None
    base_dir: Path = Path("checkpoints")

    # to be overridden
    name: str = ""
    sample_mode: Optional[Literal["stratified"]] = None
    overfit: bool = False
    wandb_id: str = None

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def data_paths(self) -> tuple[Path, ...]:
        # may use the cache dir
        return tuple(Path(data_paths[data_name]) for data_name in self.data_names)

    @property
    def logdir(self):
        if self._logdir is not None:
            return self._logdir
        return self.base_dir / self.name

    @logdir.setter
    def logdir(self, value: Path):
        self._logdir = value

    def _make_diffusion_conf(self, T: int):
        # can use T < self.T for evaluation
        # follows the guided-diffusion repo conventions
        # t's are evenly spaced
        if self.beatgans_gen_type == GenerativeType.ddpm:
            section_counts = [T]
        elif self.beatgans_gen_type == GenerativeType.ddim:
            section_counts = f"ddim{T}"
        else:
            raise NotImplementedError()

        return SpacedDiffusionBeatGansConfig(
            gen_type=self.beatgans_gen_type,
            model_type=self.model_type,
            betas=get_named_beta_schedule(self.beta_scheduler, self.T),
            model_mean_type=self.beatgans_model_mean_type,
            model_var_type=self.beatgans_model_var_type,
            loss_type=self.beatgans_loss_type,
            scanner_loss_weight=self.scanner_loss_weight,
            content_loss_weight=self.content_loss_weight,
            ortho_loss_weight=self.ortho_loss_weight,
            cross_correlation_loss_weight=self.cross_correlation_loss_weight,
            double_gin_scanner_contrasting=self.net_double_gin_scanner_contrasting,
            z_sem_dim=self.net_enc_z_sem_dim,
            z_id_dim=self.net_enc_z_id_dim,
            do_rescale_timesteps=self.beatgans_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T, section_counts=section_counts),
            fp16=self.fp16,
            train_pred_xstart_detach=self.train_pred_xstart_detach,
            info_nce_loss_agg_fn=self.info_nce_loss_agg_fn,
        )

    def _make_latent_diffusion_conf(self, T=None):
        raise NotImplementedError("latent diffusion not implemented")

    def make_T_sampler(self):
        if self.T_sampler == "uniform":
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_latent_diffusion_conf(self):
        return self._make_latent_diffusion_conf(T=self.T)

    def make_latent_eval_diffusion_conf(self):
        # latent can have different eval T
        raise NotImplementedError("latent diffusion not implemented")
        # return self._make_latent_diffusion_conf(T=self.latent_T_eval)

    def make_datasets(
        self, paths: tuple[Path, ...], **shared_kwargs
    ) -> tuple[PublicGliomaDataset, ...]:
        shared_kwargs["img_size"] = self.img_size
        shared_kwargs["norm_range"] = self.norm_range
        shared_kwargs["mri_sequences"] = self.mri_sequences

        dataset_dict = {
            "glioma_public": lambda path: PublicGliomaDataset(data_dir=path, **shared_kwargs),
            "glioma_public_64": lambda path: PublicGliomaDataset(data_dir=path, **shared_kwargs),
            "glioma_public_translate": lambda path: PublicGliomaTranslateDataset(
                data_dir=path, **shared_kwargs
            ),
            "wmh": lambda path: WMHDataset(data_dir=path, **shared_kwargs),
            "ms_public": lambda path: MSPublicDataset(data_dir=path, **shared_kwargs),
            "ixi_cached": lambda path: IXIDatasetCached(
                root_dir=path,
                stage=shared_kwargs["split"],
                img_size=self.img_size,
                offset=self.slices_around_middle,
            ),
            "ixi": lambda path: IXIDataset(
                spatial_dims=self.dims,
                data_dir=path,
                fit_sites=self.fit_sites["ixi"],
                test_sites=self.test_sites["ixi"],
                target_prop=self.target_prop,
                augmentations=(
                    GeometricAugment(self.img_size, self.dims, dict_mode=True, keys=["img", "seg"])
                    if self.net_use_geometric_augs  # and kwargs["split"] == "train"
                    else None
                ),
                biasfield_corrected=self.data_biasfield_corrected,
                load_anat_label_maps="anat_reg" in self.net_intensity_augs_names,
                slices_around_middle=self.slices_around_middle,
                **shared_kwargs,
            ),
            "oasis3": lambda path: OASIS3Dataset(
                spatial_dims=self.dims,
                data_dir=path,
                fit_sites=self.fit_sites["oasis3"],
                test_sites=self.test_sites["oasis3"],
                target_prop=self.target_prop,
                augmentations=(
                    GeometricAugment(self.img_size, self.dims, dict_mode=True, keys=["img", "seg"])
                    if self.net_use_geometric_augs  # and kwargs["split"] == "train"
                    else None
                ),
                biasfield_corrected=self.data_biasfield_corrected,
                **shared_kwargs,
            ),
        }

        return tuple(
            dataset_dict[data_name](path) for data_name, path in zip(self.data_names, paths)
        )

    def make_loader(
        self,
        dataset,
        shuffle: bool,
        num_worker: bool = None,
        drop_last: bool = True,
        batch_size: int = None,
    ):
        sampler = None
        if self.sample_mode == "stratified" and shuffle:
            all_labels = dataset.datasets[0].dataset.targets
            cur_labels = all_labels[dataset.datasets[0].indices]

            cur_labels_int = torch.from_numpy(
                cur_labels[:, None] == np.unique(all_labels)[None]
            ).nonzero()[:, 1]

            sampler = UniformStratifiedSampler(
                batch_size=batch_size or self.batch_size,
                labels=cur_labels_int,
            )
        uses_cache = False  # ( any(isinstance(d, CacheDataset) for d in dataset.datasets))

        return DataLoader(
            dataset,
            batch_size=min(batch_size or self.batch_size, len(dataset)),
            sampler=sampler,
            # with sampler, use the sample instead of this option
            shuffle=False if sampler else shuffle,
            num_workers=((num_worker or self.num_workers) if not uses_cache else 0),
            pin_memory=not uses_cache,
            drop_last=drop_last,
            persistent_workers=(num_worker or self.num_workers) > 0 and not uses_cache,
            # multiprocessing_context=get_context('fork'),
        )

    def make_model_conf(self):
        conf_dict = self.update_config_from_json()

        if isinstance(self.img_size, Sequence):
            if self.img_size[0] == 64:
                if "batch_size" not in conf_dict:
                    self.batch_size = 13
                if "accum_batches" not in conf_dict:
                    self.accum_batches = 1
                if "net_gin_n_views" not in conf_dict:
                    self.net_gin_n_views = 13
                if "num_workers" not in conf_dict:
                    self.num_workers = 10
            else:
                if "batch_size" not in conf_dict:
                    self.batch_size = 16
                if "net_gin_n_views" not in conf_dict:
                    self.net_gin_n_views = 16
                if "accum_batches" not in conf_dict:
                    self.accum_batches = 1
                if "num_workers" not in conf_dict:
                    self.num_workers = 15

        self.img_size = self.img_size[: self.dims]
        # scale  eval_every_steps by accum batches to match global_step counter
        self.eval_every_batch = (
            self.eval_every_steps * self.accum_batches
            if self.eval_every_steps is not None
            else None
        )
        print(f"{self.eval_every_steps=} * {self.accum_batches=} = {self.eval_every_batch=}")

        if self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = DiffConfig(
                attention_resolutions=self.net_attn_resolutions,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=self.dims,
                dropout=self.dropout,
                t_embed_channels=self.net_t_embed_channels,
                image_size=self.img_size,
                in_channels=self.in_channels,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
                cond_channels=self.net_cond_channels,
            )
        elif self.model_name in [
            ModelName.beatgans_autoenc,
            ModelName.beatgans_autoenc_id,
        ]:
            # self.model_name = ModelName.beatgans_autoenc_id

            cls = {
                ModelName.beatgans_autoenc: DiffAEConfig,
                ModelName.beatgans_autoenc_id: DiffAEIDConfig,
            }[self.model_name]
            # supports both autoenc and vaeddpm
            if self.model_name in [
                ModelName.beatgans_autoenc,
                ModelName.beatgans_autoenc_id,
            ]:
                self.model_type = ModelType.autoencoder
            else:
                raise NotImplementedError()

            if self.net_latent_net_type == LatentNetType.none:
                latent_net_conf = None
            elif self.net_latent_net_type == LatentNetType.skip:
                latent_net_conf = MLPSkipNetConfig(
                    num_channels=self.net_cond_channels,
                    skip_layers=self.net_latent_skip_layers,
                    num_hid_channels=self.net_latent_num_hid_channels,
                    num_layers=self.net_latent_layers,
                    num_time_emb_channels=self.net_latent_time_emb_channels,
                    activation=self.net_latent_activation,
                    use_norm=self.net_latent_use_norm,
                    condition_bias=self.net_latent_condition_bias,
                    dropout=self.net_latent_dropout,
                    last_act=self.net_latent_net_last_act,
                    num_time_layers=self.net_latent_num_time_layers,
                    time_last_act=self.net_latent_time_last_act,
                )
            else:
                raise NotImplementedError()

            model_conf_params = dict(
                attention_resolutions=self.net_attn_resolutions,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=self.dims,
                dropout=self.dropout,
                t_embed_channels=self.net_t_embed_channels,
                cond_channels=self.net_cond_channels,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=self.in_channels,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                use_attention=self.net_use_attn,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
                latent_net_conf=latent_net_conf,
                gin_alpha_range=self.net_gin_alpha_range,
                augmentor_type=self.net_augmentor_type,
                gin_n_views=self.net_gin_n_views,
                gin_n_layers=self.net_rc_gin_n_layers,
                gin_n_hidden_dims=self.net_rc_gin_n_hidden_dims,
                rc_type=self.net_rc_type,
                rc_normalization=self.net_rc_normalization,
                apply_aug_to_fg_only=self.net_apply_aug_to_fg_only,
                intensity_augs_names=self.net_intensity_augs_names,
                rc_rotationally_symmetric=self.net_rc_rotationally_symmetric,
                rand_resize_aug=self.net_rand_resize_aug,
                enc_learnable_downsampling=self.net_enc_learnable_downsampling,
                rc_do_updownsampling=self.net_rc_do_updownsampling,
                rc_resize_mode=self.net_rc_resize_mode,
                rc_target_size=self.net_rc_target_size,
            )
            if self.model_name == ModelName.beatgans_autoenc_id:
                model_conf_params.update(
                    enc_z_sem_dim=self.net_enc_z_sem_dim,
                    enc_z_id_dim=self.net_enc_z_id_dim,
                    enc_use_non_linear_head=self.net_enc_use_non_linear_head,
                    enc_separate_encoders=self.net_enc_separate_encoders,
                )
            self.model_conf = cls(**model_conf_params)
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf

    def update_config_from_json(self):
        conf_dict = {}
        if (
            conf_json := Path(__file__).resolve().parent.parent / "conf" / f"{self.name}.json"
        ).exists():
            with open(conf_json, "r", encoding="utf-8") as f:
                conf_dict = json.load(f)
            diff_dict = {
                k: f"{self.__dict__[k]} -> {new_v}"
                for k, new_v in conf_dict.items()
                if k in self.__dict__ and self.__dict__[k] != new_v
            }
            # ensure batch size stays the same

            assert all(
                conf_key in self.__dict__ for conf_key in conf_dict.keys()
            ), f"conf keys {list(c for c in conf_dict.keys() if c not in self.__dict__)} not in config"

            # convert types of loaded values
            for k in conf_dict.keys():
                target_type = self.__dict__[k].__class__
                conf_dict[k] = target_type(conf_dict[k])

            self.__dict__.update(conf_dict)
            diff_dict_json = json.dumps(diff_dict, indent=2, sort_keys=True)
            print(f"updated conf from {conf_json} keys {diff_dict_json} ")
        return conf_dict


def conf_from_wandb_id(
    wandb_id: Optional[str] = None,
    target_prop: Literal["scanner", "age", "sex"] = "scanner",
) -> TrainConfig:
    # local import to avoid circular imports
    from diffae.templates import templates_dict

    template_name = "scanner_harm"
    conf_name = "ixi_guys-hh"
    conf = templates_dict[template_name](conf_name)
    if wandb_id is None:
        # conf.wandb_id = "3tymxqib"
        # conf.wandb_id = "pyqa8qd5"
        # conf.wandb_id = "d09604vp"
        # conf.wandb_id = "8rfe51pi"
        conf.wandb_id = "dhf5k96o"
        conf.wandb_id = "9zswfbse"
        conf.wandb_id = "wn9wnlok"
        conf.wandb_id = "6x4qwt4z"
        conf.wandb_id = "u4kqows4"
    else:
        conf.wandb_id = wandb_id

    conf.sample_size = 4
    conf.target_prop = target_prop

    conf.T_eval = 25

    # get run name from wandb_id
    run = wandb.Api().run(f"med-image-translation/{conf.wandb_id}")
    # use group name as the run name
    conf.name = run.group if run.group else run.name

    conf.make_model_conf()

    return conf


def customize_conf_for_eval(conf: TrainConfig, with_t2: bool = False):
    if conf.fit_sites is not None and (len(conf.fit_sites) == 1):
        conf.fit_sites = ["Guys", "HH"]
        conf.test_sites = ["IOP"]

    conf.net_use_geometric_augs = False
    if with_t2:
        conf.mri_sequences = ("T1", "T2")
