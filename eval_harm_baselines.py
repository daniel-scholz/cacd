import json
import logging
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, get_args

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics.image
import wandb
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon
from torchvision.utils import save_image
from tqdm import tqdm

from diffae.config import conf_from_wandb_id
from diffae.data.on_harmony import OnHarmonyDataset, ONHScanner
from diffae.eval.model import load_harm_model
from diffae.eval.paired import load_images_subjects_scanners_sessions
from diffae.experiment import LitModel
from diffae.metrics.masked_psnr import MaskedPSNR
from harm_model import (
    DiffAEHarmonizationModel,
    HACA3HarmonizationModel,
    HarmonizationMethodName,
    HistogramMatchingModel,
    UnharmonizeModel,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
# add prefix to log messages
logger.handlers[0].setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)5s - %(message)s")
)


def diffae_setup(wandb_id: str, device: torch.device) -> tuple[LitModel, int]:

    conf = conf_from_wandb_id(wandb_id)
    # just to shorten data loading times, has no actual effect on the evalu
    conf.data_names = ["ixi"]

    conf.T_eval = 250
    model, _, ckpt_name = load_harm_model(LitModel, conf, device=device, metric="None")
    global_step = int(re.search(r"step=(\d+)", ckpt_name).group(1))

    return model, global_step


@click.command()
@click.option(
    "--method",
    type=click.Choice(get_args(HarmonizationMethodName)),
    help="Set harmonization model name",
)
@click.option(
    "--target_scanner",
    type=click.Choice(get_args(ONHScanner)),
    help="Set target scanner",
)
@click.option("--wandb_id", type=str, default=None, help="Set wandb id for DiffAE model")
def cli(method, target_scanner, wandb_id=None):
    main(method, target_scanner, wandb_id)


def save_figure_wrapper(args):
    (
        sub,
        session,
        source_img,
        harmonized_img,
        target_img,
        img_results_dir,
        target_scanner,
    ) = args

    save_figure(
        img_results_dir,
        target_scanner,
        sub,
        session,
        source_img,
        harmonized_img,
        target_img,
    )


# group by
def calc_mean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["method", "method_specific_name", "wandb_id", "global_step"], dropna=False)
    return df.mean(numeric_only=True)


def get_image_size(method_specific_name: str | None) -> tuple[int, ...]:
    default_size = (192, 224, 192)
    if method_specific_name is None or method_specific_name == "pretrained":
        return default_size
    # load config json
    config_fp = Path("conf") / f"{method_specific_name}.json"
    with open(config_fp, "r") as f:
        config = json.load(f)
    loaded_size = config.get("img_size")
    if loaded_size is None:
        return default_size
    logger.info(f"loaded image size: {loaded_size}")
    return tuple(loaded_size)


def get_method_specific_name(
    method: HarmonizationMethodName, wandb_id: Optional[str] = None
) -> Optional[str]:
    match method:
        case "HACA3":
            return "pretrained"
        case "DiffAE":
            wandb_run = wandb.Api().run(f"med-image-translation/{wandb_id}")
            name = wandb_run.group
            if name is None:
                name = wandb_run.name
            return name
        case _:
            return None


def main(
    method: HarmonizationMethodName,
    target_scanner: ONHScanner,
    wandb_id: Optional[str] = None,
):
    # constants
    base_results_dir = Path("results") / "baseline_comp"
    device = torch.device(
        # "cpu"
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    dims = 2

    logger.info(f"{base_results_dir=}")
    logger.info(f"{target_scanner=}")
    logger.info(f"{method=}")

    if method == "DiffAE":
        dims = 2
        logger.debug("DiffAE model requires 2D images")

    # load dataset
    on_harmony_dir = Path("~/datasets/on-harmony").expanduser()

    method_specific_name = get_method_specific_name(method, wandb_id)
    if method != "DiffAE":
        wandb_id = None

    img_size = get_image_size(method_specific_name)[:dims]
    dataset = OnHarmonyDataset(
        on_harmony_dir,
        spatial_dims=dims,
        img_size=img_size,
        norm_range=(-1, 1),
        skullstrip=method != "HACA3",
        biasfield_corrected=True,
    )

    # load images, subjects, scanners
    subjects, scanners, sessions, source_imgs_torch, source_brainmasks = (
        load_images_subjects_scanners_sessions(dataset, device)
    )
    uniq_subjects = np.unique(subjects)

    # get target images
    target_images, target_imgs_matched_to_source_image = load_target_images(
        target_scanner, subjects, scanners, source_imgs_torch, uniq_subjects
    )
    target_images_bms, target_imgs_bm_matched_to_source = load_target_images(
        target_scanner, subjects, scanners, source_brainmasks, uniq_subjects
    )
    # target_debug_subject = subjects[0]
    # logger.warning(f"For debugging use only subject {target_debug_subject}")
    # target_images = imgs_torch[
    #     (scanners == target_scanner) & (subjects == target_debug_subject)
    # ]

    # define metrics
    logger.warning("psnr dimensions is 1 (channels), 2 (width), 3 (height), change for 3d images")
    psnr, psnr_fg, ssim, ms_ssim, mse_list, mse_fg_list = init_metrics(device)

    harmonization_model, global_step = load_model(method, target_images, device, wandb_id)
    df_results_fp = base_results_dir / "methods_average.csv"
    if df_results_fp.exists():
        df_results_avg_old = pd.read_csv(df_results_fp)

        # map None to np.nan for method_specific_name
        def check_column_with_possible_none(series: pd.Series, method_identifier: str | None):
            if method_identifier is None:
                return series.isnull()
            return series == method_identifier

        # check if current method already exists
        if (
            df_results_avg_old[
                (df_results_avg_old["method"] == method)
                & check_column_with_possible_none(
                    df_results_avg_old["method_specific_name"], method_specific_name
                )
                & check_column_with_possible_none(df_results_avg_old["wandb_id"], wandb_id)
                & check_column_with_possible_none(df_results_avg_old["global_step"], global_step)
            ].shape[0]
            > 0
        ):
            # raise ValueError(
            #     f"Results for method {method} {method_specific_name} {wandb_id} {global_step} already exist"
            # )
            logger.warning(
                f"Results for method {method} {method_specific_name} {wandb_id} {global_step} already exist"
            )

    img_results_dir = base_results_dir / f"{method}_{method_specific_name}_{wandb_id}"
    img_results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{img_results_dir=}")

    batch_size = 40
    # chunk subjects into batches
    subjects_loader = np.array_split(subjects, int(np.ceil(len(subjects) / batch_size)))

    source_imgs_loader = torch.chunk(
        source_imgs_torch,
        chunks=int(np.ceil(len(source_imgs_torch) / batch_size)),
        dim=0,
    )
    target_imgs_loader = torch.chunk(
        target_imgs_matched_to_source_image,
        chunks=int(np.ceil(len(target_imgs_matched_to_source_image) / batch_size)),
        dim=0,
    )
    brainmasks_loader = torch.chunk(
        source_brainmasks,
        chunks=int(np.ceil(len(source_brainmasks) / batch_size)),
        dim=0,
    )
    target_bm_loader = torch.chunk(
        target_imgs_bm_matched_to_source,
        chunks=int(np.ceil(len(target_imgs_bm_matched_to_source) / batch_size)),
        dim=0,
    )

    harmonized_imgs_out = []
    target_imgs_out = []
    source_imgs_out = []
    brainmasks_intersection_out = []

    pbar = tqdm(
        iterable=zip(source_imgs_loader, target_imgs_loader, brainmasks_loader, target_bm_loader),
        total=len(subjects_loader),
        desc=f"Harmonization {method}",
    )
    for source_img_batch, target_img_batch, brainmask_batch, bm_target_batch in pbar:

        # add batch dimension
        harmonized_img_batch = harmonization_model.harmonize(source_images=source_img_batch)

        # extract middle axial slice of 3D images
        if dims == 3:
            harmonized_img_batch = harmonized_img_batch[..., harmonized_img_batch.size(-1) // 2]
            target_img_batch = target_img_batch[..., target_img_batch.size(-1) // 2]
            source_img_batch = source_img_batch[..., source_img_batch.size(-1) // 2]
            brainmask_batch = brainmask_batch[..., brainmask_batch.size(-1) // 2]
            bm_target_batch = bm_target_batch[..., bm_target_batch.size(-1) // 2]

        assert source_img_batch.size() == harmonized_img_batch.size() == target_img_batch.size()
        # pad all images to [192, 224, 192][:dims]
        target_size = [192, 224, 192][:dims]
        padding = [t - s for t, s in zip(target_size, source_img_batch.shape[-dims:])]
        # invert to match parameters of torch.nn.functional.pad
        padding = padding[::-1]

        pad_val = source_img_batch.min()
        assert pad_val == -1

        source_img_batch = torch.nn.functional.pad(
            source_img_batch,
            [padding[i // 2] // 2 for i in range(len(padding) * 2)],
            value=pad_val,
        )
        harmonized_img_batch = torch.nn.functional.pad(
            harmonized_img_batch,
            [padding[i // 2] // 2 for i in range(len(padding) * 2)],
            value=pad_val,
        )
        target_img_batch = torch.nn.functional.pad(
            target_img_batch,
            [padding[i // 2] // 2 for i in range(len(padding) * 2)],
            value=pad_val,
        )
        brainmask_batch = torch.nn.functional.pad(
            brainmask_batch,
            [padding[i // 2] // 2 for i in range(len(padding) * 2)],
            value=0,
        )
        bm_target_batch = torch.nn.functional.pad(
            bm_target_batch,
            [padding[i // 2] // 2 for i in range(len(padding) * 2)],
            value=0,
        )

        # calculate brainmask intersection between harmonized and target

        brainmask_intersection = brainmask_batch & bm_target_batch
        background_intersection = ~brainmask_intersection
        save_image((bm_target_batch ^ brainmask_batch).float(), "test_bm.png", normalize=True)
        # mask images with background_intersection
        bg_val = -1
        harmonized_img_batch[background_intersection] = bg_val
        # mask images with brainmask_intersection
        target_img_batch[background_intersection] = bg_val
        source_img_batch[background_intersection] = bg_val

        harmonized_imgs_out.extend(harmonized_img_batch)
        target_imgs_out.extend(target_img_batch)
        source_imgs_out.extend(source_img_batch)
        brainmasks_intersection_out.extend(brainmask_intersection)

        logger.debug("for now compute metrics in 2d of the middle slice")
        logger.debug(f"Metrics image size: {source_img_batch.size()}")
        # check size is [192, 224, 1^92][:dims]
        assert list(source_img_batch.shape[-dims:]) == target_size
        assert list(harmonized_img_batch.size()[-dims:]) == target_size
        assert list(target_img_batch.size()[-dims:]) == target_size
        assert list(brainmask_batch.size()[-dims:]) == target_size

        harmonized_img_01 = (harmonized_img_batch + 1) / 2
        target_img_01 = (target_img_batch + 1) / 2

        ssim(harmonized_img_01, target_img_01)
        ms_ssim(harmonized_img_01, target_img_01)
        psnr(harmonized_img_01, target_img_01)
        psnr_fg(harmonized_img_01, target_img_01, brainmask_intersection)
        mse_img = (harmonized_img_batch - target_img_batch).square()
        mse_list.extend(mse_img.flatten(start_dim=1).mean(dim=1))
        mse_fg_list.extend([mi[bm].mean() for mi, bm in zip(mse_img, brainmask_intersection)])

    harmonized_imgs_out = torch.stack(harmonized_imgs_out)
    target_imgs_out = torch.stack(target_imgs_out)
    source_imgs_out = torch.stack(source_imgs_out)
    brainmasks_intersection_out = torch.stack(brainmasks_intersection_out)

    with Pool() as pool:
        pbar_save_images = tqdm(
            pool.imap_unordered(
                # map(
                save_figure_wrapper,
                zip(
                    subjects,
                    sessions,
                    source_imgs_out.cpu().numpy(),
                    harmonized_imgs_out.cpu().numpy(),
                    target_imgs_out.cpu().numpy(),
                    [img_results_dir] * len(subjects),
                    [target_scanner] * len(subjects),
                ),
            ),
            total=len(subjects),
            desc="Saving images",
        )
        for _ in pbar_save_images:
            pass

    psnr_vals = psnr.compute()
    psnr_fg_vals = psnr_fg.compute()
    ssim_vals, ssim_imgs = ssim.compute()
    # calculate ssim of foreground of brainmask

    ssim_fg_vals = torch.stack(
        [ssim_v[bms].mean() for ssim_v, bms in zip(ssim_imgs, brainmasks_intersection_out)]
    )
    ms_ssim_vals = ms_ssim.compute()
    mse_vals = torch.stack(mse_list)
    mse_fg_vals = torch.stack(mse_fg_list)

    # ignore infs (whent target == harmonized), only occurs with unharmonized images
    source_is_target_mask = torch.from_numpy(scanners == target_scanner).to(device)
    logger.info(f"PSNR: {psnr_vals[~source_is_target_mask].mean()}")
    logger.info(f"PSNR FG: {psnr_fg_vals[~source_is_target_mask].mean()}")
    logger.info(f"SSIM: {ssim_vals[~source_is_target_mask].mean()}")
    logger.info(f"SSIM FG: {ssim_fg_vals[~source_is_target_mask].mean()}")
    logger.info(f"MS-SSIM: {ms_ssim_vals[~source_is_target_mask].mean()}")
    logger.info(f"MSE: {mse_vals[~source_is_target_mask].mean()}")
    logger.info(f"MSE FG: {mse_fg_vals[~source_is_target_mask].mean()}")

    # save results into table
    df_results_all_cur_method = pd.DataFrame(
        {
            "subject": subjects,
            "scanner": scanners,
            "session": sessions,
            "target_scanner": target_scanner,
            "method": method,
            "method_specific_name": method_specific_name,
            "global_step": global_step,
            "wandb_id": wandb_id,
            "ssim": ssim_vals.cpu().numpy(),
            "ssim_fg": ssim_fg_vals.cpu().numpy(),
            "ms_ssim": ms_ssim_vals.cpu().numpy(),
            "psnr": psnr_vals.cpu().numpy(),
            "psnr_fg": psnr_fg_vals.cpu().numpy(),
            "mse": mse_vals.cpu().numpy(),
            "mse_fg": mse_fg_vals.cpu().numpy(),
        }
    )
    metrics = [
        "ssim_fg",
        "psnr_fg",
        "mse_fg",
        "ssim",
        "psnr",
        "mse",
        "ms_ssim",
    ]

    df_results_fp_all = base_results_dir / "all_images.csv"

    if df_results_fp.exists():
        # load and append
        df_results_all_old = pd.read_csv(df_results_fp_all)
        df_results_all_old = pd.concat([df_results_all_old, df_results_all_cur_method])
        df_results_all_images = df_results_all_old.drop_duplicates(
            subset=[
                "subject",
                "scanner",
                "session",
                "target_scanner",
                "method",
                "method_specific_name",
                "wandb_id",
            ],
            keep="last",
        )

    else:
        df_results_all_images = df_results_all_cur_method

    # calculate improvement over unharmonized

    df_unharmonized = df_results_all_images[df_results_all_images["method"] == "unharmonized"]
    # if improvement columns not in df_results_all_images, add them
    if not all(
        col in df_results_all_images.columns
        for col in [f"{metric}_improvement" for metric in metrics]
    ):
        df_results_all_images = df_results_all_images.assign(
            **{f"{metric}_improvement": np.nan for metric in metrics}
        )
    # reset index
    df_results_all_images.reset_index(drop=True, inplace=True)

    # if unharmonized method exists, calculate improvement
    if df_unharmonized.shape[0]:
        df_grouped_method = df_results_all_images.groupby(
            ["method", "method_specific_name", "wandb_id", "global_step"]
        )

        for method_id, df_method in df_grouped_method:

            df_method_unharmonized = df_method.merge(
                df_unharmonized,
                on=["subject", "session", "scanner", "target_scanner"],
                suffixes=("", "_unh"),
            )
            # calculate improvement in ssim, psnr, mse

            for metric in metrics:

                improvement = (
                    df_method_unharmonized[metric] - df_method_unharmonized[f"{metric}_unh"]
                ).values
                df_method = df_method.assign(**{f"{metric}_improvement": improvement})
                # reassing to grouped df
                group_index = df_grouped_method.groups[method_id]
                df_results_all_images.loc[group_index, f"{metric}_improvement"] = improvement

                # calculate wilcoxon signed rank test
                statistic, p_value = wilcoxon(
                    df_method_unharmonized[metric],
                    df_method_unharmonized[f"{metric}_unh"],
                    alternative="greater" if "mse" not in metric else "less",
                )

                # add column with p_value
                df_results_all_images.loc[group_index, f"{metric}_improvement_p_value"] = p_value

                logger.info(
                    f"{method_id}: Wilcoxon signed rank test for {metric} improvement: {p_value}"
                )

    # sort by subject, session, scanner, method, method_specific_name, wandb_id
    df_results_all_images = df_results_all_images.sort_values(
        by=[
            "method",
            "method_specific_name",
            "wandb_id",
            "global_step",
            "target_scanner",
            "subject",
            "session",
        ]
    )

    # average over subjects and scanners
    df_results_filtered = df_results_all_images[
        df_results_all_images["target_scanner"] != df_results_all_images["scanner"]
    ]
    df_results_avg = calc_mean_metrics(df_results_filtered)
    # sort by metrics: psnr, ssim, mse
    df_results_avg = df_results_avg.sort_values(
        by=metrics,
        ascending=["mse" in metric for metric in metrics],
    )

    df_results_all_images.to_csv(df_results_fp_all, index=False)
    df_results_avg.to_csv(df_results_fp, index=True)


def init_metrics(device):
    psnr = torchmetrics.image.PeakSignalNoiseRatio(
        data_range=(0, 1), reduction="none", dim=list(range(1, 2 + 2))
    ).to(device)
    psnr_fg = MaskedPSNR(
        dim=list(range(1, 2 + 2)),
        reduction="none",
        data_range=(0, 1),
    ).to(device)

    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
        data_range=(0, 1),
        reduction="none",
        return_full_image=True,
    ).to(device)
    ms_ssim = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(
        data_range=(0, 1), reduction="none", kernel_size=7
    ).to(device)
    mse_list = []
    mse_fg_list = []
    return psnr, psnr_fg, ssim, ms_ssim, mse_list, mse_fg_list


def load_target_images(
    target_scanner: str,
    subjects: np.ndarray,
    scanners: np.ndarray,
    imgs_torch: torch.Tensor,
    uniq_subjects: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_imgs_mask = scanners == target_scanner
    target_images = imgs_torch[target_imgs_mask]
    logger.info(f"{target_images.size()}")

    target_imgs_matched_to_source_image = torch.zeros_like(imgs_torch)
    # for each subject, insert the corresponding target image
    for uniq_sub in uniq_subjects:
        target_imgs_matched_to_source_image[subjects == uniq_sub] = imgs_torch[
            (scanners == target_scanner) & (subjects == uniq_sub)
        ]

    return target_images, target_imgs_matched_to_source_image


def load_model(
    method: HarmonizationMethodName,
    target_images: torch.Tensor,
    device: torch.device,
    wandb_id: Optional[str] = None,
) -> tuple[DiffAEHarmonizationModel, Optional[int]]:

    global_step = None
    match method:
        case "DiffAE":
            _model, global_step = diffae_setup(wandb_id=wandb_id, device=device)
            # option to only noise and denoise a few steps
            noise_steps = None
            harmonization_model = DiffAEHarmonizationModel(
                model=_model,
                target_images=target_images,
                noise_steps=noise_steps,
            )

        case "HACA3":
            pretrained = True
            harmonization_model = HACA3HarmonizationModel(
                pretrained=pretrained,
                target_images=target_images,
            )

        case "histogram_matching":
            harmonization_model = HistogramMatchingModel(target_images=target_images)
        case "unharmonized":
            harmonization_model = UnharmonizeModel()
        case _:
            raise ValueError(f"Invalid harmonization model name: {method}")

    if method != "DiffAE" and wandb_id is not None:
        raise ValueError(f"wandb_id should be None for {method}")
    return harmonization_model, global_step


def sign_preserving_log(diff_img: np.ndarray) -> np.ndarray:
    # noop
    return diff_img
    sign = np.sign(diff_img)
    diff_img_log_abs = np.log(np.abs(diff_img) + 1)
    return sign * diff_img_log_abs


def save_figure(
    results_dir: Path,
    target_scanner: str,
    sub: str,
    session: str,
    source_img: np.ndarray,
    harmonized_img: np.ndarray,
    target_img: np.ndarray,
):
    source_img = source_img.squeeze(0)
    harmonized_img = harmonized_img.squeeze(0)
    target_img = target_img.squeeze(0)

    # diff imgs in range [-2, 2] (because images are in range [-1, 1])
    diff_img_before = source_img - target_img
    diff_img_after = harmonized_img - target_img

    # absolute improvement image in range [-1, 1] (because diff images are in range [-2, 2])
    improv_img = np.abs(diff_img_before) - np.abs(diff_img_after)
    improv_img = sign_preserving_log(improv_img)  # ->[-log(3),-0] and  [0,log(3)]

    # log scale but with sign
    diff_img_before = sign_preserving_log(diff_img_before)
    diff_img_after = sign_preserving_log(diff_img_after)
    # log(3) because diff images are in range [-log(3),0] and [0,log(3)]
    diff_img_vmin = -np.log(3)
    diff_img_vmax = np.log(3)
    # noop bounds, because images are in range [-1, 1] so theoretically the diff could be at most +- 2
    diff_img_vmin = -2
    diff_img_vmax = 2

    diff_cmap = "seismic"
    # plot original, diff_before, harmonized, diff_after, target
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))

    axs[0].imshow(source_img, cmap="gray", vmin=-1, vmax=1)
    axs[0].axis("off")
    axs[0].set_title(f"Original {sub}, {session}", wrap=True)

    axs[1].imshow(diff_img_before, cmap=diff_cmap, vmin=diff_img_vmin, vmax=diff_img_vmax)
    axs[1].axis("off")
    axs[1].set_title("Diff before")

    axs[2].imshow(harmonized_img, cmap="gray", vmin=-1, vmax=1)
    axs[2].axis("off")
    axs[2].set_title(f"Harmonized to {target_scanner}")

    axs[3].imshow(diff_img_after, cmap=diff_cmap, vmin=diff_img_vmin, vmax=diff_img_vmax)
    axs[3].axis("off")
    axs[3].set_title("Diff after")

    axs[4].imshow(target_img, cmap="gray", vmin=-1, vmax=1)
    axs[4].axis("off")
    axs[4].set_title(f"Target {target_scanner}")

    custom_cmap = sns.diverging_palette(0, 120, as_cmap=True)

    improv_img_vmin, improv_img_vmax = improv_img.min(), improv_img.max()
    # normalize vmax to be in [0,1] instead of [-2,2]
    vmin_norm = (improv_img_vmin + 2) / 4
    vmax_norm = (improv_img_vmax + 2) / 4

    cut_colors = custom_cmap(np.linspace(vmin_norm, vmax_norm, custom_cmap.N))
    cut_color_map = LinearSegmentedColormap.from_list("cut_custom", cut_colors)
    sm = plt.cm.ScalarMappable(
        cmap=cut_color_map,
        norm=plt.Normalize(vmin=improv_img_vmin, vmax=improv_img_vmax),
    )
    axs[5].imshow(
        improv_img,
        cmap=cut_color_map,
        vmin=improv_img_vmin,  # scale all the same
        vmax=improv_img_vmax,
    )
    axs[5].axis("off")
    axs[5].set_title("Improvement")
    # add colorbar
    fig.colorbar(
        sm,
        ax=axs[5],
        orientation="vertical",
        # pad=0.045,
        ticks=[improv_img_vmin, 0, improv_img_vmax],
        shrink=0.8,
    )
    axs[5].set_aspect(1.25, adjustable="box")

    # fig.suptitle(f"{sub}, {session} -> {target_scanner}")
    plt.tight_layout(w_pad=0.0, h_pad=0.1)
    plt.savefig(results_dir / f"{sub}_{session}_target-{target_scanner}.png")
    print(f"Saved {results_dir / f'{sub}_{session}_target-{target_scanner}.png'}")
    plt.close()
    # Plot histograms of original, diff before, harmonized, target, and improvement
    fig_hist, axs_hist = plt.subplots(1, 6, figsize=(18, 3.2))

    brainmask_batch = source_img > -1
    # Calculate histograms only on the foreground
    source_img_fg = source_img[brainmask_batch]
    diff_img_before_fg = diff_img_before[brainmask_batch]
    harmonized_img_fg = harmonized_img[brainmask_batch]
    target_img_fg = target_img[brainmask_batch]
    improv_img_fg = improv_img[brainmask_batch]
    diff_img_after_fg = diff_img_after[brainmask_batch]

    bins = 100
    axs_hist[0].hist(source_img_fg.ravel(), bins=bins, color="gray", range=(-1, 1), log=True)
    axs_hist[0].set_title("Original Histogram (Foreground)")

    axs_hist[1].hist(diff_img_before_fg.ravel(), bins=bins, color="red", range=(-2, 2), log=True)
    axs_hist[1].set_title("Diff Before Histogram (Foreground)")

    axs_hist[2].hist(harmonized_img_fg.ravel(), bins=bins, color="blue", range=(-1, 1), log=True)
    axs_hist[2].set_title("Harmonized Histogram (Foreground)")

    axs_hist[3].hist(diff_img_after_fg.ravel(), bins=bins, color="orange", range=(-2, 2), log=True)
    axs_hist[3].set_title("Diff After Histogram (Foreground)")

    axs_hist[4].hist(target_img_fg.ravel(), bins=bins, color="green", range=(-1, 1), log=True)
    axs_hist[4].set_title("Target Histogram (Foreground)")

    axs_hist[5].hist(improv_img_fg.ravel(), bins=bins, color="purple", range=(-2, 2), log=True)
    axs_hist[5].set_title("Improvement Histogram (Foreground)")

    plt.tight_layout(pad=0.0)
    plt.savefig(results_dir / f"{sub}_{session}_histograms_target-{target_scanner}.png")
    print(f"Saved {results_dir / f'{sub}_{session}_histograms_target-{target_scanner}.png'}")
    plt.close(fig_hist)
    # save each image individually
    source_dir = results_dir / "source"
    source_dir.mkdir(exist_ok=True)
    target_dir = results_dir / "target"
    target_dir.mkdir(exist_ok=True)
    harmonized_dir = results_dir / "harmonized"
    harmonized_dir.mkdir(exist_ok=True)

    plt.imsave(
        source_dir / f"{sub}_{session}_source.png",
        source_img,
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    plt.imsave(
        target_dir / f"{sub}_{session}_target.png",
        target_img,
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    plt.imsave(
        harmonized_dir / f"{sub}_{session}_harmonized.png",
        harmonized_img,
        cmap="gray",
        vmin=-1,
        vmax=1,
    )


if __name__ == "__main__":

    cli()
