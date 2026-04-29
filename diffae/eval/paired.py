import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import wandb
from matplotlib import rc
from matplotlib.colors import ListedColormap
from sklearn.calibration import LinearSVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from diffae.config import TrainConfig
from diffae.data.on_harmony import OnHarmonyDataset
from diffae.vis_utils import plt_to_np

if TYPE_CHECKING:
    from diffae.experiment import LitModel


plt.rcParams["image.cmap"] = "Accent"


def paired_analysis(
    conf: TrainConfig,
    diffae_model: "LitModel",
    results_dir: Optional[Path] = None,
) -> Optional[dict[str, float]]:
    if results_dir is not None:
        paired_results_dir = results_dir / "paired"
        paired_results_dir.mkdir(exist_ok=True)
        print(f"Saving results to {paired_results_dir}")
    else:
        print("Evaluating during training")
        paired_results_dir = None

    paired_dataset = load_paired_dataset(conf)

    subjects, scanners, _, imgs_torch, brainmasks = load_images_subjects_scanners_sessions(
        paired_dataset
    )
    unique_scanners = np.unique(scanners)
    unique_subjects = np.unique(subjects)

    imgs_torch = imgs_torch.to(diffae_model.device)
    imgs = imgs_torch.cpu().numpy()  # for plotting

    z_sem, z_id = encode_latents(conf, diffae_model, imgs_torch)

    # plot_latents_by_sub_and_scanner(
    #     subjects, scanners, z_sem, z_id, paired_results_dir, diffae_model
    # )

    # instance classification on identity latents
    classify_latents_by_sub_and_scanner(
        z_sem=z_sem,
        z_id=z_id,
        subjects=subjects,
        scanners=scanners,
        paired_results_dir=paired_results_dir,
        experiment=diffae_model,
    )

    if paired_results_dir is not None and False:
        # only evaluate harmonization not during training
        df_results_harmonization, df_eval_z_sem = evaluate_harmonization(
            subjects,
            scanners,
            imgs_torch,
            unique_scanners,
            unique_subjects,
            imgs,
            z_sem,
            z_id,
            diffae_model,
            paired_results_dir,
        )
        # store all results in csv
        df_results_harmonization.to_csv(
            paired_results_dir / "results_harmonization.csv", index=False
        )

        # average all columns into a dictionary
        results_harmonization = (
            df_results_harmonization.select_dtypes(include="number").mean().to_dict()
        )

        # calculate averaged improvement
        results_harmonization["improvement_percent_mean"] = float(
            df_results_harmonization["improvement"].mean()
            / df_results_harmonization["mse_wo_harm"].mean()
        )

        print(json.dumps(results_harmonization, indent=2, sort_keys=True))
        with open(paired_results_dir / "results_harmonization_mean.json", "w") as f:
            json.dump(results_harmonization, f, indent=2, sort_keys=True)

        df_eval_z_sem.to_csv(paired_results_dir / "eval_z_sem.csv", index=False)

        print(f"Saved evaluation results to {paired_results_dir}")

        return results_harmonization


def evaluate_harmonization(
    subjects: npt.NDArray[np.str_],
    scanners: npt.NDArray[np.str_],
    imgs_torch: torch.Tensor,
    uniqe_scanners: npt.NDArray[np.str_],
    unique_subjects: npt.NDArray[np.str_],
    imgs: npt.NDArray[np.float32],
    z_sem_np: npt.NDArray[np.float32],
    z_id_np: npt.NDArray[np.float32],
    diffae_model: "LitModel",
    paired_results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # move latents back to GPU
    z_sem = torch.from_numpy(z_sem_np).to(diffae_model.device)
    z_id = torch.from_numpy(z_id_np).to(diffae_model.device)
    latents_original = diffae_model.combine_sem_id(z_sem, z_id)

    df_harm_results_list = []
    df_eval_z_sem_list = []

    # harmonize images to a single center
    for target_scanner in tqdm(uniqe_scanners, desc="Harmonizing images"):
        target_scanner_mask = torch.from_numpy(scanners == target_scanner).to(diffae_model.device)

        (mean_target_sem, df_eval_z_sem) = analyze_target_z_sem(z_sem, target_scanner_mask)
        df_eval_z_sem["scanner"] = target_scanner

        df_eval_z_sem_list.append(df_eval_z_sem)

        imgs_harmonized_torch = harmonize_img_to_target_with_mean(
            imgs_torch=imgs_torch,
            gen_model=diffae_model,
            mean_target_sem=mean_target_sem,
            z_id=z_id,
            latents_original=latents_original,
        )

        # plot harmonized images compared to the original image from the scanner
        imgs_diff_after_harm_to_orig = torch.abs(imgs_harmonized_torch - imgs_torch)

        # init array to store the difference between the harmonized and target images
        imgs_diff_after_harm = torch.zeros_like(
            imgs_diff_after_harm_to_orig, device=imgs_diff_after_harm_to_orig.device
        )
        imgs_diff_before_harm = torch.zeros_like(
            imgs_diff_after_harm_to_orig, device=imgs_diff_after_harm_to_orig.device
        )

        for sub in unique_subjects:
            # create mask for the current subject
            df_results_subject_scanner = calc_harm_error_per_image_pair(
                sub,
                subjects,
                scanners,
                imgs_torch,
                target_scanner,
                imgs_diff_after_harm_to_orig,
                imgs_harmonized_torch,
                imgs_diff_before_harm,
                imgs_diff_after_harm,
            )
            df_harm_results_list.append(df_results_subject_scanner)

        # create a subfigure for each subject
        plot_original_and_harmonized(
            subjects,
            scanners,
            uniqe_scanners,
            unique_subjects,
            target_scanner,
            imgs,
            # move to cpu for plotting
            imgs_harmonized_torch.cpu().numpy(),
            imgs_diff_before_harm.cpu().numpy(),
            imgs_diff_after_harm.cpu().numpy(),
            paired_results_dir,
        )
    df_harm_results = pd.concat(df_harm_results_list)
    df_eval_z_sem = pd.concat(df_eval_z_sem_list)
    return df_harm_results, df_eval_z_sem


def calc_harm_error_per_image_pair(
    sub: str,
    subjects: npt.NDArray[np.str_],
    scanners: npt.NDArray[np.str_],
    imgs: torch.Tensor,
    target_scanner: str,
    imgs_diff_after_harm_to_orig: torch.Tensor,
    imgs_harmonized: torch.Tensor,
    imgs_diff_before_harm: torch.Tensor,
    imgs_diff_after_harm: torch.Tensor,
) -> pd.DataFrame:
    mask_subject = torch.from_numpy(subjects == sub).to(imgs.device)
    mask_target_scanner = torch.from_numpy(scanners == target_scanner).to(imgs.device)

    # images of the target scanner (only one)
    img_target = imgs[mask_subject & mask_target_scanner]

    # intra-subject-intra-scanner error
    if len(img_target) > 1:
        img_target_fg_mask = img_target > -1

        # calculate the MSE between the all in img_target using broadcasting
        mse_intra_subject_intra_scanner = masked_intra_scanner_mse(img_target, img_target_fg_mask)

        img_target = img_target[:1]

    else:
        mse_intra_subject_intra_scanner = None

    # images to harmonize (not the target image)
    img_before_harmonize = imgs[mask_subject & ~mask_target_scanner]

    # harmonized images of the same subject
    img_harmonized = imgs_harmonized[mask_subject & ~mask_target_scanner]

    # difference between the original and the harmonized images (to measure edit amount)
    img_to_harmonize_diff_to_orig = imgs_diff_after_harm_to_orig[
        mask_subject & ~mask_target_scanner
    ]

    # calculate the difference between the target_image and the harmonized images
    img_diff_after_harm = torch.abs(img_harmonized - img_target)

    # calculate the difference between the original and target image
    # (to measure improvement after harmonization)
    img_diff_before_harm = torch.abs(img_before_harmonize - img_target)

    # store the diff images between harmonized and target images
    imgs_diff_after_harm[mask_subject & ~mask_target_scanner] = img_diff_after_harm
    imgs_diff_before_harm[mask_subject & ~mask_target_scanner] = img_diff_before_harm

    # calculate foreground masks for all images because MSE is only calculated on foreground mask
    fg_mask = (img_target > -1) | (img_before_harmonize > -1)

    n_fg_voxels = torch.sum(fg_mask, dim=(1, 2, 3))

    # calculate the difference between the original and harmonized images
    # corresponds to the amount of edit applied to the image

    mse_edit_amount = (
        (torch.sum(img_to_harmonize_diff_to_orig.square() * fg_mask, dim=(1, 2, 3)) / n_fg_voxels)
        .cpu()
        .numpy()
    )

    # error between original and target image (to measure improvement after harmonization)
    mse_wo_harm = (
        (torch.sum(img_diff_before_harm.square() * fg_mask, dim=(1, 2, 3)) / n_fg_voxels)
        .cpu()
        .numpy()
    )

    # error to the target image
    # corresponds to the quality of the harmonization
    mse_after_harm = (
        (torch.sum(img_diff_after_harm.square() * fg_mask, dim=(1, 2, 3)) / n_fg_voxels)
        .cpu()
        .numpy()
    )

    # create subject dataframe with same columns as df_results
    df_results_subject_scanner = pd.DataFrame()
    df_results_subject_scanner["scanner"] = scanners[
        (mask_subject & ~mask_target_scanner).cpu().numpy()
    ]
    df_results_subject_scanner["subject"] = sub
    df_results_subject_scanner["target_scanner"] = target_scanner
    df_results_subject_scanner["mse_wo_harm"] = mse_wo_harm
    df_results_subject_scanner["mse_after_harm"] = mse_after_harm
    df_results_subject_scanner["mse_intra_subject_intra_scanner"] = mse_intra_subject_intra_scanner
    df_results_subject_scanner["mse_edit_amount"] = mse_edit_amount

    improvement = mse_wo_harm - mse_after_harm
    df_results_subject_scanner["improvement"] = improvement

    # improvement in percent
    df_results_subject_scanner["improvement_percent_pairs"] = (
        df_results_subject_scanner["improvement"] / df_results_subject_scanner["mse_wo_harm"]
    )

    print(df_results_subject_scanner)

    return df_results_subject_scanner


def masked_intra_scanner_mse(
    img_target: torch.Tensor, img_target_fg_mask: torch.Tensor
) -> np.ndarray:
    _diff_img = img_target.unsqueeze(0) - img_target.unsqueeze(1)
    # remove diagonal
    _diff_img = _diff_img[
        ~torch.eye(_diff_img.size(0), dtype=torch.bool, device=img_target.device)
    ].flatten(end_dim=1)

    #

    # construct fg mask for all pairs of images
    img_target_fg_mask_pairs = img_target_fg_mask.unsqueeze(0) & img_target_fg_mask.unsqueeze(1)
    # remove diagnoal
    img_target_fg_mask_pairs = img_target_fg_mask_pairs[
        ~torch.eye(
            img_target_fg_mask_pairs.size(0),
            dtype=torch.bool,
            device=img_target.device,
        )
    ].flatten(end_dim=1)

    _diff_img_square = _diff_img.square()
    _diff_img_masked = _diff_img_square * img_target_fg_mask_pairs

    _diff_img_masked_sum = torch.sum(_diff_img_masked)
    _n_fg_voxels = torch.sum(img_target_fg_mask_pairs)

    mse_intra_subject_intra_scanner = (_diff_img_masked_sum / _n_fg_voxels).cpu().numpy()
    return mse_intra_subject_intra_scanner


def harmonize_img_to_target_with_mean(
    imgs_torch: torch.Tensor,
    gen_model: "LitModel",
    mean_target_sem: torch.Tensor,
    z_id: torch.Tensor,
    latents_original: torch.Tensor,
    T=50,  # denoising diffusion iterations
) -> torch.Tensor:
    latent_harmonized = torch.cat([mean_target_sem.expand(z_id.size(0), -1), z_id], dim=1)
    # DDIM inversion with original latents, just like in the diffAE paper
    xT = gen_model.encode_stochastic_ema(imgs_torch, cond=latents_original, T=T)

    imgs_harmonized_torch = gen_model.render(xT, cond={"cond": latent_harmonized}, T=T)

    return imgs_harmonized_torch


def analyze_target_z_sem(
    z_sem: torch.Tensor,
    target_scanner_mask: torch.Tensor,
) -> tuple[torch.Tensor, pd.DataFrame]:
    mean_target_sem = torch.mean(z_sem[target_scanner_mask], dim=0)
    std_target_sem = torch.std(z_sem[target_scanner_mask], dim=0)

    # measure how well the mean actually represents the scanner cluster
    mean_target_sem_error = torch.mean(torch.square(z_sem[target_scanner_mask] - mean_target_sem))

    results_latent_space = {
        "std": std_target_sem.mean().cpu().item(),
        "mean_error": mean_target_sem_error.cpu().item(),
    }
    df_eval_z_sem = pd.DataFrame(results_latent_space, index=[0])

    return mean_target_sem, df_eval_z_sem


def plot_original_and_harmonized(
    subjects: npt.NDArray[np.str_],
    scanners: npt.NDArray[np.str_],
    unique_scanners: npt.NDArray[np.str_],
    unique_subjects: npt.NDArray[np.str_],
    target_scanner: str,
    imgs_np: npt.NDArray[np.float32],
    imgs_harmonized: npt.NDArray[np.float32],
    imgs_diff_before_harm: npt.NDArray[np.float32],
    imgs_diff_after_harm: npt.NDArray[np.float32],
    paired_results_dir: Path,
):
    paired_results_fig_dir = paired_results_dir / "harmonization_figures" / target_scanner
    paired_results_fig_dir.mkdir(exist_ok=True, parents=True)

    for i_sub, sub in enumerate(unique_subjects):
        # filter images by subject
        sub_mask = subjects == sub
        imgs_sub = imgs_np[sub_mask]
        imgs_diff_before_harm_sub = imgs_diff_before_harm[sub_mask]
        imgs_harmonized_sub = imgs_harmonized[sub_mask]
        scanners_sub = scanners[sub_mask]
        img_target_sub = imgs_sub[scanners_sub == target_scanner]
        if len(img_target_sub) > 1:
            img_target_sub = img_target_sub[:1]
        imgs_diff_after_harm_sub = imgs_diff_after_harm[sub_mask]

        s = 1
        width = len(imgs_sub) + 1
        height = 4

        # create a subplot with 4 rows and n_scanner+1 columns.
        # 1 for the target image

        subfig, axs = plt.subplots(nrows=height, ncols=width, figsize=(s * width, s * height))
        subfig.suptitle(f"Harmonization of {sub} to {target_scanner}")

        # first column, 4th row: target image
        axs[3, 0].imshow(img_target_sub.squeeze(0).squeeze(0), cmap="gray", vmin=-1, vmax=1)
        # hide 0-2 rows and turn off axis for target image
        for ax in axs[:4, 0]:
            ax.axis("off")

        for i_scanner in range(len(imgs_sub)):
            # filter images by scanner
            scanner_sub = scanners_sub[i_scanner]
            img_sub_scanner = imgs_sub[i_scanner]
            img_harmonized_sub_scanner = imgs_harmonized_sub[i_scanner]
            img_diff_before_harm_sub_scanner = imgs_diff_before_harm_sub[i_scanner]
            img_diff_after_harm_sub_scanner = imgs_diff_after_harm_sub[i_scanner]

            # store maximumg of diff image before and after harm
            max_diff = max(
                img_diff_before_harm_sub_scanner.max(),
                img_diff_after_harm_sub_scanner.max(),
            )

            axs[0, i_scanner + 1].imshow(img_sub_scanner.squeeze(0), cmap="gray", vmin=-1, vmax=1)
            axs[1, i_scanner + 1].imshow(
                img_diff_before_harm_sub_scanner.squeeze(0),
                cmap="gray",
                vmin=0,
                vmax=max_diff,
            )
            axs[2, i_scanner + 1].imshow(
                img_harmonized_sub_scanner.squeeze(0), cmap="gray", vmin=-1, vmax=1
            )
            axs[3, i_scanner + 1].imshow(
                img_diff_after_harm_sub_scanner.squeeze(0),
                cmap="gray",
                vmin=0,
                vmax=max_diff,
            )

            # set descriptive titles
            axs[0, i_scanner + 1].set_title(scanner_sub)
            axs[0, i_scanner + 1].set_ylabel("Orig")
            axs[1, i_scanner + 1].set_ylabel("Orig - Target")
            axs[2, i_scanner + 1].set_ylabel("Harm")
            axs[3, i_scanner + 1].set_ylabel("Harm - Target")

            # .axes.get_xaxis().set_ticks([])
            # .axes.get_yaxis().set_ticks([])
            # for all axes

            for ax in axs[:, i_scanner + 1]:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        padding = 0.5
        plt.tight_layout(w_pad=padding, h_pad=padding)
        fig_fp = paired_results_fig_dir / f"harmonization_{sub}_{target_scanner}.png"
        plt.savefig(fig_fp, bbox_inches="tight")
        plt.close()


def encode_latents(
    conf: TrainConfig, diffae_model: "LitModel", imgs: torch.Tensor
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    with torch.no_grad():
        latents: torch.Tensor = diffae_model.encode_ema(imgs)["cond"]
    latents = latents.cpu()

    z_sem, z_id = diffae_model.split_sem_id(latents)
    z_sem = z_sem.numpy()
    z_id = z_id.numpy()
    return z_sem, z_id


def classify_latents_by_sub_and_scanner(
    z_sem: npt.NDArray,
    z_id: npt.NDArray,
    subjects: npt.NDArray,
    scanners: npt.NDArray,
    paired_results_dir: Optional[Path] = None,
    experiment: Optional["LitModel"] = None,
):
    results = {}
    mean_acc = train_and_eval_linear_clf(z_id, subjects)
    results["z_id_subject"] = mean_acc

    mean_acc = train_and_eval_linear_clf(z_id, scanners)
    results["z_id_scanner"] = mean_acc

    mean_acc = train_and_eval_linear_clf(z_sem, scanners)
    results["z_sem_scanner"] = mean_acc
    mean_acc = train_and_eval_linear_clf(z_sem, subjects)
    results["z_sem_subject"] = mean_acc

    results_json = json.dumps(results, indent=2, sort_keys=True)
    if paired_results_dir is not None:
        results_fp = paired_results_dir / "clf_latents_sub_scanner.json"
        with open(results_fp, "w") as f:
            f.write(results_json)
        print(f"Saved latent subject and scanner classification results to {results_fp}")
        print(results_json)

    if experiment is not None and experiment.logger is not None:
        # add prefix "paired_latents_eval" to results
        results = {f"paired_latents_eval/{k}": v for k, v in results.items()}

        # log results to wandb
        experiment.log_dict(results)


def train_and_eval_linear_clf(features: npt.NDArray, labels: npt.NDArray) -> float:
    clf = LinearSVC(max_iter=10000, random_state=42, C=1, dual=True)
    # clf = LogisticRegression(max_iter=1000, random_state=42)
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # filter out classes that only have a single sample
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels = unique_labels[counts > 1]
    valid_idx = np.isin(labels, valid_labels)
    features = features[valid_idx]
    labels = labels[valid_idx]

    accs = []
    for train_idx, test_idx in k_fold.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        # acc = roc_auc_score(y_test, clf.decision_function(X_test), multi_class="ovr")
        accs.append(acc)

    return np.mean(accs).item()


def plot_latents_by_sub_and_scanner(
    subjects: npt.NDArray,
    scanners: npt.NDArray,
    z_sem: npt.NDArray,
    z_id: npt.NDArray,
    paired_results_dir: Optional[Path] = None,
    experiment: Optional["LitModel"] = None,
):
    for dim_red_fn in ["pca", "tsne"]:
        if dim_red_fn == "pca":
            z_sem_pc = PCA(n_components=2).fit_transform(z_sem)
            z_id_pc = PCA(n_components=2).fit_transform(z_id)

        elif dim_red_fn == "tsne":
            from sklearn.manifold import TSNE

            z_sem_pc = TSNE(n_components=2).fit_transform(z_sem)
            z_id_pc = TSNE(n_components=2).fit_transform(z_id)

        markersize = 10
        s = 3
        fig, axs = plt.subplots(2, 2, figsize=(2 * s, 0.8 * s))

        ses_map = {
            "GEM": "GE MR750 ",
            "TRI": "Siemens Trio",
            "PRI": "Siemens Prisma",
            "ACH": "Philips Achieva",
            "ING": "Philips Ingenia",
        }
        for i_ses, ses in enumerate(set(scanners)):
            idx = scanners == ses
            if np.sum(idx) < 2:
                continue
            z_sem_pc_se = z_sem_pc[idx]
            z_id_pc_ses = z_id_pc[idx]

            axs[0, 0].scatter(
                z_sem_pc_se[:, 0],
                z_sem_pc_se[:, 1],
                label=f"{ses_map[ses]}",
                # label=f"{ses} (n={np.sum(idx)})""
                s=markersize,
            )
            axs[0, 1].scatter(
                z_id_pc_ses[:, 0],
                z_id_pc_ses[:, 1],
                # label=f"{ses} (n={np.sum(idx)})"
                label=f"{ses_map[ses]}",
                s=markersize,
            )
            leg = axs[0, 1].legend(
                # bbox_to_anchor=(1.05, 1.3),
                bbox_to_anchor=(1.05, 0.9),
                loc="upper left",
                handletextpad=0.1,
                # title="Scanner",
                prop={"size": 6},
            )
            leg._legend_box.align = "left"

        for i_sub, sub in enumerate(set(subjects)):
            idx = subjects == sub
            z_sem_pc_sum = z_sem_pc[idx]
            z_id_pc_sub = z_id_pc[idx]
            label = f"{i_sub}"
            axs[1, 0].scatter(
                z_sem_pc_sum[:, 0],
                z_sem_pc_sum[:, 1],
                label=label,
                s=markersize,
                marker="x",
            )
            axs[1, 1].scatter(
                z_id_pc_sub[:, 0],
                z_id_pc_sub[:, 1],
                label=label,
                s=markersize,
                marker="x",
            )

            leg = axs[1, 1].legend(
                bbox_to_anchor=(1.05, 0.8),
                loc="upper left",
                ncols=3,
                prop={"size": 6},
                columnspacing=0.83,
                handletextpad=0.1,
                # title="Subject",
            )
            leg._legend_box.align = "left"

        # set column title
        axs[0, 0].set_title(r"Contrast $z_c$", fontsize=10)
        axs[0, 1].set_title(r"Anatomy $z_a$", fontsize=10)
        # set row title
        axs[0, 0].set_ylabel("Scanner", rotation=0, labelpad=25)
        axs[1, 0].set_ylabel("Subject", rotation=0, labelpad=25)
        rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
        rc("text", usetex=True)

        for ax in axs.flat:
            ax.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )  # labels along the bottom edge are off
            ax.tick_params(
                axis="y",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False,
            )

        plt.tight_layout(pad=1)
        if paired_results_dir is not None:
            fig_fp = paired_results_dir / f"paired_latents_eval_{dim_red_fn}.png"
            plt.savefig(fig_fp, bbox_inches="tight", pad_inches=0, dpi=500)
            # pdf
            plt.savefig(fig_fp.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
            print(f"Saved subject and scanner latents plot to {fig_fp}")

        # log to wandb
        if experiment is not None and experiment.logger is not None:
            wandb_img = wandb.Image(plt_to_np(fig))
            experiment.logger.log_image(
                f"paired_latents_eval/{dim_red_fn}",
                [wandb_img],
                step=experiment.global_step,
            )
        plt.close(fig)


def load_images_subjects_scanners_sessions(
    paired_dataset: OnHarmonyDataset,
    device: torch.device = torch.device("cpu"),
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.str_], npt.NDArray[np.str_], torch.Tensor]:

    subjects = []
    sessions = []
    scanners = []
    imgs = []
    brainmasks = []

    cpu_count = torch.multiprocessing.cpu_count()

    # wrap dataset in torch.utils.data.DataLoader
    paired_dataloader = torch.utils.data.DataLoader(
        paired_dataset,
        batch_size=1,
        num_workers=min(len(paired_dataset) // 2, cpu_count - 1),
        shuffle=False,
    )
    # load complete dataset grouped by subject
    for subject_data in tqdm(
        paired_dataloader, total=len(paired_dataloader), desc="Loading images"
    ):
        subject_data.pop("index")

        brainmasks.extend([subject_data.pop(k) for k in list(subject_data.keys()) if "mask" in k])
        subjects.extend([*subject_data.pop("subject_id")] * len(subject_data))

        sessions.extend(subject_data.keys())
        imgs.extend(subject_data.values())
        scanners.extend([s[8:11] for s in subject_data.keys()])

    assert len(subjects) == len(imgs) == len(scanners)  # == len(sessions)
    print(f"Number of unique subjects: {len(set(subjects))}")
    print(f"Number of unique sessions: {len(set(sessions))}")
    print(f"Number of unique scanners: {len(set(scanners))}")
    print(f"Number of images: {len(imgs)}")

    imgs = torch.stack(imgs).to(device)
    subjects = np.array(subjects)
    sessions = np.array(sessions)
    scanners = np.array(scanners)
    brainmasks = torch.cat(brainmasks).to(device)
    return subjects, scanners, sessions, imgs, brainmasks


def load_paired_dataset(conf: TrainConfig) -> OnHarmonyDataset:
    size_suffix = "_64" if 64 in conf.img_size else ""
    paired_dataset = OnHarmonyDataset(
        data_dir=Path(f"~/datasets/on-harmony{size_suffix}").expanduser(),
        spatial_dims=conf.dims,
        img_size=conf.img_size,
        norm_range=(-1, 1),
        skullstrip=True,
        biasfield_corrected=True,
    )

    return paired_dataset
