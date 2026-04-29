import json
from pathlib import Path
from typing import Literal, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.base import BaseEstimator
from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from tqdm import tqdm

from diffae.config import TrainConfig
from diffae.eval.data import get_split_list, get_stage_loader
from diffae.experiment import LitModel


class PatientAttrs(TypedDict):
    scanner_int: torch.Tensor
    scanner: npt.NDArray[np.str_]
    sex: npt.NDArray[np.str_]
    seq: npt.NDArray[np.str_]
    split: npt.NDArray[np.str_]
    age: torch.Tensor


def get_w(target_label: int, clf: LogisticRegression | LinearSVC) -> torch.Tensor:
    # get decision boundary normal vector
    is_binary = len(clf.classes_) == 2

    if is_binary:
        # get normal of the separating hyperplane

        w = torch.from_numpy(clf.coef_[0])
        #  flip normal if target label is 0
        if target_label == 0:
            w = -w
    else:
        # multiclass

        # get index of current label in conditions_int
        label_idx = np.where(clf.classes_ == target_label)[0][0]

        # get decision boundary vector for current label
        w = torch.from_numpy(clf.coef_[label_idx])

        # flip sign to point away from current label
        # to the "rest" class for the current OvR classifier
        w = -w

    # w_norm = F.normalize(w, dim=0)
    w = w.float().cpu()
    return w


def get_z_sems_stats(z_sems):
    z_sem_mean = z_sems.mean(dim=0).cpu()
    z_sem_std = z_sems.std(dim=0).cpu()

    print("z_semsstats size", z_sem_mean.size(), z_sem_std.size())
    return z_sem_mean, z_sem_std


def train_model_on_latents(
    conf: TrainConfig,
    target_prop: str,
    z_sems: torch.Tensor,
    split_list: np.ndarray,
    conditions: torch.Tensor,
    latent_model_report_fp: Optional[Path],
) -> tuple[BaseEstimator, dict]:
    match target_prop:
        case "scanner" | "sex" | "sequence":
            model = LogisticRegression(
                max_iter=10000,
                random_state=conf.seed,  # class_weight="balanced"
            )
            # model = LinearSVC(
            #     random_state=conf.seed,
            #     class_weight="balanced",
            #     dual="auto",  # type: ignore
            #     max_iter=10000,
            #     C=1,
            # )
        case "age":
            model = Ridge(random_state=conf.seed)
        case _:
            raise ValueError(f"target_prop {target_prop} not supported")

    X = z_sems.clone().cpu().numpy()  # or z_sems_2d
    Y = conditions.numpy()
    # filter out nans
    X = X[~np.isnan(Y)]

    conditions = conditions[~np.isnan(Y)]
    split_list = split_list[~np.isnan(Y)]
    Y = Y[~np.isnan(Y)]

    X_train = X[split_list == "train"]
    Y_train = Y[split_list == "train"]

    target_names_train = np.unique(conditions[split_list == "train"])

    # fit scaler on train set
    # scaler = StandardScaler().fit(X_train)
    # transform all data
    # X_train_norm = scaler.transform(X_train)

    # fit classifier
    model.fit(X_train, Y_train)
    # eval on train set
    preds_train = model.predict(X_train)

    eval_set_str: Literal["train", "val", "test"] = "val"
    # normalize val set
    eval_mask = split_list == eval_set_str
    X_eval = X[eval_mask]
    # X_eval_norm = scaler.transform(X_eval)

    Y_eval = Y[eval_mask]

    # predict on val set
    preds_eval = model.predict(X_eval)

    match target_prop:
        case "scanner" | "sex" | "sequence":
            mcc_val = matthews_corrcoef(Y_eval, preds_eval)
            mcc_train = matthews_corrcoef(Y_train, preds_train)

            # save as json
            report_dict = classification_report(
                Y_eval,
                preds_eval,
                target_names=target_names_train.astype(str),
                output_dict=True,
            )
            assert isinstance(report_dict, dict)
            # update dict with mcc
            report_dict["mcc_val"] = mcc_val
            report_dict["mcc_train"] = mcc_train

            if latent_model_report_fp is not None:
                cm = confusion_matrix(Y_eval, preds_eval)
                conf_disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=target_names_train
                )
                conf_disp.plot()

                conf_matrix_fp = (
                    latent_model_report_fp.parent
                    / "confmatrix"
                    / (latent_model_report_fp.stem.replace("_report", "_confmatrix") + ".png")
                )
                conf_matrix_fp.parent.mkdir(exist_ok=True)
                # rotate x labels to avoid overlap
                plt.xticks(rotation=90)
                plt.savefig(
                    conf_matrix_fp,
                    bbox_inches="tight",
                )
                plt.show()
                plt.close()
        case "age":
            report_dict = {
                "mae": np.mean(np.abs(Y_eval - preds_eval)).item(),
                "mse": np.mean((Y_eval - preds_eval) ** 2).item(),
            }
    if latent_model_report_fp is not None:
        with open(latent_model_report_fp, "w+") as f:
            json.dump(report_dict, f, indent=2, sort_keys=True)
        print(f"saved latent model report to {latent_model_report_fp}")

    return model, report_dict


class LatentEditor:
    def __init__(
        self,
        edit_weight: float,
        z_sem_mean: torch.Tensor,
        z_sem_std: torch.Tensor,
        normalize: bool = True,
    ):
        self.edit_weight = edit_weight
        self.z_sem_mean = z_sem_mean
        self.z_sem_std = z_sem_std
        self.normalize = normalize

        self.edit_fn = self.edit if edit_weight != 0 else self.id

    def edit(
        self,
        z_sem: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        # normalize z_sem
        if self.normalize:
            z_sem_norm = (z_sem - self.z_sem_mean) / self.z_sem_std
        else:
            z_sem_norm = z_sem

        # point on the hyperplane closest to z_sem:
        # ((z_sem + dist * F.normalize(w, dim=0)) @ w + b) / torch.norm(w)

        # edit
        dist = torch.abs((z_sem @ w + bias)) / torch.norm(w)
        w_norm = F.normalize(w, dim=0)
        # add the edit vector to z_sem twice to flip it to the other side of the decision boundary
        z_sem_edit_norm = z_sem_norm + 2 * self.edit_weight * dist[:, None] * w_norm[None]

        if self.normalize:
            # denorm
            z_sem_edit = z_sem_edit_norm * self.z_sem_std + self.z_sem_mean
        else:
            z_sem_edit = z_sem_edit_norm

        return z_sem_edit

    def id(self, z_sem: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return z_sem

    def __call__(self, z_sem: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return self.edit_fn(z_sem, w, bias)


@torch.no_grad
def get_latents_full_dataset(
    diffae_model: LitModel,
    ckpt_fp: Optional[Path] = None,
    ckpt_name: Optional[str] = None,
) -> tuple[torch.Tensor, PatientAttrs]:
    """Get latent codes, either infer or load"""

    load_latents = False
    if ckpt_fp is not None:
        ckpt_dir = ckpt_fp.parent
        cond_fp = ckpt_dir / f"cond_{ckpt_name}.npy"
        cond_t2_fp = ckpt_dir / f"cond_t2_{ckpt_name}.npy"
        patient_attrs_fp = ckpt_dir / f"patient_attrs_{ckpt_name}.json"
        load_latents = cond_fp.exists() and patient_attrs_fp.exists()
    else:
        cond_fp = None
        cond_t2_fp = None
        patient_attrs_fp = None

    if load_latents:
        print(f"Loading latent features from {cond_fp}")

        conds = torch.from_numpy(np.load(cond_fp)).to(device=diffae_model.device)

        if cond_t2_fp.exists():

            conds_t2 = torch.from_numpy(np.load(cond_t2_fp)).to(device=diffae_model.device)
            print(f"Loaded conds_t2 from {cond_t2_fp}")
        else:
            conds_t2 = torch.tensor([])

        with open(patient_attrs_fp, "r") as f:
            patient_attrs = json.load(f)

        print(f"Loaded patient attrs from {patient_attrs_fp}")

        # convert to torch tensors
        patient_attrs: PatientAttrs = {
            "scanner_int": torch.tensor(patient_attrs["scanner_int"]),
            "scanner": np.array(patient_attrs["scanner"]),
            "seq": np.array(patient_attrs["seq"]),
            "split": np.array(patient_attrs["split"]),
            "age": torch.tensor(patient_attrs["age"]),
            "sex": np.array(patient_attrs["sex"]),
        }

    else:
        loader = get_stage_loader(
            diffae_model,
            stage="full",
            batch_size=2,  # diffae_model.conf.batch_size,
            num_workers=1,  # diffae_model.conf.num_workers,
        )

        print("Encoding full dataset")

        conds_list = []
        conds_t2_list = []

        patient_attrs_py = {}

        for batch in tqdm(loader, total=len(loader)):
            batch: dict[str, torch.Tensor]

            # t1 sequence per default
            imgs: torch.Tensor = batch["img"].to(diffae_model.device)

            # get z_sem for every patch
            cond = diffae_model.encode_ema(imgs)["cond"]
            conds_list.append(cond)

            imgs_t2 = batch.get("T2")
            if imgs_t2 is not None:
                imgs_t2 = imgs_t2.to(diffae_model.device)
                with torch.no_grad():
                    # get z_sem for every patch
                    cond_t2 = diffae_model.encode_ema(imgs_t2)["cond"]
                    conds_t2_list.append(cond_t2)

            # append to outputs
            for key in ["age", "sex", "scanner", "scanner_int"]:
                if key in batch:
                    if key not in patient_attrs_py:
                        patient_attrs_py[key] = []
                    patient_attrs_py[key].append(batch[key])

        conds = torch.cat(conds_list, dim=0)

        if conds_t2_list:
            conds_t2 = torch.cat(conds_t2_list, dim=0)
        else:
            conds_t2 = torch.tensor([])

        seq_list = [
            "T1",
        ] * len(conds)

        if len(conds_t2):
            # stack conds
            if len(conds_t2) != len(conds):
                raise ValueError("T1 and T2 conds have different lengths")

            conds = torch.cat([conds, conds_t2], dim=0)
            seq_list = seq_list + [
                "T2",
            ] * len(conds_t2)

        patient_attrs = {
            "age": torch.cat(patient_attrs_py["age"]),
            "sex": np.concatenate(patient_attrs_py["sex"]),
            "scanner": np.concatenate(patient_attrs_py["scanner"]),
            "scanner_int": torch.cat(patient_attrs_py["scanner_int"]),
            "split": get_split_list(diffae_model, stage="full"),
            "seq": np.array(seq_list),
        }

        # store z_sems
        if cond_fp is not None:
            np.save(cond_fp, conds.cpu().numpy())
            print(f"saved conds to {cond_fp}")
            if len(conds_t2):

                np.save(cond_t2_fp, conds_t2.cpu().numpy())
                print(f"saved conds_t2 to {cond_t2_fp}")

        if patient_attrs_fp is not None:
            with open(patient_attrs_fp, "w+") as f:
                patient_attrs_py = {k: v.tolist() for k, v in patient_attrs.items()}
                json.dump(patient_attrs_py, f, indent=2, sort_keys=True)
                print(f"saved patient attrs to {patient_attrs_fp}")

        patient_attrs["scanner_int"] = recalc_scanner_int_full(
            patient_attrs["scanner_int"], patient_attrs["scanner"]
        )

    return conds, patient_attrs


def recalc_scanner_int_full(scanner_int: torch.Tensor, scanners: np.ndarray) -> torch.Tensor:
    unique_scanners = np.unique(scanners)
    # assign condition such that each condition_name has a unique condition (integer)
    scanner_int = scanner_int.new_tensor([np.where(unique_scanners == n)[0][0] for n in scanners])
    return scanner_int


def calc_nearest_neighbor_top_accuracy(
    z_t1: torch.Tensor,
    z_t2: torch.Tensor,
    split_list: np.ndarray,
    z_type: Literal["sem", "id"],
    results_dir: Path,
):

    accuracies = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        split_mask = split_list == split
        z_t1_split = z_t1[split_mask]
        z_t2_split = z_t2[split_mask]
        # rank by distance
        z_diff_split = torch.norm(z_t1_split[None] - z_t2_split[:, None], dim=-1)

        for k in tqdm(
            range(1, z_diff_split.size(0)),
            total=z_diff_split.size(0),
            desc=f"Computing top-k accuracy for z_{z_type} distance",
        ):
            topk_min_dist_idc = torch.topk(z_diff_split, k, largest=False).indices
            topk_correct_nearest_neighbor = (
                topk_min_dist_idc == torch.arange(z_diff_split.size(0)).unsqueeze(1)
            ).any(dim=1)

            topk_acc_min_dist = topk_correct_nearest_neighbor.float().mean().item()
            accuracies[split].append(topk_acc_min_dist)

    nn_dir = results_dir / "nearest_neighbor"
    nn_dir.mkdir(exist_ok=True)

    auc = {
        split: np.trapz(accuracies[split]) / (len(accuracies[split]) - 1) for split in accuracies
    }

    # plot
    # add text for top-1 and top-5 accuracy to plot
    plt.figure()
    for split in ["train", "val", "test"]:
        # top1_acc = accuracies[split][1]
        # top5_acc = accuracies[split][5]
        # plt.text(1, top1_acc, f"{top1_acc:.2f}")
        # plt.text(5, top5_acc, f"{top5_acc:.2f}")
        x = np.linspace(0, 1, len(accuracies[split]))
        plt.plot(x, accuracies[split], label=f"{split} AUC={auc[split]:.4f}")

    # plot diagnoal dashed line for random accuracy
    plt.plot(
        torch.linspace(0, 1, 100).numpy(),
        torch.linspace(0, 1, 100).numpy(),
        "--",
        color="gray",
    )
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Top-k accuracy")
    plt.title(f"Top-k accuracy for z_{z_type} nearest neighbor")
    plt.savefig(nn_dir / f"z_{z_type}_topk_acc.png")
    print(f"saved top-k accuracy plot to {nn_dir / f'z_{z_type}_topk_acc.png'}")
    plt.close()
    # print top 1 and top5 accuracy
    print("Top-1 accuracy")
    for split in ["train", "val", "test"]:
        print(f"{split}: {accuracies[split][1]:.2f}")
    print("Top-5 accuracy")
    for split in ["train", "val", "test"]:
        print(f"{split}: {accuracies[split][5]:.2f}")

    with open(nn_dir / f"z_{z_type}_topk_acc.json", "w+") as f:
        # dump top1 and top5 accuracy
        top1_dict = {split: accuracies[split][1] for split in accuracies}
        top5_dict = {split: accuracies[split][5] for split in accuracies}
        json.dump(
            {
                "top1": top1_dict,
                "top5": top5_dict,
                "auc": auc,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"saved top-k accuracy to {nn_dir / f'z_{z_type}_topk_acc.json'}")


def optimize_z_sem(
    data_loader: torch.utils.data.DataLoader,
    diffae_model: LitModel,
    target_ixi_id: str,
    optim_fig_dir: Path,
):
    """
    Optimize reconstruction error by editing z_sem.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader for loading the input images.
        diffae_model (LitModel): The model used for image reconstruction.
        target_ixi_id (str): The target ixi_id for which the optimization is performed.
        optim_fig_dir (Path): The directory to save the optimization figures.

    Returns:
        None
    """
    T = 10
    for batch in data_loader:
        ixi_id = batch["subject_id"][0]
        if target_ixi_id != ixi_id:
            continue
        img_t1 = batch["T1"].to(diffae_model.device)
        img_t2 = batch["T2"].to(diffae_model.device)

        cond_t1 = diffae_model.encode_ema(img_t1)["cond"]
        xT_t1 = diffae_model.encode_stochastic_ema(img_t1, cond=cond_t1, T=T).detach().cpu()
        _, z_id_t1 = diffae_model.split_sem_id(cond_t1)

        cond_t2 = diffae_model.encode_ema(img_t2)["cond"].cpu().detach()
        z_sem_t2, _ = diffae_model.split_sem_id(cond_t2)

        optimal_z_sem = z_sem_t2.clone().detach().to(diffae_model.device).requires_grad_(True)
        z_id_t1 = z_id_t1.to(diffae_model.device)  # .requires_grad_(True)
        xT_t1 = xT_t1.to(diffae_model.device)  # .requires_grad_(True)
        img_t2 = img_t2.to(diffae_model.device)  # .requires_grad_(True)

        optim = torch.optim.adam.Adam([optimal_z_sem], lr=1e-2)
        n_optim_steps = 1000
        for i_optim_step in (
            pbar := tqdm(
                range(n_optim_steps),
                total=n_optim_steps,
                desc=f"Optimizing z_sem for {ixi_id}",
            )
        ):
            optimal_cond = torch.cat([optimal_z_sem, z_id_t1], dim=1)

            img_t1_to_t2 = diffae_model.render_differentiable(
                xT_t1,
                cond={"cond": optimal_cond},
                T=T,
            )

            rec_error = F.mse_loss(img_t1_to_t2, img_t2)

            # update z_sem
            optim.zero_grad()

            rec_error.backward()
            optim.step()

            if i_optim_step % 50 == 0:
                # plot t1, t2, t1_to_t2
                fig, axs = plt.subplots(1, 3, figsize=(6, 2))
                for img, title, ax in zip(
                    [img_t1, img_t1_to_t2, img_t2],
                    ["T1", "T1 to T2", "T2"],
                    axs,
                ):
                    ax.imshow(img.detach().squeeze().cpu().numpy(), cmap="gray")
                    ax.axis("off")
                    ax.set_title(title)
                # format error to be pasted into tqdm and the file name
                rec_error_fmt = f"{rec_error.detach().item():.2e}"
                optim_fp = optim_fig_dir / f"{ixi_id}" / f"{i_optim_step}_{rec_error_fmt}.png"
                optim_fp.parent.mkdir(parents=True, exist_ok=True)
                print(optim_fp)
                plt.savefig(optim_fp, dpi=100)
                plt.close()
                pbar.set_description(f"Rec error: {rec_error_fmt}")


# function that swaps latents between T1 and T2 and synthesizes resulting images
def swap_latents(
    data_loader: torch.utils.data.DataLoader,
    diffae_model: LitModel,
    swapped_fig_dir: Path,
):
    """
    Swaps all possible combination of latents (z_id, z_sem, xT) representations of two input images and visualizes the results. # noqa

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader object containing the input images.
        diffae_model (LitModel): The trained model used for encoding and rendering.
        swapped_fig_dir (Path): Directory to save the resulting images.

    Returns:
        None
    """
    print(f"Saving swap figs to {swapped_fig_dir}")
    T = 25
    for batch in data_loader:
        img_t1 = batch["T1"]
        img_t2 = batch["T2"]
        ixi_id = batch["subject_id"][0]

        def latents(img) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Computes the latent representations for an input image.

            Args:
                img (torch.Tensor): Input image.

            Returns:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    Tuple containing the semantic, identity, and transformed latents.
            """
            img = img.to(diffae_model.device)
            cond = diffae_model.encode_ema(img)["cond"]
            z_sem, z_id = diffae_model.split_sem_id(cond)

            xT = diffae_model.encode_stochastic_ema(img, cond, T=T)
            return z_sem, z_id, xT

        # generate latents (z_sem, z_id, xT) for both images
        z_sem_t1, z_id_t1, xT_t1 = latents(img_t1)
        z_sem_t2, z_id_t2, xT_t2 = latents(img_t2)

        # print differences between latents
        z_sem_diff = F.normalize(z_sem_t1) @ F.normalize(z_sem_t2).T
        z_id_diff = F.normalize(z_id_t1) @ F.normalize(z_id_t2).T
        xT_diff = torch.norm(xT_t1 - xT_t2, dim=1)
        print(
            ixi_id,
            f"z_sem_diff: {z_sem_diff.mean().item()}",
            f"z_id_diff: {z_id_diff.mean().item()}",
            f"xT_diff: {xT_diff.mean().item()}",
        )

        swap_xT: bool = False
        if swap_xT:
            latent_permutations = [
                [z_sem_t1, z_id_t1, xT_t1],
                [z_sem_t1, z_id_t1, xT_t2],
                [z_sem_t1, z_id_t2, xT_t1],
                [z_sem_t1, z_id_t2, xT_t2],
                [z_sem_t2, z_id_t1, xT_t1],
                [z_sem_t2, z_id_t1, xT_t2],
                [z_sem_t2, z_id_t2, xT_t1],
                [z_sem_t2, z_id_t2, xT_t2],
            ]

            latent_descs = [
                "T1",
                ["z_sem_t1", "z_id_t1", "xT_t1"],
                ["z_sem_t1", "z_id_t1", "xT_t2"],
                ["z_sem_t1", "z_id_t2", "xT_t1"],
                ["z_sem_t1", "z_id_t2", "xT_t2"],
                "T2",
                ["z_sem_t2", "z_id_t1", "xT_t1"],
                ["z_sem_t2", "z_id_t1", "xT_t2"],
                ["z_sem_t2", "z_id_t2", "xT_t1"],
                ["z_sem_t2", "z_id_t2", "xT_t2"],
            ]
        else:
            latent_permutations = [
                [z_sem_t1, z_id_t1, xT_t1],
                [z_sem_t1, z_id_t2, xT_t2],
                [z_sem_t2, z_id_t1, xT_t1],
                [z_sem_t2, z_id_t2, xT_t2],
            ]
            latent_descs = [
                "T1",
                ["z_sem_t1", "z_id_t1"],
                ["z_sem_t1", "z_id_t2"],
                "T2",
                [
                    "z_sem_t2",
                    "z_id_t1",
                ],
                [
                    "z_sem_t2",
                    "z_id_t2",
                ],
            ]

        imgs = [img_t1]
        all_zsems = torch.cat([latents_[0] for latents_ in latent_permutations], dim=0)
        all_zids = torch.cat([latents_[1] for latents_ in latent_permutations], dim=0)
        all_xTs = torch.cat([latents_[2] for latents_ in latent_permutations], dim=0)
        render_imgs = (
            diffae_model.render(
                all_xTs,
                cond={"cond": torch.cat([all_zsems, all_zids], dim=1).to(diffae_model.device)},
                T=T,
            )
            .detach()
            .cpu()
        )
        imgs.extend(render_imgs)
        # insert t2 image in the middle
        imgs.insert(len(imgs) // 2 + 1, img_t2)

        imgs_t1 = imgs[: len(imgs) // 2]
        imgs_t2 = imgs[len(imgs) // 2 :]
        diff_imgs_t1 = [torch.abs(imgs_t1[0] - img) for img in imgs_t1]
        diff_imgs_t2 = [torch.abs(imgs_t2[0] - img) for img in imgs_t2]
        imgs = [imgs_t1, diff_imgs_t1, imgs_t2, diff_imgs_t2]

        desc_t1 = latent_descs[: len(latent_descs) // 2]
        desc_t2 = latent_descs[len(latent_descs) // 2 :]
        diff_desc_t1 = [f"{d} - T1" for d in desc_t1]
        diff_desc_t2 = [f"{d} - T2" for d in desc_t2]
        latent_descs = [desc_t1, diff_desc_t1, desc_t2, diff_desc_t2]

        fig, axs = plt.subplots(
            len(imgs), len(imgs[0]), figsize=(len(imgs[0]) * 2.5, len(imgs) * 2)
        )

        for row, (img_set, title_set, ax_row) in enumerate(zip(imgs, latent_descs, axs)):
            for img, title, ax in zip(img_set, title_set, ax_row):
                ax.imshow(img.squeeze().cpu().numpy(), cmap="gray")

                ax.axis("off")
                is_edited_to_t1 = all("t1" in t for t in title[1:]) and "t1" not in title[0]
                is_edited_to_t2 = all("t2" in t for t in title[1:]) and "t2" not in title[0]
                if is_edited_to_t1 or is_edited_to_t2:
                    # add red border despite axis off
                    ax.spines["top"].set_color("red")
                    ax.spines["right"].set_color("red")
                    ax.spines["bottom"].set_color("red")
                    ax.spines["left"].set_color("red")
                    ax.spines["top"].set_linewidth(2)
                    ax.spines["right"].set_linewidth(2)
                    ax.spines["bottom"].set_linewidth(2)
                    ax.spines["left"].set_linewidth(2)

                    # set axis on
                    ax.axis("on")
                    ax.set_xticks([])
                    ax.set_yticks([])

                ax.set_title(title)

        plt.suptitle(f"{ixi_id}")
        plt.tight_layout()
        fp = swapped_fig_dir / f"{ixi_id}.png"
        plt.savefig(fp, bbox_inches="tight", dpi=200, pad_inches=0)
        print(f"saved {fp}")
        plt.close()
