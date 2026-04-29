from pathlib import Path
from typing import Literal

import numpy as np
import torch.utils.data
from brain_age_estimate.model import AgeRegModel
from monai.transforms.croppad.array import SpatialPad
from torchvision.utils import save_image


def load_age_reg_model(age_reg_id: str, spatial_dims: Literal[2, 3]) -> tuple[AgeRegModel, Path]:
    ckpt_dir = Path(".") / "checkpoints" / "age_est"

    # list all checkpoints
    ckpt_fps = list(ckpt_dir.glob(f"*/{age_reg_id}/*.ckpt"))

    # choose last one (best one wrt mse)
    ckpt_fps.sort(key=lambda x: int(x.stem.split("=")[-1]) if x.name != "last.ckpt" else -1)
    age_ckpt_fp = ckpt_fps[-1]

    # age_ckpt_fp = age_ckpt_dir / "epoch=543-step=7616.ckpt"
    # age_ckpt_fp = age_ckpt_dir / "last-v3.ckpt"
    # age_ckpt_fp = age_ckpt_dir / ckpt_name

    age_reg_model = AgeRegModel.load_from_checkpoint(
        checkpoint_path=age_ckpt_fp, spatial_dims=spatial_dims
    )
    age_reg_model.eval().requires_grad_(False)
    print(f"loaded age_reg_model from {age_ckpt_fp}")
    return age_reg_model, age_ckpt_fp


def infer_internal_validation_images(
    val_loader: torch.utils.data.DataLoader,
    age_reg_model: AgeRegModel,
    target_scanner: str,
) -> tuple[list[str], dict[str, float]]:
    # infer all images matching the target condition
    pred_ages = []
    gt_ages = []
    ixi_ids = []

    for i, val_batch in enumerate(val_loader):
        if i == 0:
            assert len(val_batch["subject_id"]) == 1, "only batch size 1 supported."

        cur_ixi_id = val_batch["subject_id"][0]
        img: torch.Tensor = val_batch["img"]
        age_gt = val_batch["age"][0]

        # filter out all subjects that are from target condition
        is_target_subject = val_batch["scanner"][0] == target_scanner
        if not is_target_subject:
            continue

        ixi_ids.append(cur_ixi_id)

        age_pred = age_reg_model(img.to(age_reg_model.device))

        pred_ages.append(age_pred.cpu().detach().numpy())
        gt_ages.append(age_gt)

    pred_ages = np.array(pred_ages).squeeze()
    gt_ages = np.array(gt_ages).squeeze()

    mae = np.abs(pred_ages - gt_ages).mean()
    mae_std = np.abs(pred_ages - gt_ages).std()
    print(f"intern val: {mae = } +- {mae_std = }")
    results = {"mae_internal_val": float(mae), "mae_internal_val_std": float(mae_std)}
    return ixi_ids, results


# %% [markdown]
# # Edit all the images


def infer_edited_images(
    test_loader,
    edited_imgs: list[torch.Tensor],
    ixi_ids_edited: list[str],
    age_reg_model: AgeRegModel,
) -> tuple[dict[str, float], dict[str, list]]:
    pred_ages_real = []
    pred_ages_edited = []
    gt_ages = []
    older_all = []
    ixi_ids = []

    for i_batch, (test_batch) in enumerate(test_loader):
        cur_ixi_id = test_batch["subject_id"][0]
        try:
            cur_ixi_id_idx = ixi_ids_edited.index(cur_ixi_id)
        except ValueError:
            continue
        edited_img = edited_imgs[cur_ixi_id_idx]

        age_gt = test_batch["age"][0].numpy()
        img_batch: torch.Tensor = test_batch["img"]

        dims = img_batch.ndim - 2
        original_size = img_batch.shape[-dims:]

        padder = SpatialPad(original_size, mode="constant", value=-1)
        img_batch = padder(img_batch[0])[None]
        edited_img = padder(edited_img[0])[None]

        age_pred_edit: torch.Tensor = age_reg_model(edited_img.to(age_reg_model.device))
        age_pred_edit = age_pred_edit.cpu().detach().numpy()

        age_pred_real: torch.Tensor = age_reg_model(img_batch.to(age_reg_model.device))
        age_pred_real = age_pred_real.cpu().detach().numpy()

        mae_edit = float(np.abs(age_pred_edit - age_gt).mean())
        mae_real = float(np.abs(age_pred_real - age_gt).mean())

        # older or younger
        older = age_pred_edit > age_gt

        print(
            f"age_gt: {age_gt.item():.2f}, age_pred_edit: {age_pred_edit.item():.2f},",
            f"mae_edit: {mae_edit:.2f}, mae: {mae_real:.2f}, ",
            f"older: {bool(older)}",
        )

        save_image(img_batch, "test_real.png", value_range=(-1, 1), normalize=True)
        save_image(edited_img, "test_edited.png", value_range=(-1, 1), normalize=True)
        save_image(
            torch.abs(edited_img - img_batch),
            "test_diff.png",
            # value_range=(0, 2),
            normalize=True,
        )

        gt_ages.append(age_gt)
        pred_ages_real.append(age_pred_real)
        pred_ages_edited.append(age_pred_edit)
        older_all.append(bool(older))
        ixi_ids.append(cur_ixi_id)

    # calc mae
    pred_ages_real = np.array(pred_ages_real).squeeze()
    pred_ages_edited = np.array(pred_ages_edited).squeeze()
    gt_ages = np.array(gt_ages).squeeze()

    mae_real = np.abs(pred_ages_real - gt_ages).mean()
    mae_edited = np.abs(pred_ages_edited - gt_ages)[~np.isnan(pred_ages_edited)].mean()

    mae_real_std = np.abs(pred_ages_real - gt_ages).std()
    mae_edited_std = np.abs(pred_ages_edited - gt_ages)[~np.isnan(pred_ages_edited)].std()

    print(f" {mae_real = } +- {mae_real_std = }")
    print(f" {mae_edited = } +- {mae_edited_std = }")
    n_older = np.sum(older_all)
    n_younger = len(older_all) - n_older
    print(f"older: {n_older}, younger: {n_younger}")

    results = {
        "mae_real": float(mae_real),
        "mae_real_std": float(mae_real_std),
        "mae_edited": float(mae_edited),
        "mae_edited_std": float(mae_edited_std),
        "n_older": int(n_older),
        "n_younger": int(n_younger),
    }
    per_sample_results = {
        "pred_age_real": pred_ages_real.tolist(),
        "pred_age_edited": pred_ages_edited.tolist(),
        "gt_age": gt_ages.tolist(),
        "older": older_all,
        "subject_id": ixi_ids,
    }
    return results, per_sample_results
