# %%
# append the path of the parent directory to sys.path (hacky)
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data
from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from diffae.choices import ModelName
from diffae.config import TrainConfig
from diffae.eval.latent import (
    LatentEditor,
    get_latents_full_dataset,
    get_w,
    get_z_sems_stats,
    train_model_on_latents,
)
from diffae.eval.sliding_window_infer import MySlidingWindowInferer
from diffae.experiment import LitModel
from diffae.vis_utils import center_slices


def harmonize_full_dataset(
    model: LitModel,
    scanner: npt.NDArray[np.str_],
    target_scanner_int: int,
    loader_full_diffae: torch.utils.data.DataLoader,
    new_ds_dir: Path,
    swap_mode: Literal["mean", "median", "random"] = "mean",
    conds: Optional[torch.Tensor] = None,
) -> tuple[list[str], list[torch.Tensor]]:
    """Calculate average z_sem for each condition and swap z_sem of test set with the average."""

    if conds is None:
        conds, patient_attrs = get_latents_full_dataset(model)

    z_sems, z_ids = model.split_sem_id(conds)

    match swap_mode:
        case "mean":
            z_sem_target = z_sems[scanner == target_scanner_int].mean(0)
        case "median":
            z_sem_target = z_sems[scanner == target_scanner_int].median(0).values
        case "random":
            rand_idx = np.random.choice(np.where(scanner == target_scanner_int)[0])
            z_sem_target = z_sems[rand_idx]
        case _:
            raise ValueError(f"Invalid swap_mode: {swap_mode}")

    edited_imgs = []
    ixi_ids_edited = []

    for i, test_batch in tqdm(
        enumerate(loader_full_diffae), total=len(loader_full_diffae), desc="Editing"
    ):
        diffae_img = test_batch["img"]
        if test_batch["scanner"][0] == target_scanner_int:
            continue
        ixi_ids_edited.append(test_batch["subject_id"][0])

        # save edited image to drive
        og_fp = Path(test_batch["fp"][0][0])
        og_seq_dir = og_fp.parent

        # new ds dir
        new_fp = new_ds_dir / og_seq_dir.name / og_fp.name
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        if not new_fp.exists():
            # save edited image
            with torch.no_grad():
                # edited_img: 1 x C x D x H x W
                edited_img = image_editor(diffae_img)
            if edited_img[0, 0].ndim == 3:
                nibabel.nifti1.save(
                    nibabel.nifti1.Nifti1Image(edited_img[0, 0].cpu().numpy(), np.eye(4)),
                    new_fp,
                )
            elif edited_img[0, 0].ndim == 2:
                np.save(new_fp, edited_img[0, 0].cpu().numpy())
            else:
                raise ValueError(f"Invalid edited_img shape: {edited_img.shape}")
            print(f"saved edited image to {new_fp}")
        else:
            if new_fp.name.endswith(".nii.gz"):
                edited_img = nibabel.nifti1.load(new_fp).get_fdata()
            elif new_fp.name.endswith(".npy"):
                edited_img = np.load(new_fp)
            else:
                raise ValueError(f"Invalid file extension: {new_fp.suffix}")

            print(f"loaded edited image from {new_fp}")
            edited_img = torch.from_numpy(edited_img).unsqueeze(0).unsqueeze(0).float()

        edited_imgs.append(edited_img)
        # save first image
        if i == 0:
            if edited_img[0, 0].ndim == 3:
                plt.imshow(center_slices(diffae_img[0, 0])[0].cpu().numpy())
                plt.savefig("test_ds_img.png")
                plt.close()
                plt.imshow(center_slices(edited_img[0, 0])[0].cpu().numpy())
                plt.savefig("test_rec.png")
                plt.close()
            elif edited_img[0, 0].ndim == 2:
                plt.imshow(diffae_img[0, 0].cpu().numpy())
                plt.savefig("test_ds_img.png")
                plt.close()
                plt.imshow(edited_img[0, 0].cpu().numpy())
                plt.savefig("test_rec.png")
                plt.close()
            else:
                raise ValueError(f"Invalid edited_img shape: {edited_img.shape}")

        # # stop after 10 images
        # if i == 1:
        #     break

    return ixi_ids_edited, edited_imgs


# %% [markdown]

# ## Edit each image of the test set


# %%
class ImageEditorWithClf:
    def __init__(
        self,
        model: LitModel,
        latent_editor: LatentEditor,
        clf: LogisticRegression | LinearSVC,
        sw_batch_size: int,
        target_cond: int,
        patch_wise=False,
    ):
        self.model = model
        self.latent_editor = latent_editor
        self.clf = clf
        self.inferer = None
        # self.target_cond = target_cond
        if patch_wise:
            self.inferer = MySlidingWindowInferer(
                patch_size=(
                    model.conf.img_size
                    if isinstance(model.conf.img_size, tuple)
                    else (model.conf.img_size,) * model.conf.dims
                ),
                overlap=0.0,
                padding_mode="replicate",
                sw_batch_size=sw_batch_size,
            )
        self.target_cond = target_cond

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        print(img.shape)
        img = img.to(self.model.device)
        cond: torch.Tensor = self.model.encode_ema(img)["cond"]
        if self.model.conf.model_name == ModelName.beatgans_autoenc_id:
            z_sem, z_id = self.model.split_sem_id(cond.cpu().detach())
        else:
            # use whole cond for editing
            z_sem = cond.cpu().detach()
            z_id = None

        if self.inferer is None:
            x_T: torch.Tensor = self.model.encode_stochastic_ema(img, cond.to(img.device))
        else:
            raise NotImplementedError("Patch-wise inference not supported")
            x_T: torch.Tensor = self.inferer(
                img, self.model.encode_stochastic_ema, cond=cond.to(img.device)
            )
        cond = cond.cpu().detach()

        # predict class of z_sem
        pred = self.clf.predict(z_sem.numpy())[0]
        print(f"Predicted class: {pred}, target class: {self.target_cond}")

        # get editing direction
        w = get_w(self.target_cond, self.clf)
        b = torch.from_numpy(self.clf.intercept_.astype(np.float32))

        # w_norm = F.normalize(w, dim=0)
        # get signed distance of z_sem to decision boundary
        dist = (z_sem @ w + b) / torch.norm(w)

        # point on decision boundary closest to z_sem
        # z_boundary = ((z_sem + dist * w_norm) @ w + b) / torch.norm(w)

        print(f"Signed Distance to decision boundary: {dist}")

        # edit latent to get z_sem in new class
        z_sem_edit = self.latent_editor(z_sem, w, b)
        dist_edit = (z_sem_edit @ w + b) / torch.norm(w)
        print(f"Signed Distance to decision boundary after edit: {dist_edit}")

        if z_id is not None:
            cond_edit = self.model.combine_sem_id(z_sem_edit, z_id)
        else:
            cond_edit = z_sem_edit

        # render the edited image
        if self.inferer is None:
            synth_img = self.model.render(
                x_T,
                cond={"cond": cond_edit.to(device=self.model.device, dtype=x_T.dtype)},
            )
        else:
            synth_img = self.inferer(
                x_T, self.model.render, cond={"cond": cond_edit.to(self.model.device)}
            )
        synth_img: torch.Tensor
        return synth_img


# %%


def synthesize_images_with_clf(
    test_loader,
    image_editor: ImageEditorWithClf,
    new_ds_dir: Path,
    target_cond: int,
) -> tuple[list[str], list[torch.Tensor]]:
    edited_imgs = []
    ixi_ids_edited = []
    for i_batch, batch in enumerate(tqdm(test_loader)):
        diffae_img = batch["img"]
        condition = batch["condition"]
        if condition == target_cond:
            continue
        ixi_ids_edited.extend(batch["subject_id"])

        # save edited image to drive
        og_fp = Path(batch["fp"][0][0])
        og_seq_dir = og_fp.parent

        # create directory to save images to
        og_png_dir = og_seq_dir.parent / (og_seq_dir.name + "_png")
        og_png_dir.mkdir(parents=True, exist_ok=True)
        png_edited_dir = new_ds_dir / (og_seq_dir.name + "_png")
        png_edited_dir.mkdir(parents=True, exist_ok=True)

        edited_fp = new_ds_dir / og_seq_dir.name / og_fp.name
        edited_fp.parent.mkdir(parents=True, exist_ok=True)
        if not edited_fp.exists():
            # save edited image
            with torch.no_grad():
                edited_img = image_editor(diffae_img)
            if edited_fp.suffix == ".npy":
                png_fp = og_png_dir / og_fp.with_suffix(".png").name
                png_edited_fp = png_edited_dir / png_fp.name

                np.save(edited_fp, edited_img[0, 0].cpu().numpy())
                # save edited as png
                plt.imsave(
                    png_edited_fp,
                    (edited_img[0, 0].cpu().numpy() + 1) / 2,
                    cmap="gray",
                )
                # save original as png
                plt.imsave(
                    png_fp,
                    (diffae_img[0, 0].cpu().numpy() + 1) / 2,
                    cmap="gray",
                )
                print(f"saved original image to {png_fp}")
                # overwrite new_fp with png_fp to log the png file for easier vis
                edited_fp = png_edited_fp
            elif edited_fp.name.endswith(".nii.gz"):
                nibabel.nifti1.save(
                    nibabel.nifti1.Nifti1Image(edited_img[0, 0].cpu().numpy(), np.eye(4)),
                    edited_fp,
                )
            else:
                raise ValueError(f"File extension {edited_fp.suffix} not supported")
            print(f"saved edited image to {edited_fp}")
        else:
            if edited_fp.name.endswith(".nii.gz"):
                edited_img = nibabel.nifti1.load(edited_fp).get_fdata()
            elif edited_fp.name.endswith(".npy"):
                edited_img = np.load(edited_fp)
            else:
                raise ValueError(f"File extension {edited_fp.suffix} not supported")

            print(f"loaded edited image from {edited_fp}")
            edited_img = torch.from_numpy(edited_img).unsqueeze(0).unsqueeze(0).float()

        # append to list
        edited_imgs.append(edited_img.detach().cpu())

    return ixi_ids_edited, edited_imgs


def edit_imgs_with_clf(
    edit_weight: float,
    conf: TrainConfig,
    model: LitModel,
    z_sems: torch.Tensor,
    z_id: torch.Tensor,
    conditions: torch.Tensor,
    split_list: npt.NDArray[np.str_],
    test_loader_diffae: torch.utils.data.DataLoader,
    new_ds_dir: Path,
    sw_batch_size: int,
    patch_wise: bool,
    target_cond: int,
) -> tuple[list[str], list[torch.Tensor]]:
    assert not patch_wise, "patch wise inference not implemented for clf editing"

    clf, _ = train_model_on_latents(  # type:ignore
        conf,
        "scanner",
        z_sems,
        split_list,
        conditions,
        None,
    )

    clf: LinearSVC | LogisticRegression

    z_sem_mean, z_sem_std = get_z_sems_stats(z_sems)

    latent_editor = LatentEditor(edit_weight, z_sem_mean, z_sem_std, normalize=False)

    image_editor = ImageEditorWithClf(
        model=model,
        latent_editor=latent_editor,
        clf=clf,
        sw_batch_size=sw_batch_size,
        patch_wise=patch_wise,
        #  target condition as integer
        target_cond=target_cond,
    )

    ixi_ids_edited, edited_imgs = synthesize_images_with_clf(
        test_loader_diffae, image_editor, new_ds_dir, target_cond
    )

    return ixi_ids_edited, edited_imgs
