import io
from pathlib import Path
from typing import Literal, Optional

import lightning.pytorch.loggers as pl_loggers
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from monai.transforms.spatial.array import Orientation
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import make_grid


def vis_3d(img: torch.Tensor, fn: str | Path = "test_3d_vis.png") -> None:
    """img is a 3D tensor of shape (C, H, W, D)."""

    img = Orientation(axcodes="IAR")(img)

    img = img.detach().cpu()
    # extract slices along each axes
    x_slice, y_slice, z_slice = center_slices(img)

    # plot each slice
    n_seq = img.size(0)
    n_col = 3  # number of views

    # get vmin and vmax for all slices
    vmin = img.min().item()
    vmax = img.max().item()

    fig = plt.figure(figsize=(10, 10))
    for i_ax in range(n_seq):
        plt.subplot(
            n_seq,
            n_col,
            i_ax * n_col + 1,
        ).imshow(x_slice[i_ax], cmap="gray", vmin=vmin, vmax=vmax)
        plt.subplot(
            n_seq,
            n_col,
            i_ax * n_col + 2,
        ).imshow(
            y_slice[i_ax],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax_img = plt.subplot(
            n_seq,
            n_col,
            i_ax * n_col + 3,
        ).imshow(
            z_slice[i_ax],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

    fig.colorbar(ax_img, ax=fig.get_axes(), orientation="horizontal")
    plt.savefig(fn)
    plt.close()


def center_slices(img: torch.Tensor) -> list[torch.Tensor]:
    """img is a 3D tensor of shape (C, H, W, D)."""
    img = img.detach()
    # extract slices along each axes
    x_slice = img[..., img.shape[-1] // 2]
    y_slice = img[..., img.shape[-2] // 2, :]
    z_slice = img[..., img.shape[-3] // 2, :, :]

    return [x_slice, y_slice, z_slice]


def grid_and_log_img(
    img: torch.Tensor,
    logger: pl_loggers.WandbLogger,
    tag: str,
    step: Optional[int] = None,
):
    # get batch size and channel dimension
    b, c = img.shape[:2]

    if img.dim() == 5:  # i.e. if 3d
        # ->(c, d, h, w), remove batch dim
        img = img.flatten(0, 1)
        # extract center slices along all 3 planes and stack in a batch
        img = torch.stack(center_slices(img), dim=1).view(b * 3, c, *img.shape[-2:])

    img_grid = make_grid(img, nrow=img.size(0) // b)
    # 0-1 norm grid, assuming -1, 1
    img_grid = (img_grid + 1) / 2
    img_grid = img_grid.detach().cpu()

    logger.log_image(
        tag,
        [img_grid],
        step=step,
    )


def plt_to_np(fig: matplotlib.figure.Figure, *args, **kwargs):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    # convert plt figure to image
    buf.seek(0)
    plt_img = np.array(Image.open(buf))

    plt.close(fig)
    return plt_img


def plot_latents(
    latents_proj: np.ndarray,
    conditions: np.ndarray,
    condition_names: npt.NDArray[np.str_],
    cond_type: Literal["cat", "cont"],
    split_list: np.ndarray,
    projection_fp: Optional[Path] = None,
) -> Optional[matplotlib.figure.Figure]:
    fig, ax = plt.subplots(figsize=(10, 5))
    splits = ["train", "val", "test"]

    # color by condition
    def plot_cat(conditions: np.ndarray, condition_names: npt.NDArray[np.str_]):
        conditions_set = np.unique(conditions[~np.isnan(conditions)])
        conditions_set.sort()

        for cond_i, cond in enumerate(conditions_set):
            cond_mask = conditions == cond
            cond_str = condition_names[cond_mask][0]
            for split_ in splits:
                # marker x if train, o if val
                marker = ("x", "o", "^")[splits.index(split_)]

                idx = np.where(cond_mask & (split_list == split_))[0]
                if len(idx) == 0:
                    print(f"Skipping {cond_str}_{split_}")
                    continue

                c = [sns.color_palette("tab10")[cond_i]] * len(idx)

                ax.scatter(
                    latents_proj[idx, 0],
                    latents_proj[idx, 1],
                    label=f"{cond_str}_{split_}",
                    s=10,
                    marker=marker,
                    c=c,
                )
        plt.legend(ncol=int(np.sqrt(len(conditions_set))), bbox_to_anchor=(1.1, 1.05))

    def plot_cont(conditions: np.ndarray):
        cmap = sns.cubehelix_palette(as_cmap=True)
        for split_ in splits:
            idx = np.where(split_list == split_)[0]
            c = conditions[idx]
            im = ax.scatter(
                latents_proj[idx, 0],
                latents_proj[idx, 1],
                c=c,
                cmap=cmap,
                label=split_,
            )
        fig.colorbar(im, ax=ax)

    match cond_type:
        case "cat":
            plot_cat(conditions, condition_names)
        case "cont":
            plot_cont(conditions)

    if projection_fp is not None:
        plt.title(projection_fp.stem)
    if projection_fp is None:
        return fig
    plt.savefig(projection_fp, bbox_inches="tight", dpi=300)
    print(f"Saved {projection_fp}")
    # plt.show()
    plt.close()


def project_latents(
    latents: np.ndarray | torch.Tensor,
    fit_latents: Optional[np.ndarray | torch.Tensor] = None,
    dim_red_fn: Literal["pca", "tsne", "umap"] = "pca",
) -> tuple[np.ndarray, PCA | TSNE]:
    seed = 42
    match dim_red_fn:
        case "pca":
            proj_fn = PCA(n_components=2, random_state=seed)

            # fit pca on train set
            proj_fn.fit(latents if fit_latents is None else fit_latents)
            latents_proj = proj_fn.transform(latents)

        case "tsne":
            proj_fn = TSNE(n_components=2)
            latents_proj = proj_fn.fit_transform(latents)

        case "umap":
            from umap import UMAP

            proj_fn = UMAP(random_state=seed, n_components=2)
            proj_fn.fit(latents if fit_latents is None else fit_latents)

            latents_proj = proj_fn.transform(latents)

        case _:  # default
            raise ValueError(f"Unknown dim_red_fn: {dim_red_fn}")

    return latents_proj, proj_fn
