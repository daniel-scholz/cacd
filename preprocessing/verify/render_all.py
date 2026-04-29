"""
Package for extracting the middle slice from each volume and rendering it as an image.
Serves to verify the soundness of the created dataset.
"""

import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm


def render_image(
    path: Path, save_dir: Path, atlas_slices: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Path:
    """
    Renders the middle slice of a volume as an image.
    """
    img = nib.nifti1.load(str(path))
    data = img.get_fdata()
    # axial slice
    img_slices = (
        data[data.shape[0] // 2].T,
        data[:, data.shape[1] // 2].T,
        data[:, :, data.shape[2] // 2].T,
    )
    img_slices = tuple(np.flip(s, axis=0) for s in img_slices)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(img_slices[0], cmap="gray")
    axs[0, 0].set_title("Axial")
    axs[0, 1].imshow(img_slices[1], cmap="gray")
    axs[0, 1].set_title("Coronal")
    axs[0, 2].imshow(img_slices[2], cmap="gray")
    axs[0, 2].set_title("Sagittal")

    axs[1, 0].imshow(atlas_slices[0], cmap="gray")
    axs[1, 1].imshow(atlas_slices[1], cmap="gray")
    axs[1, 2].imshow(atlas_slices[2], cmap="gray")
    # all axis off
    for i in range(3):
        axs[0, i].axis("off")
        axs[1, i].axis("off")
    # remove white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fp = save_dir / f"{path.stem}.png"
    plt.suptitle(save_fp.stem)
    plt.savefig(save_fp, bbox_inches="tight", pad_inches=0)
    plt.savefig("./verify/test.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_dir / f"{path.stem}.png"


def render_all(data_dir: Path, save_dir: Path, cur_sequence: str):
    """
    Renders the middle slice of each volume in the data directory as an image.
    """
    atlas_slices = load_atlas()
    save_dir.mkdir(exist_ok=True, parents=True)
    file_list = sorted(data_dir.glob(f"*{cur_sequence}.nii.gz"))
    pbar = tqdm(file_list)

    # Create a multiprocessing pool
    pool = multiprocessing.Pool()

    # Use the pool to map the render_image function to each file in parallel
    results = pool.starmap_async(render_image, ((file, save_dir, atlas_slices) for file in pbar))

    # Get the results
    results.get()
    print(f"Saved images to {save_dir}.")
    print(f"Total images: {len(file_list)}")

    # Close the pool
    pool.close()
    pool.join()


def load_atlas() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the atlas and returns it as a numpy array.
    """
    atlas_fp = Path("~/datasets/atlases/sub-mni152_space-mni_t1.nii.gz").expanduser()
    atlas = nib.nifti1.load(str(atlas_fp)).get_fdata()
    atlas_slices = (
        atlas[atlas.shape[0] // 2].T,
        atlas[:, atlas.shape[1] // 2].T,
        atlas[:, :, atlas.shape[2] // 2].T,
    )
    # flip slices by 180 degrees
    atlas_slices = tuple(np.flip(s, axis=0) for s in atlas_slices)
    return atlas_slices


if __name__ == "__main__":
    cur_sequence = "T2"
    data_dir = Path(f"~/datasets/ixi_reg_skullstrip/{cur_sequence}").expanduser()
    print(f"Rendering images from {data_dir}...")
    save_dir = data_dir.parent / f"{data_dir.name}_mid_slices"
    render_all(data_dir, save_dir, cur_sequence)
    print("Done rendering images.")
