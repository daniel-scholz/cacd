"""dataset for class paired dataset on-harmony"""

# add to path

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import Orientation
from torch.utils.data import Dataset

from diffae.transforms import ScaleIntensityRangePercentilesForegroundd

ONHScanner = Literal["GEM", "PRI", "TRI", "ACH", "ING"]


class OnHarmonyDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        spatial_dims: int,
        img_size: tuple[int, ...],
        norm_range: tuple[float, float] = (-1, 1),
        skullstrip: bool = True,
        biasfield_corrected: bool = True,
        mri_sequences: tuple[str, ...] = ("T1w",),
    ):
        super().__init__()

        if spatial_dims == 2:
            # append _2d two data dir
            data_dir = Path(str(data_dir) + "_2d")

        self.file_ext = "nii.gz" if spatial_dims == 3 else "npy"
        self.load_affine = True
        self.space = "mni"
        if self.load_affine:
            self.space = "mni_affine"
        self.skullstrip = skullstrip
        self.preproc_str: str = "_proc-skullstrip" if self.skullstrip else ""
        self.preproc_str += "_proc-bfcorrected" if biasfield_corrected else ""
        self.data_dir = data_dir
        self.spatial_dims = spatial_dims
        self._mri_sequences = mri_sequences

        self.subjects_dirs = [
            subj
            for subj in self.data_dir.glob("sub-*")
            for seq in mri_sequences
            if len(list(subj.rglob(f"*{self.preproc_str}_{seq}.{self.file_ext}"))) > 0
        ]

        if self.spatial_dims == 3:
            self.load_transform = Compose(
                [
                    LoadImage(image_only=True, ensure_channel_first=True),
                ]
            )
        else:
            self.load_transform = self._load_transform_tensor

        self.norm = ScaleIntensityRangePercentilesForegroundd(
            keys=["img", "brainmask"],
            channel_wise=True,
            lower=0.005,
            upper=0.995,
            b_min=norm_range[0],
            b_max=norm_range[1],
            clip=True,
        )
        self.crop = CenterSpatialCrop(img_size)
        print(f"Number of subjects: {len(self)}")

    def _load_transform_tensor(self, fp: Path) -> torch.Tensor:
        return torch.tensor(np.load(fp))[None]

    def __len__(self):
        return len(self.subjects_dirs)

    def __getitem__(self, index: int) -> dict:
        subject = self.subjects_dirs[index]

        sessions = list(subject.glob("ses-*"))

        subject_data = {}
        brainmasks = {}
        for sess in sessions:
            sess_imgs = []
            sess_brainmasks = []
            for mri_seq in self.mri_sequences:
                sess_fp = (
                    sess
                    / "anat"
                    / f"{subject.name}_{sess.name}_space-{self.space}{self.preproc_str}_{mri_seq}.{self.file_ext}"
                )
                if not sess_fp.exists():
                    continue

                sess_seq = self.load_transform(sess_fp)
                sess_seq = torch.nan_to_num(sess_seq, nan=0.0)
                sess_seq = self.crop(sess_seq)

                mask_fp = sess_fp.with_name(
                    f"{subject.name}_{sess.name}_space-{self.space}_proc-skullstrip_T1w_mask.{self.file_ext}"
                )

                assert mask_fp.exists(), f"Mask file {mask_fp} does not exist"
                brainmask_sess = self.load_transform(
                    # str(mask_fp).replace("_proc-bfcorrected", "")
                    mask_fp
                )

                brainmask_sess = torch.nan_to_num(brainmask_sess, nan=0.0)
                brainmask_sess = self.crop(brainmask_sess)
                assert brainmask_sess.unique().numel() == 2, f"Brainmask: {brainmask_sess.unique()}"
                # make bool
                brainmask_sess = brainmask_sess.bool()

                sess_imgs.append(sess_seq)
                sess_brainmasks.append(brainmask_sess)
            if len(sess_imgs) == 0:
                continue
            sess_imgs = torch.stack(sess_imgs)
            sess_brainmasks = torch.stack(sess_brainmasks)

            # scanner_name = sess.name[8:11]
            subject_data[sess.name] = sess_imgs
            brainmasks[sess.name] = sess_brainmasks

        all_imgs = torch.cat(
            [subject_data[sess.name] for sess in sessions if sess.name in subject_data]
        )
        all_brainmasks = torch.cat(
            [brainmasks[sess.name] for sess in sessions if sess.name in brainmasks]
        )

        all_imgs_norm = []
        all_brainmasks_from_norm = []
        for img, brainmask in zip(all_imgs, all_brainmasks):
            img_norm = self.norm({"img": img, "brainmask": brainmask})["img"]
            # if not self.skullstrip:
            #     # recalculating brainmask
            #     brainmask = img_norm > img_norm.min()

            all_brainmasks_from_norm.append(brainmask)

            all_imgs_norm.append(img_norm)
        all_imgs_norm = torch.cat(all_imgs_norm)

        subject_data.update(
            {
                sess.name: all_imgs_norm[i]
                for i, sess in enumerate((sess for sess in sessions if sess.name in subject_data))
            }
        )
        subject_data.update(
            {
                sess.name + "_mask": all_brainmasks[i]
                for i, sess in enumerate((sess for sess in sessions if sess.name in brainmasks))
            }
        )
        subject_data["index"] = index
        subject_data["subject_id"] = subject.name
        return subject_data

    @property
    def mri_sequences(self) -> tuple[str, ...]:
        return self._mri_sequences


# collate function that can handle missing keys in some sample
def on_harmony_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function that can handle missing keys in some sample.
    Leave out all missing keys samples.
    """

    # get all keys that are present in all samples
    common_keys = set(batch[0].keys())
    for sample in batch[1:]:
        common_keys = common_keys.union(set(sample.keys()))

    # filter out all keys that are not in all samples
    collated_batch = {}
    for key in common_keys:
        batched_key = []
        for sample in batch:
            if key in sample:
                batched_key.append(sample[key])
        # stack if the key is a tensor
        if isinstance(batched_key[0], torch.Tensor):
            collated_batch[key] = torch.stack(batched_key)
        else:
            collated_batch[key] = batched_key

    return collated_batch


if __name__ == "__main__":

    def main():
        data_dir = Path("~/datasets/on-harmony").expanduser()
        spatial_dims = 2
        img_size = (168, 208)
        dataset = OnHarmonyDataset(data_dir, spatial_dims, img_size=img_size, skullstrip=True)

        print(f"Number of subjects: {len(dataset)}")

        sample_subject = dataset[0]
        imgs = torch.cat(
            [sample_subject[subject_key] for subject_key in sample_subject if "ses" in subject_key]
        )
        print(imgs.shape)

        if spatial_dims == 3:
            imgs_2d = imgs[..., imgs.shape[-1] // 2]
        else:
            imgs_2d = imgs

        print(imgs_2d.shape, imgs_2d.min(), imgs_2d.max())

        fig, axs = plt.subplots(2, imgs_2d.size(0), figsize=(15, 5))

        for i, img in enumerate(imgs_2d):
            axs[0, i].imshow(img.squeeze(0), cmap="gray")
            axs[0, i].axis("off")

            # plot histogram for each image in the second row
            axs[1, i].hist(img[img > -0.99].flatten(), bins=100, color="gray", density=True)
            # set title

        plt.tight_layout()
        plt.savefig("test_on_harmony.png")

    main()
