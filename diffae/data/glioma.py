from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np
import torch
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CenterSpatialCropd
from torch.utils.data import Dataset

from diffae.transforms import ScaleIntensityRangePercentilesForegroundd


class PublicGliomaDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        spatial_dims: Literal[2, 3],
        split: Literal["train", "val", "test"],
        img_size: int,
        mri_sequences: Optional[tuple[str, ...]] = None,
        norm_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.split = split
        self.spatial_dims = spatial_dims

        self._mri_sequences = mri_sequences

        data_dir = Path(data_dir)
        self.data_dir = data_dir
        print(f"loading data from {data_dir} (split: {split})")

        self.subject_dirs = self._glob_subject_dirs()
        # sort  subject directories by name
        self.subject_dirs = sorted(self.subject_dirs, key=lambda x: x.name)

        self._gt_label = "seg"
        self._brain_mask = "brainmask"
        self.mri_modes = [*self.mri_sequences, self._gt_label, self._brain_mask]

        self.preproc = Compose(
            [
                self.reshape_dict,
                ScaleIntensityRangePercentilesForegroundd(
                    keys=["img", "brainmask"],
                    channel_wise=True,
                    lower=0.005,
                    upper=0.995,
                    b_min=norm_range[0],
                    b_max=norm_range[1],
                    clip=True,
                ),
            ]
        )

        self.img_size = img_size
        self._init_crop()

    def _init_crop(
        self,
    ) -> None:
        self.crop = CenterSpatialCropd(
            keys=["img", "seg", "brainmask"],
            roi_size=self.img_size,
            allow_missing_keys=True,
        )

    @property
    def mri_sequences(self) -> tuple[str, ...]:
        if self._mri_sequences is None:
            # fallback to default sequences
            return ("t1", "t1c", "t2", "flair")
        return self._mri_sequences

    def reshape_dict(self, x):
        return {k: v.view(-1, *v.shape[-self.spatial_dims :]) for k, v in x.items()}

    @property
    def subset_names(self):
        match self.split:
            case "train" | "val":
                return (
                    "brats_2021_train",
                    "brats_2021_valid",
                    "erasmus",
                    "lumiere",
                    "rembrandt",
                    "ucsf_glioma",
                    "upenn_gbm",
                )
            case "test":
                return ("tcga",)
            case _:
                raise ValueError(f"split {self.split} not supported")

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        subject_dir = self.subject_dirs[index]

        subject_seq_fns = {seq: self._seq2fn(subject_dir, seq) for seq in self.mri_sequences}
        subject_data = {k: self._load_mri(fn) for k, fn in subject_seq_fns.items()}
        subject_data["img"] = torch.stack(
            [subject_data.pop(mode) for mode in self.mri_sequences], dim=0
        )

        subject_data["seg"] = self._load_mri(self._seq2fn(subject_dir, self._gt_label))
        subject_data["index"] = torch.tensor(index)

        # normalize data to zero mean and unit variance
        subject_data = self.preproc(subject_data)

        # crop data randomly
        subject_data = self.crop(subject_data)

        return subject_data

    def fn2subject(self, fp: Path) -> str:
        """Split the path to the file and return the patient id"""
        subject_parts = fp.with_suffix("").stem.split("_")
        patient_str = subject_parts[0].replace("sub-", "")
        return patient_str

    def _seq2fn(self, subject_dir: Path, mri_sequence: str) -> Path:
        fn = (
            "_".join(
                [
                    f"sub-{subject_dir.parent.name}",
                    f"ses-{subject_dir.name}",
                    "space-sri",
                    mri_sequence,
                ]
            )
            + ".nii.gz"
        )
        return subject_dir / fn

    def _glob_subject_dirs(
        self,
    ) -> list[Path]:
        subject_dirs = []
        for subset_name in self.subset_names:
            subset_dir = self.data_dir / subset_name
            subject_dirs.extend([fn for fn in subset_dir.glob("*/preop")])
        subject_dirs = sorted(subject_dirs)
        return subject_dirs

    def _load_mri(self, fn: Path) -> torch.Tensor:
        """Load MRI data from a file and convert it to a torch tensor."""

        data = nib.load(fn)
        np_data = data.get_fdata()
        torch_data = torch.from_numpy(np_data).float()

        return torch_data


class PublicGliomaTranslateDataset(PublicGliomaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sequence_list = [self._fn2seq(fp) for fp in self.subject_dirs]
        self.targets = np.array(self._sequence_list)

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        fp = self.subject_dirs[index]
        subject_data = {"img": self._load_mri(fp)[None]}
        subject_data["fn"] = str(fp)

        subject_data["seg"] = self._load_mri(self._seq2fn(fp.parent, self._gt_label))[None]

        if (fn_brainmask := self._seq2fn(fp.parent, self._brain_mask)).exists():
            subject_data["brainmask"] = self._load_mri(fn_brainmask)[None]
        else:
            # create dummy brainmask
            subject_data["brainmask"] = subject_data["img"] > 0

        subject_data["index"] = torch.tensor(index)
        subject_data["seq"] = self._fn2seq(fp)
        subject_data["condition"] = subject_data["seq"]

        # normalize data to zero mean and unit variance
        subject_data = self.preproc(subject_data)

        # crops data randomly within brain region
        subject_data = self.crop(subject_data)
        return subject_data

    def _fn2seq(self, fp: Path) -> str:
        return fp.with_suffix("").stem.split("_")[-1]

    def _glob_subject_dirs(self) -> list[Path]:
        # store individual filenames instead of subject directories
        subject_dirs = super()._glob_subject_dirs()
        fps = []
        for subject_dir in subject_dirs:
            niftis = [self._seq2fn(subject_dir, seq) for seq in self.mri_sequences]
            fps.extend(niftis)
        sorted(fps)
        return fps
