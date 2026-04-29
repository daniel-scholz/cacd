from pathlib import Path

import numpy as np
import torch
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Spacingd
from torch import Tensor

from diffae.data.glioma import PublicGliomaTranslateDataset


class WMHDataset(PublicGliomaTranslateDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.registration_space = "mni"
        super().__init__(*args, **kwargs)
        self.preproc = Compose(
            [
                self.skull_strip,
                self.preproc,
                Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0)),
                # Orientationd(keys=["img", "seg"], axcodes="IAR"),
            ]
        )

        self.site_list = [self._fn2scanner(fp) for fp in self.subject_dirs]
        self.targets = np.array(self.site_list)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        subject_data = super().__getitem__(index)
        fp = self.subject_dirs[index]
        # add scanner info
        subject_data["scanner"] = self._fn2scanner(fp)
        # set scanner info as conditioning
        subject_data["condition"] = subject_data["scanner"]

        # vis_3d(subject_data["img"])

        return subject_data

    @property
    def mri_sequences(self) -> tuple[str, ...]:
        return ("flair",)

    @property
    def subset_names(self) -> tuple[str, ...]:
        # all_dirs = [_p for _p in self.data_dir.iterdir() if _p.is_dir()]
        # subset_names = list(
        #     set(["-".join(_p.stem.split("-")[1:-1]) for _p in all_dirs])
        # )
        # subset_names = ("Ams-GE3T", "Ams-PETMR", "Sin", "Utr", "Ams-GE15T")
        sites_train = (
            "Ams-GE3T",
            "Sin",
            "Utr",
        )
        sites_test = ("Ams-GE15T", "Ams-PETMR")
        match self.split:
            case "train" | "val":
                return sites_train
            case "test":
                return sites_test
            case _:
                raise ValueError(f"split {self.split} not supported")

    @staticmethod
    def skull_strip(data_dict: dict[str, torch.Tensor]):
        """Remove skull from MRI image."""
        img = data_dict["img"]
        mask = data_dict["brainmask"]

        img_brain = img * mask
        data_dict["img"] = img_brain

        return data_dict

    def _fn2scanner(self, fp: Path) -> str:
        fp = fp.resolve()
        parent_dir = fp.parent.parent.name
        scanner_parts = parent_dir.split("-")[1:-1]
        scanner = "-".join(scanner_parts)
        return scanner

    def fn2subject(self, fp: Path) -> str:
        return fp.parent.parent.name

    def _seq2fn(self, subject_dir: Path, mri_sequence: str) -> Path:
        fn = (
            "_".join(
                [
                    f"sub-{subject_dir.parent.name}",
                    f"ses-{subject_dir.name}",
                    f"space-{self.registration_space}",
                    mri_sequence,
                ]
            )
            + ".nii.gz"
        )
        return subject_dir / fn

    def _glob_subject_dirs(self) -> list[Path]:
        """Get filenames directly."""
        subject_dirs = []
        for subset_name in self.subset_names:
            for mri_seq in self.mri_sequences:
                subject_dirs.extend(
                    [fn for fn in self.data_dir.glob(self._get_glob_str(subset_name, mri_seq))]
                )
        return subject_dirs

    def _get_glob_str(self, subset_name: str, mri_seq: str) -> str:
        return f"*{subset_name}*/*/*space-{self.registration_space}_{mri_seq}*"
