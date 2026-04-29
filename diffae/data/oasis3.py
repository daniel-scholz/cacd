from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch

from diffae.data.ixi import IXIDataset


class OASIS3Dataset(IXIDataset):

    def __init__(
        self,
        data_dir: Path,
        spatial_dims: Literal[2] | Literal[3],
        split: Literal["train"] | Literal["val"] | Literal["test"],
        fit_sites: tuple[str, ...],
        test_sites: tuple[str, ...],
        target_prop: Literal["age"] | Literal["scanner"] | Literal["sex"] = "scanner",
        norm_range: tuple[float, float] = ...,
        augmentations: Callable[..., Any] | None = None,
        biasfield_corrected=True,
        **kwargs,
    ):
        super().__init__(
            data_dir,
            spatial_dims,
            split,
            fit_sites,
            test_sites,
            target_prop,
            norm_range,
            augmentations,
            biasfield_corrected,
            **kwargs,
        )
        self.fit_sites = ()
        self.test_sites = ()

    def _load_labels(self):
        scanners = np.array(["oasis3"]).repeat(len(self.subjects_fps))
        ages = np.zeros(len(self.subjects_fps))
        sexes = np.array(["m"]).repeat(len(self.subjects_fps))
        self.labels = pd.DataFrame({"scanner": scanners, "age": ages, "sex": sexes})

    def _load_subject_fps(self, data_dir: Path, spatial_dims: Literal[2] | Literal[3]):
        file_ext = "nii.gz"
        # dont set flag because it is already in the name
        self.use_affine = False

        if self.use_biasfield_corrected:
            data_dir = data_dir.with_name(data_dir.name + "_bfcorrected")

        self.subjects_fps = sorted(list(data_dir.rglob(f"*T1w.{file_ext}")))
        print(f"Found {len(self.subjects_fps)} subjects")

    def get_patient_attrs(self, index):
        # return only scanner
        return {
            "scanner": self.labels.iloc[index]["scanner"],
            "age": self.labels.iloc[index]["age"],
            "sex": self.labels.iloc[index]["sex"],
        }

    @property
    def mri_sequences(self) -> tuple[str, ...]:
        if self._mri_sequences is not None:
            return self._mri_sequences
        return ("T1w",)

    @property
    def subset_names(self):
        all_subsets = ("OAS",)

        return all_subsets

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[Path]]:
        out = super().__getitem__(index)

        return out
