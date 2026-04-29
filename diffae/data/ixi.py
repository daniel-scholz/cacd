from pathlib import Path
from typing import Callable, Literal, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from diffae.data.dataframe import map_col_to_int
from diffae.data.glioma import PublicGliomaDataset


class IXIDataset(PublicGliomaDataset):
    """A wrapper around torchio.datasets.IXI."""

    def __init__(
        self,
        data_dir: Path,
        spatial_dims: Literal[2, 3],
        split: Literal["train", "val", "test"],
        fit_sites: tuple[str, ...],
        test_sites: tuple[str, ...],
        target_prop: Literal["age", "scanner", "sex"] = "scanner",
        norm_range: tuple[float, float] = (-1.0, 1.0),
        augmentations: Optional[Callable] = None,
        biasfield_corrected=True,
        load_anat_label_maps: bool = False,
        slices_around_middle: int = 0,  # default: only middle slice
        **kwargs,
    ):

        exclude_ids = []
        self.use_affine = True
        if self.use_affine:
            exclude_ids = ["IXI077", "IXI456"]
            data_dir = Path(str(data_dir).replace("reg", "reg_affine"))
        self.exclude_ids = np.array(exclude_ids)
        if spatial_dims == 2:
            # append _2d to data dir
            # data_dir = Path(str(data_dir) + "_2d")

            # # -> all slices, leads to errors because we get empty slices sometimes
            # self.slices_around_middle = 192 // 2
            # self.slices_around_middle = 0  # -> only middle slice
            assert (
                0 <= slices_around_middle <= (192 // 2)
            ), "slices_around_middle must be between 0 and 96"

            self.slices_around_middle = slices_around_middle
            self.slicer_rng = torch.Generator().manual_seed(42)

        self.split = split
        self.fit_sites = fit_sites  # e.g., ("Guys", "HH")
        self.test_sites = test_sites  # e.g., ("IOP",)
        self.augmentations = augmentations
        self.target_prop = target_prop
        self._mri_sequences = kwargs.pop("mri_sequences", None)

        self.data_dir = data_dir
        self.use_biasfield_corrected = biasfield_corrected
        self.load_anat_label_maps = load_anat_label_maps

        self._load_subject_fps(data_dir, spatial_dims)
        print("Number of subjects in dataset:", len(self.subjects_fps))  # 577 T1 images
        super().__init__(
            data_dir,
            spatial_dims=spatial_dims,
            split=split,
            norm_range=norm_range,
            **kwargs,
        )

        self._init_subject_ids()

        self._load_labels()

        # init subject ids to match current state of base_dataset
        self._init_subject_ids()

        if target_prop == "age":
            self.scanner_int = self.labels["age"].to_numpy()
            self.target_names = self.labels["age"].to_numpy()
        else:
            self.scanner_int, self.target_names = map_col_to_int(self.labels[target_prop])

        sample_img = self[0]["img"]
        print("Shape of data:", sample_img.shape)  # [1,  193, 229, 193]
        assert (
            sample_img.dim() == self.spatial_dims + 1
        ), f"Expected {self.spatial_dims + 1} dimensions, got {sample_img.dim()}"

    def _init_subject_ids(self):
        self.subject_ids = [
            sub_fp.with_suffix("").stem.replace(f"-{self.mri_sequences[0]}", "")
            for sub_fp in self.subjects_fps
        ]

    def _id_from_fp(self, fp: Path) -> str:
        return fp.stem.split("-")[0]

    def _isin_labels(self, subject_id):
        return self.ixi_id_int(subject_id) in self.labels.index

    def _load_labels(self):
        csv_path = self.data_dir / "IXI.csv"
        labels = pd.read_csv(csv_path)

        # convert all columns to lower case
        labels.columns = labels.columns.str.lower()

        labels.rename(columns={"SEX_ID (1=m, 2=f)".lower(): "sex"}, inplace=True)
        # map numbers to f and m
        labels["sex"] = labels["sex"].map({1: "m", 2: "f"})

        # deduplicate
        # groupby index column
        labels = (
            labels.groupby("ixi_id", as_index=False)
            .filter(lambda x: x["dob"].nunique() == 1)
            .groupby("ixi_id")  # also sets as index
            .first()
        )
        # drop unnamed
        labels = labels.loc[:, ~labels.columns.str.contains("^unnamed")]
        # get scanners from subject ids
        scanners = {
            self.ixi_id_int(sub_id): self._id2scanner(sub_id) for sub_id in self.subject_ids
        }
        df_scanners = pd.DataFrame.from_dict(scanners, orient="index", columns=["scanner"])
        # add name to index column
        df_scanners.index.name = "ixi_id"

        labels_with_scanners = labels.merge(df_scanners, on="ixi_id", how="right")

        # fillnas for age, and sex
        labels_with_scanners["age"] = labels_with_scanners["age"].fillna(value=-1)
        labels_with_scanners["sex"] = labels_with_scanners["sex"].fillna(value="x")

        self.labels = labels_with_scanners

        # filter dataset for subjects with labels
        self.subjects_fps = [
            sub_fp for sub_fp in self.subjects_fps if self._isin_labels(self._id_from_fp(sub_fp))
        ]

        # filter labels for subject ids in dataset
        subject_ids_int = np.array(
            [self.ixi_id_int(self._id_from_fp(sub_fp)) for sub_fp in self.subjects_fps]
        )
        self.labels = self.labels.loc[self.labels.index.isin(subject_ids_int)]

    def ixi_id_int(self, ixi_id: str) -> int:
        """Get the integer part of the IXI ID."""
        return int(ixi_id.split("-")[0].replace("IXI", ""))

    def _load_subject_fps(self, data_dir: Path, spatial_dims: Literal[2, 3]):
        seq_str = self.mri_sequences[0]

        file_ext = "nii.gz"

        # subjects
        if self.use_biasfield_corrected:
            seq_str += "_biasfield_corrected"

        self.subjects_fps = sorted(
            list((data_dir / seq_str).glob(f"*{self.mri_sequences[0]}.{file_ext}"))
        )

        # filter out excluded ids
        self.subjects_fps = [
            sub_fp
            for sub_fp in self.subjects_fps
            if self._id_from_fp(sub_fp) not in self.exclude_ids
        ]
        # filter masks
        self.subjects_fps = [sub_fp for sub_fp in self.subjects_fps if "mask" not in sub_fp.stem]

        # filter by subset names
        self.subjects_fps = [
            sub_fp
            for sub_fp in self.subjects_fps
            if any(s in sub_fp.name for s in self.subset_names)
        ]

    def _id2scanner(self, subject_name: str) -> str:
        """Get the scanner name from the subject ID."""
        return subject_name.split("-")[1]

    def __len__(self):
        return len(self.subjects_fps)

    @property
    def mri_sequences(self) -> tuple[str, ...]:
        if self._mri_sequences is not None:
            return self._mri_sequences
        return ("T1",)

    @property
    def subset_names(self):
        all_subsets = ("Guys", "HH", "IOP")

        if not self.fit_sites:
            self.fit_sites = all_subsets
        if not self.test_sites:
            self.test_sites = all_subsets
        match self.split:
            case "train" | "val":
                return self.fit_sites
            case "test":
                return self.test_sites
            case _:
                raise ValueError(f"split {self.split} not supported")

    def loader(self, fp, slice_offset):

        nii_img = nib.nifti1.load(fp)
        # get the slices around the middle slice
        nii_img_slices = nii_img.slicer[
            ...,
            nii_img.shape[-1] // 2 + slice_offset : nii_img.shape[-1] // 2 + slice_offset + 1,
        ]
        # load and permute to be (slices, height, width)
        return torch.from_numpy(nii_img_slices.get_fdata()).float().permute(2, 0, 1)

    def get_slice_offset(self):
        if self.slices_around_middle > 0:
            # index around middle slice
            return torch.randint(
                low=-self.slices_around_middle,
                high=self.slices_around_middle,
                size=(1,),
                generator=self.slicer_rng,
            )
        return 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[Path]]:
        subject_fp = self.subjects_fps[index]
        subject_data = {}
        mask_fp = subject_fp.with_name(
            subject_fp.name.replace(f"{self.mri_sequences[0]}.", f"{self.mri_sequences[0]}_mask.")
        )
        slice_offset = self.get_slice_offset()
        brainmask = self.loader(mask_fp, slice_offset).bool()
        if self.load_anat_label_maps:
            anat_label_map = self.load_anat_label_map(subject_fp, slice_offset)

        for seq in self.mri_sequences:
            seq_fp = Path(str(subject_fp).replace(f"{self.mri_sequences[0]}", seq))

            if seq_fp.exists():
                img = self.loader(seq_fp, slice_offset)
                seq_data = {"img": img}

                seq_data["seg"] = brainmask
                seq_data["brainmask"] = brainmask
                seq_data: dict = self.preproc(seq_data)
                # augmentations
                if self.augmentations is not None:
                    seq_data = self.augmentations(seq_data)

                # crop data randomly
                seq_data = self.crop(seq_data)

                subject_data[seq] = seq_data["img"]
                subject_data["seg"] = seq_data["seg"]

        subject_data["img"] = subject_data[self.mri_sequences[0]]
        if self.load_anat_label_maps:
            subject_data["anat_label_map"] = anat_label_map

        patient_attrs = self.get_patient_attrs(index)

        subject_data.update(
            {
                "index": torch.tensor(index),
                "scanner_int": self.scanner_int[index],
                "subject_id": self.subject_ids[index],
                "fp": [str(subject_fp)],
            }
        )
        subject_data.update(patient_attrs)
        # rename key "T1" with "T1w"
        if "T1" in subject_data:
            subject_data["T1w"] = subject_data.pop("T1")
        if "T2" in subject_data:
            subject_data["T2w"] = subject_data.pop("T2")

        return subject_data

    def get_patient_attrs(self, index):
        patient_attrs = {
            "age": self.labels.loc[self.ixi_id_int(self.subject_ids[index]), "age"],
            "sex": self.labels.loc[self.ixi_id_int(self.subject_ids[index]), "sex"],
            "scanner": self.labels.loc[self.ixi_id_int(self.subject_ids[index]), "scanner"],
        }

        return patient_attrs

    def load_anat_label_map(self, subject_fp, slice_offset):
        anat_label_map_fp = subject_fp.with_name(
            subject_fp.name.replace(f"{self.mri_sequences[0]}.", f"{self.mri_sequences[0]}_seg.")
        )
        anat_label_map = self.loader(anat_label_map_fp, slice_offset)
        anat_label_map = self.crop({"img": anat_label_map[None], "seg": anat_label_map[None] > 0})[
            "img"
        ][0]

        return anat_label_map

    def fn2subject(self, fp: Path) -> str:
        """Split the path to the file and return the patient id"""
        return fp.stem.replace(f"-{self.mri_sequences[0]}", "")

    def _glob_subject_dirs(self) -> list[Path]:
        return self.subjects_fps
