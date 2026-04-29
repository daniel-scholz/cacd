from pathlib import Path
from typing import Literal, Optional, TypedDict

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data.dataset import CacheDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CenterSpatialCropd
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utility.dictionary import EnsureTyped, Lambdad

from diffae.transforms import ScaleIntensityRangePercentilesForegroundd


class LoadNiftiSlices(MapTransform):
    def __init__(self, offset: int, keys: list[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.offset = offset + 1
        self.data_key = keys[0]

    def __call__(self, data: dict):
        offset = self.offset
        fp = data["fp"]
        slices = self.loader(fp, offset)
        data[self.data_key] = slices
        # convert fp to string
        return data

    def loader(self, fp, slice_offset):

        nii_img = nib.nifti1.load(fp)
        # get the slices around the middle slice
        nii_img_slices = nii_img.slicer[
            ...,
            nii_img.shape[-1] // 2 - slice_offset : nii_img.shape[-1] // 2 + slice_offset + 1,
        ]
        # load and permute to be (slices, height, width)
        return torch.from_numpy(nii_img_slices.get_fdata()).float().permute(2, 0, 1)


class RandomSlice(MapTransform, Randomizable):
    def __init__(self, keys: list[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: dict):
        idx = torch.randint(0, data[self.keys[0]].shape[0], (1,))
        for key in self.keys:
            data[key] = data[key][idx]
        return data


class LoadBrainMask(LoadNiftiSlices):
    def __init__(self, offset: int, keys: list[str], allow_missing_keys: bool = False):
        super().__init__(offset, keys, allow_missing_keys)

    def __call__(self, data):
        data["fp"] = data["fp"].parent / data["fp"].name.replace("-T1", "-T1_mask")
        return super().__call__(data)


class CacheDatasetParams(TypedDict):
    copy_cache: bool
    cache_rate: float
    progress: bool
    num_workers: int
    runtime_cache: bool


def load_file_names(
    root_dir: Path,
    dataset_name: Literal["ixi", "oasis3"],
    stage: Literal["train", "val", "test"],
    subset_filters: Optional[list[Literal["IOP", "Guys", "HH"]]] = None,
) -> list[Path]:

    # replace "reg" with "reg_affine" in root_dir
    root_dir = Path(str(root_dir).replace("reg", "reg_affine"))

    if subset_filters is None:
        fps_with_filenames = [Path("dataset") / dataset_name / f"{stage}.txt"]
    else:
        fps_with_filenames = [
            Path("dataset") / dataset_name / subset / f"{stage}.txt" for subset in subset_filters
        ]
    subject_ids = []
    for fp in fps_with_filenames:
        with open(fp, "r", encoding="utf-8") as f:
            subject_ids.extend(f.read().splitlines())

    # strip
    subject_ids = [subject_id.strip() for subject_id in subject_ids]

    if dataset_name == "ixi":
        data_dir = root_dir / "T1_biasfield_corrected"

        subjects_fps = sorted([data_dir / f"{subject_id}-T1.nii.gz" for subject_id in subject_ids])
    elif dataset_name == "oasis3":
        data_dir = root_dir.with_name(root_dir.name + "_bfcorrected")

        subjects_fps = sorted(
            [
                next(data_dir.joinpath(subject_id).glob("anat*")).glob(f"{subject_id}_T1w.nii.gz")
                for subject_id in subject_ids
            ]
        )

    return subjects_fps


def oasis_subject_id_to_dir_name(subject_id: str) -> str: ...


def IXIDatasetCached(root_dir: Path, stage: Literal["train", "val", "test"], *args, **kwargs):
    dataset_name = "ixi"
    subset_filters = ["Guys", "HH"] if stage != "test" else ["IOP"]

    file_names = load_file_names(
        root_dir=root_dir,
        dataset_name=dataset_name,
        stage=stage,
        subset_filters=subset_filters,
    )

    scanners = [fn.name.split("-")[1] for fn in file_names]

    return DatasetCached(file_names=file_names, scanners=scanners, **kwargs)


def OASIS3DatasetCached(root_dir: Path, stage: Literal["train", "val", "test"], **kwargs):
    dataset_name = "oasis3"

    file_names = load_file_names(
        root_dir=root_dir, dataset_name=dataset_name, stage=stage, subset_filters=None
    )

    scanners = np.array(["oasis3"]).repeat(len(file_names))

    return DatasetCached(file_names=file_names, scanners=scanners, **kwargs)


def DatasetCached(
    file_names: list[Path],
    scanners: list[str],
    img_size: tuple[int, ...],
    offset: int,
    num_workers: int = 15,
    # fast_dev_run: bool,
):
    device = torch.device("cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() else device

    cache_dataset_params: CacheDatasetParams = {
        "cache_rate": 1.0,  # if not fast_dev_run else 0.0,
        "progress": True,
        "num_workers": num_workers,
        "runtime_cache": False,  # "threads" if device != "cpu" else True,
    }
    crop = CenterSpatialCropd(
        keys=["img", "brainmask"],
        roi_size=img_size,
        allow_missing_keys=True,
    )
    scaling = ScaleIntensityRangePercentilesForegroundd(
        keys=["img", "brainmask"],
        channel_wise=True,
        lower=0.005,
        upper=0.995,
        b_min=-1,
        b_max=1,
        clip=True,
    )
    data = [{"fp": fp, "scanner": np.array(scanner)} for fp, scanner in zip(file_names, scanners)]

    load_transform = Compose(
        [
            LoadNiftiSlices(offset=offset, keys=["img"]),
            LoadBrainMask(offset=offset, keys=["brainmask"]),
            EnsureTyped(keys=["img"], dtype=torch.float32, device=device),
            EnsureTyped(keys=["brainmask"], dtype=torch.bool, device=device),
            # make collatable
            Lambdad(keys=["fp", "scanner"], func=lambda x: [str(x)]),
            crop,
            scaling,
            RandomSlice(keys=["img", "brainmask"]),
        ]
    )

    dataset = CacheDataset(
        data=data,
        transform=load_transform,
        **cache_dataset_params,
    )

    print(f"Loaded {len(dataset)} samples")

    return dataset
