from pathlib import Path
from typing import Literal, Optional

from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CenterSpatialCropd
from torch.utils.data import Dataset

from diffae.transforms import ScaleIntensityRangePercentilesForegroundd


class BaseMRIDataset(Dataset):
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
        self.subject_dirs = sorted(self.subject_dirs, key=lambda x: x.name)

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

    def _init_crop(self) -> None:
        self.crop = CenterSpatialCropd(
            keys=["img", "seg", "brainmask"],
            roi_size=self.img_size,
            allow_missing_keys=True,
        )

    @property
    def mri_sequences(self) -> tuple[str, ...]:
        return self._mri_sequences or ()

    def reshape_dict(self, x):
        return {k: v.view(-1, *v.shape[-self.spatial_dims :]) for k, v in x.items()}

    def _glob_subject_dirs(self) -> list[Path]:
        raise NotImplementedError
