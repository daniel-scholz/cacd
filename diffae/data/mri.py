from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class MRIImageDataset(Dataset, ABC):
    # mri sequences
    @property
    @abstractmethod
    def mri_sequences(self) -> tuple[str, ...]:
        pass

    # subset names
    @property
    @abstractmethod
    def subset_names(self) -> tuple[str, ...]:
        pass
