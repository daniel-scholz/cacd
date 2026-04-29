from typing import Iterator

import torch
from torch.utils.data.sampler import Sampler


class UniformStratifiedSampler(Sampler[int]):
    def __init__(self, batch_size: int, labels: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if labels.ndim == 2 and labels.size(1) > 1:
            # map labels to int => also works for soft labels
            labels = torch.argmax(labels, dim=1)
        else:
            labels = labels.flatten().long()  # map to int to be used as index

        self.labels = labels
        self.strata: list[int] = self.labels.unique().tolist()
        # hallo
        self.strata_samples: list[torch.IntTensor] = []
        for stratum in self.strata:
            # get indices of samples in the stratum
            stratum_samples = torch.where(self.labels == stratum)[0]

            # ensure that the permutation is not the same as the previous one by adding the last index
            self.strata_samples.append(stratum_samples)

        self.n_per_stratum = self._get_n_per_stratum(batch_size)

    def __iter__(self):
        # create iters for each stratum
        strata_samples_iters = [iter(s.tolist()) for s in self.strata_samples]

        while True:
            # sample from each stratum
            for stratum in self.strata:
                # sample from the stratum n_per_stratum times
                for _ in range(self.n_per_stratum[stratum]):
                    try:
                        # get next in line
                        stratum_sample = next(strata_samples_iters[stratum])
                        yield stratum_sample
                    except StopIteration:
                        # if the iterator is exhausted, reinitialize it
                        strata_samples_iters[stratum] = self._init_stratum_samples_iter(stratum)

    def _init_stratum_samples_iter(self, stratum: int) -> Iterator[int]:
        # get current samples
        cur_samples = self.strata_samples[stratum]

        # permute samples (every day I'm shufflin')
        cur_samples = cur_samples[torch.randperm(len(cur_samples))]
        return iter(cur_samples.tolist())

    def _get_n_per_stratum(self, batch_size: int) -> dict[int, int]:
        # return a dict with the number of samples to take from each stratum in each batch
        n_strata = len(self.strata)
        return {s: max(1, batch_size // n_strata) for s in self.strata}

    def __len__(self):
        return len(self.labels)
