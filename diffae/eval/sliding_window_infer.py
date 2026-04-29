from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import torch
from monai.inferers import SlidingWindowSplitter
from tqdm import tqdm

from diffae.vis_utils import vis_3d


def grouper(iterable, n, *, incomplete="fill", fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks."
    # grouper('ABCDEFG', 3, fillvalue='x') → ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') → ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') → ABC DEF
    iterators = [iter(iterable)] * n
    match incomplete:
        case "fill":
            return zip_longest(*iterators, fillvalue=fillvalue)
        case "strict":
            return zip(*iterators, strict=True)
        case "ignore":
            return zip(*iterators)
        case _:
            raise ValueError("Expected fill, strict, or ignore")


class MySlidingWindowInferer:
    def __init__(
        self,
        patch_size: tuple[int, ...],
        overlap: float,
        padding_mode: str,
        sw_batch_size: int,
        aggregate: Optional[Literal["mean", "max"]] = "mean",
        save_dir: Optional[Path] = None,
        progress: bool = False,
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding_mode = padding_mode
        self.aggregate = aggregate
        self.save_dir = save_dir
        self.pbar = tqdm if progress else lambda x, *args, **kwargs: x
        self.splitter = SlidingWindowSplitter(
            patch_size=patch_size,
            overlap=overlap,
            pad_mode=padding_mode,
        )
        self.sw_batch_size = sw_batch_size

    def __call__(
        self,
        img: torch.Tensor,
        model: Callable[[torch.Tensor], torch.Tensor],
        do_vis: bool = False,
        **nn_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Sequence[Sequence[int]]]:
        patch_gen = self.splitter(img)
        patch_gen = grouper(patch_gen, self.sw_batch_size, incomplete="fill", fillvalue=None)

        n_patches = np.prod(
            [
                np.ceil((i / (s * (1.0 - self.splitter.overlap)))).astype(np.int64)
                for i, s in zip(img.shape[-3:], self.splitter.patch_size[-3:])
            ]
        )

        # infer on patches
        outputs = []
        locations = []
        for i_p, batch in enumerate(
            self.pbar(
                patch_gen,
                total=np.ceil(n_patches / self.sw_batch_size).astype(np.int64),
                desc="Infering patches",
            )
        ):
            batch_patch, loc = zip(*(b_ for b_ in batch if b_ is not None))

            # stack tensors to [B x BP x [...]]
            batch_patch = torch.stack(batch_patch, dim=1)
            sw_batch_size = batch_patch.size(1)

            # flatten batch and batch-patch dim
            batch_patch = batch_patch.flatten(0, 1)

            if self.save_dir is not None and do_vis:
                if i_p == 0:
                    print(f"Saving vis patches to {self.save_dir}")
                for i_s, sample in enumerate(batch_patch):
                    if (sample > -1).any():
                        # save patches
                        sample_fp = self.save_dir / f"patch_{i_p}_{i_s}.png"
                        if not sample_fp.exists():
                            vis_3d(sample, sample_fp)

            with torch.no_grad():
                i_p_batch = i_p * self.sw_batch_size
                patch_kwargs = {
                    k: nn_kwargs[k][i_p_batch : i_p_batch + self.sw_batch_size] for k in nn_kwargs
                }
                # stack to batch dim if not a tensor
                for k, v in patch_kwargs.items():
                    if not isinstance(v, torch.Tensor):
                        patch_kwargs[k] = torch.cat(v, dim=0)

                out = model(batch_patch, **patch_kwargs)
            # reshape to [B x BP x [...]]
            out = out.view([img.size(0), sw_batch_size] + list(out.shape[1:]))
            outputs.append(out)
            locations.extend(loc)

        # concatenate outputs, list of [B x BP x [...]] -> [B x P x [...]],
        # BP: patch-batch size, P: total number of patches
        outputs = torch.cat(outputs, dim=1)

        match self.aggregate:
            case "mean":
                return outputs.mean(dim=1)
            case "max":
                return outputs.max(dim=1).values
            case None:
                # With no aggregation, return output as B x P x [...]] tensor,
                # B batch size, P number of patches
                return outputs, locations
            case _:
                raise ValueError(f"aggregate {self.aggregate} not supported")
