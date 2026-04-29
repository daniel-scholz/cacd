"""Minimal inference entry point for CACD harmonization.

Loads a trained CACD checkpoint and harmonizes a tensor of source-site
slices toward a tensor of target-site slices. Inputs are expected to be
already preprocessed (registered to MNI152, intensity-normalized to
[-1, 1], cropped to the training image size). See ``preprocessing/`` for
the registration scripts and ``conf/<config>.json`` for the expected
``img_size``.

    uv run python infer.py \\
        --config ixi_guys-hh_2d_cacd_diffae_all_augs \\
        --checkpoint <cacd.ckpt> \\
        --source <source.pt> --target <target.pt> --output <out.pt>

``--source`` / ``--target`` are ``(N, 1, H, W)`` tensors in ``[-1, 1]``
(``.pt`` or ``.npy``). The output is the same shape, written as ``.pt``.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import torch

from diffae.experiment import LitModel
from diffae.templates import templates_dict


class BrainMRIHarmonizationModel(ABC):
    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor: ...


class DiffAEHarmonizationModel(BrainMRIHarmonizationModel):
    def __init__(
        self,
        model: LitModel,
        target_images: torch.Tensor,
        noise_steps: Optional[int] = None,
    ):
        super().__init__(model)
        self.model: LitModel
        self.T = self.model.conf.T_eval

        target_cond = self.model.encode_ema(target_images)["cond"]
        z_sem_target, _ = self.model.split_sem_id(target_cond)
        self.z_sem_target_mean = z_sem_target.mean(dim=0, keepdim=True)

        self.noise_steps = noise_steps or self.T

    def harmonize(self, source_images: torch.Tensor) -> torch.Tensor:
        source_cond = self.model.encode_ema(source_images)["cond"]
        source_xT = self.model.encode_stochastic_ema(
            source_images,
            source_cond,
            self.T,
            self.noise_steps,
            imgs=source_images if self.model.conf.in_channels == 2 else None,
        )

        z_sem_source, z_id_source = self.model.split_sem_id(source_cond)
        cond_harmonized = self.model.combine_sem_id(
            self.z_sem_target_mean.expand_as(z_sem_source), z_id_source
        )

        harmonized_images = self.model.render(
            source_xT,
            {"cond": cond_harmonized},
            self.T,
            T_offset=self.T - self.noise_steps,
            imgs=source_images if self.model.conf.in_channels == 2 else None,
        )

        harmonized_images[source_images == -1] = source_images[source_images == -1]
        return harmonized_images


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    return torch.load(path, map_location="cpu", weights_only=True).float()


@click.command()
@click.option("--config", required=True, help="Config name in conf/ (without .json)")
@click.option(
    "--checkpoint",
    "ckpt_fp",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--source", "source_fp", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--target", "target_fp", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--output", "out_fp", required=True, type=click.Path(path_type=Path))
@click.option("--template", default="scanner_harm", show_default=True)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
)
@click.option(
    "--t-eval",
    "t_eval",
    type=int,
    default=None,
    help="Override the number of diffusion sampling steps. The paper used 250.",
)
def main(
    config: str,
    ckpt_fp: Path,
    source_fp: Path,
    target_fp: Path,
    out_fp: Path,
    template: str,
    device: str,
    t_eval: int | None,
):
    conf = templates_dict[template](config)
    if t_eval is not None:
        conf.T_eval = t_eval

    dev = torch.device(device)
    model = LitModel.load_from_checkpoint(
        ckpt_fp, conf=conf, map_location=dev, strict=False
    )
    model.setup()
    model.eval()

    source = _load_tensor(source_fp).to(dev)
    target = _load_tensor(target_fp).to(dev)

    harmonizer = DiffAEHarmonizationModel(model=model, target_images=target)
    with torch.no_grad():
        harmonized = harmonizer.harmonize(source)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(harmonized.cpu(), out_fp)
    print(f"saved {harmonized.shape} → {out_fp}")


if __name__ == "__main__":
    main()
