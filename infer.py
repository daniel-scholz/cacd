"""Minimal inference entry point for CACD harmonization.

Loads a trained CACD checkpoint and harmonizes a tensor of source-site slices
toward a tensor of target-site slices. Both inputs are expected to be already
preprocessed (registered, intensity-normalized to [-1, 1], cropped to the
training image size) — see ``preprocessing/`` for the registration scripts and
``conf/<config>.json`` for the expected ``img_size``.

Usage:
    uv run python infer.py \
        --config oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs \
        --checkpoint <path-to-cacd.ckpt> \
        --source <source.pt> \
        --target <target.pt> \
        --output <out.pt> \
        --t-eval 250

``--source`` and ``--target`` accept ``.pt`` / ``.npy`` files holding tensors of
shape ``(N, 1, H, W)`` in the range ``[-1, 1]``. The output is a tensor of the
same shape, written to ``--output`` as ``.pt``.

For a full NIfTI → NIfTI pipeline (registration, slicing, reassembly), use
``eval_harm_baselines.py``.
"""

from pathlib import Path

import click
import numpy as np
import torch

from diffae.experiment import LitModel
from diffae.templates import templates_dict
from harm_model import DiffAEHarmonizationModel


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr).float()
    return torch.load(path, map_location="cpu", weights_only=True).float()


@click.command()
@click.option("--config", required=True, help="Config name in conf/ (without .json)")
@click.option(
    "--checkpoint", "ckpt_fp", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--source", "source_fp", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--target", "target_fp", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "out_fp", required=True, type=click.Path(path_type=Path))
@click.option("--template", default="scanner_harm", show_default=True)
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
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
    model = LitModel.load_from_checkpoint(ckpt_fp, conf=conf, map_location=dev, strict=False)
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
