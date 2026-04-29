# CACD

Code for *Contrastive Anatomy-Contrast Disentanglement: A Domain-General
MRI Harmonization Method*, MICCAI 2025
([arXiv](https://arxiv.org/abs/2509.06592)).

CACD splits a diffusion-autoencoder latent into anatomy (`z_a`) and
contrast (`z_c`) with two contrastive losses, so a trained model can
harmonize a source scan to any target scanner from one reference image.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip in a Python 3.12 venv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

The shipped config trains on IXI (Guy's, HH; held-out IOP). Affine-register
to MNI152 1mm, skull-strip with HD-BET, N4-correct. Splits live in
`dataset/ixi/<site>/{train,val,test}.txt`. Point `DATASETS_DIR` (or edit
`diffae/config.py:data_paths`) at the preprocessed root. To add another
dataset (e.g. OASIS-3, as in the paper), register it in `data_paths` and
add a loader following `diffae/data/ixi.py`.

For paired evaluation on the On-Harmony multi-scanner dataset, see
[`docs/ON-HARMONY.MD`](docs/ON-HARMONY.MD).

## Train

```bash
uv run run.py scanner_harm --conf ixi_guys-hh_2d_cacd_diffae_all_augs
```

(Replace `uv run` with `python` if using a plain pip-installed venv.)

## Inference

```bash
uv run python infer.py \
    --config ixi_guys-hh_2d_cacd_diffae_all_augs \
    --checkpoint <cacd.ckpt> \
    --source <source.pt> --target <target.pt> --output <out.pt>
```

`source` and `target` are `(N, 1, H, W)` tensors in `[-1, 1]`.

Pretrained weights cannot be redistributed: the paper checkpoints were
trained on OASIS-3, whose data use agreement does not permit sharing
derived model weights publicly.

## Citation

```bibtex
@inproceedings{scholz2025contrastive,
  title={ {Contrastive Anatomy-Contrast Disentanglement: A Domain-General MRI Harmonization Method} },
  author={Scholz, Daniel and Erdur, Ayhan Can and Holland, Robbie and Ehm, Viktoria and Peeken, Jan C and Wiestler, Benedikt and Rueckert, Daniel},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={100--110},
  year={2025},
  organization={Springer}
}
```

Built on the diffusion autoencoder of
[Preechakul et al., CVPR 2022](https://diff-ae.github.io/).

## License

MIT — see `LICENSE`.
