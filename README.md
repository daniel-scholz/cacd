# CACD: Contrastive Anatomy-Contrast Disentanglement

Official implementation of **CACD**, an unsupervised, domain-general MRI
harmonization method. CACD disentangles anatomy and contrast by
training a conditional diffusion autoencoder with two contrastive heads
on anatomy-preserving augmentations, enabling cross-scanner
harmonization to a target domain — including unseen scanners — from a
single reference image and without retraining.

> Daniel Scholz, Ayhan Can Erdur, Robbie Holland, Viktoria Ehm,
> Jan C. Peeken, Benedikt Wiestler, Daniel Rueckert.
> **Contrastive Anatomy-Contrast Disentanglement: A Domain-General MRI
> Harmonization Method.** MICCAI 2025.
> [Paper (arXiv 2509.06592)](https://arxiv.org/abs/2509.06592) ·
> [Paper (MICCAI proceedings)](https://papers.miccai.org/miccai-2025/0176-Paper1897.html)

The method extends the Diffusion Autoencoder (DiffAE,
[Preechakul et al., CVPR 2022](https://diff-ae.github.io/)) with an
anatomy-contrast disentanglement module: the semantic encoder is split
into two heads, ``z_a`` (anatomy) and ``z_c`` (contrast). Two
supervised-contrastive losses pin one feature into each head, using
anatomy-preserving augmentations (random gamma, bias-field corruption,
GIN with up-down sampling) to define positive and negative views.

## Install

This repo is set up for [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                       # core training & light eval
uv sync --extra radiomics     # + Table 1 scanner classification (pyradiomics)
```

`pyradiomics` builds C extensions; if `uv` can't compile it on your
system, fall back to ``conda install -c radiomics pyradiomics``.

To run the HACA3 baseline, install it from source (it isn't on PyPI):

```bash
uv pip install git+https://github.com/lianruizuo/HACA3.git
```

Then place the pretrained weight at
``models/haca3/harmonization_public.pt`` (override with
``$HACA3_MODEL_PATH``).

## Datasets

We train on a mix of [OASIS3](https://www.oasis-brains.org/) and
[IXI](https://brain-development.org/ixi-dataset/) (Guy's & HH only). We
evaluate on the held-out IXI splits + IOP, on traveling subjects from
[OpenNeuro Harmony Phase A](https://openneuro.org/datasets/ds004215),
and via a radiomics-based scanner classifier on IXI. All scans are
affine-registered to MNI152 1×1×1 mm³ (using
[niftyreg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)),
skull-stripped with [HD-BET](https://github.com/MIC-DKFZ/HD-BET), and
N4-corrected. Splits are in ``dataset/<dataset>/<site>/{train,val,test}.txt``.

The location of the preprocessed volumes is configured by environment
variable; defaults are in ``diffae/config.py:data_paths``:

```bash
export DATASETS_DIR=~/datasets        # contains ixi_reg_skullstrip/, OASIS/, ...
```

## Train

The paper run is reproduced from
``conf/oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs.json``:

```bash
# Single-GPU local run
uv run run.py scanner_harm --conf oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs

# Single-GPU SLURM
sbatch scripts/slurm_train.sh scanner_harm --conf oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs

# Multi-GPU SLURM (override --gres and --tasks-per-node together)
sbatch --gres=gpu:2 --tasks-per-node=2 scripts/slurm_train.sh \
    scanner_harm --conf oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs

# Fast dev run (1 batch, no wandb)
uv run run.py scanner_harm --conf oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs --fast_dev_run
```

Available paper-relevant configs:

| Config | Role |
| --- | --- |
| ``oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs`` | Paper "Ours" (wandb ``1voovf9c``) |
| ``oasis3_ixi_guys-hh_2d_vanilla_diffae`` | Vanilla DiffAE baseline (wandb ``72jaa0rm``) |
| ``oasis3_ixi_guys-hh_2d_cacd_diffae_gamma`` | Gamma-only ablation |
| ``oasis3_ixi_guys-hh_2d_cacd_diffae_gamma_rc`` | Gamma + RandConv ablation |

## Inference

For end-to-end harmonization on already-preprocessed slice tensors, use
``infer.py``:

```bash
uv run python infer.py \
    --config oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs \
    --checkpoint <path/to/cacd.ckpt> \
    --source <source.pt> \
    --target <target.pt> \
    --output <out.pt>
```

``--source`` / ``--target`` are tensors of shape ``(N, 1, H, W)`` in ``[-1, 1]``
(``.pt`` or ``.npy``). For full NIfTI → NIfTI inference (registration, slicing,
reassembly), see the eval pipeline below.

Pretrained CACD checkpoints are not yet publicly distributed — the OASIS-3
Data Use Agreement requires resolving redistribution terms first. Until then,
reproduce the paper run via the training recipe above, or contact the authors.

## Evaluate

```bash
# Latent analysis + paired analysis + latent swap on the trained CACD model
uv run eval.py --wandb_id <wandb_id> --target_site Guys

# Traveling-subjects comparison vs. baselines (Figure 2)
uv run eval_harm_baselines.py --method DiffAE --wandb_id <wandb_id> --target_scanner GEM
uv run eval_harm_baselines.py --method HACA3 --target_scanner GEM
uv run eval_harm_baselines.py --method histogram_matching --target_scanner GEM
uv run eval_harm_baselines.py --method unharmonized --target_scanner GEM

# Scanner classification + age regression (Table 1) — needs pyradiomics
uv run eval_scanner_classification.py
```

## Tests

```bash
uv run pytest tests/                                            # full suite
uv run pytest tests/unit                                        # unit only
uv run pytest tests/integration -m "not slow and not needs_weights"
```

Tests are split into:

- ``tests/unit`` — pure-CPU unit tests for losses, augmentations,
  config loading, diffusion schedules, and the z_sem / z_id split.
- ``tests/integration`` — smoke tests that exercise import paths and
  the lightweight harmonization baselines. Heavy paths are gated by
  ``@pytest.mark.slow`` (full ``LitModel`` instantiation),
  ``@pytest.mark.needs_weights`` (HACA3, DiffAE, CycleGAN), and
  ``@pytest.mark.needs_data`` (real datasets).


## Repository layout

```
diffae/                   # model + diffusion + data + experiment
  model/diffae.py             # base DiffAE (vanilla)
  model/diffae_id_preserve.py # CACD: split z into z_a, z_c
  model/encoders/             # encoder variants (with/without split, separate enc.)
  model/augmentations/        # GIN/RC, bias field, gamma, b-spline
  diffusion/                  # spaced DDIM/DDPM, beta schedules, contrastive losses
  data/                       # IXI / OASIS3 / glioma / WMH dataset wrappers
  experiment.py               # LightningModule
  templates.py                # config templates per task
  config.py                   # TrainConfig, JSON-overlay loader
conf/                     # JSON config overrides per experiment
eval/                     # paired analysis, latent edit/infer, dim reduction
harm_model.py             # BrainMRIHarmonizationModel + concrete subclasses
infer.py                  # minimal harmonization entry point
eval.py                   # main eval (latent analysis, swap, paired)
eval_harm_baselines.py    # cross-method comparison
eval_scanner_classification.py  # radiomics-based scanner classifier (Table 1)
scripts/slurm_train.sh    # SLURM training template (override --gres for multi-GPU)
tests/                    # unit + integration tests
```

## Citation

```bibtex
@inproceedings{scholz2025cacd,
  title     = {Contrastive Anatomy-Contrast Disentanglement: A Domain-General
               {MRI} Harmonization Method},
  author    = {Scholz, Daniel and Erdur, Ayhan Can and Holland, Robbie and
               Ehm, Viktoria and Peeken, Jan C. and Wiestler, Benedikt and
               Rueckert, Daniel},
  booktitle = {Medical Image Computing and Computer Assisted Intervention --
               MICCAI 2025},
  year      = {2025},
}
```

The base diffusion autoencoder we build on is from:

```bibtex
@inproceedings{preechakul2022diffae,
  title     = {Diffusion Autoencoders: Toward a Meaningful and Decodable Representation},
  author    = {Preechakul, Konpat and Chatthee, Nattanat and
               Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
}
```

## License

MIT — see ``LICENSE``.
