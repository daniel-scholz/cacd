"""Unit tests for the z_sem / z_id disentanglement plumbing.

The CACD encoder produces a single ``cond`` vector of shape
``(B, z_sem_dim + z_id_dim)`` by concatenating two heads:

    cond = cat([z_sem, z_id], dim=1)            # in SemIDLayer.forward
    z_sem, z_id = split(cond, [z_sem_dim, z_id_dim], dim=1)   # in diffusion/base.py

If the cat-and-split contract drifts (e.g. order swapped, dim wrong) the
contrastive losses train on the wrong slice and silently learn nothing
meaningful. These tests pin the contract.

Building the full ``DiffAEIDModel`` requires a UNet+encoder stack that
costs a large amount of compute even at the smallest size, so we exercise
the layer in isolation and the validation logic on the config.
"""

from __future__ import annotations

import pytest
import torch

from diffae.model.encoders.out_layers import SemIDLayer


def test_sem_id_layer_concat_order() -> None:
    """``cond[:, :z_sem_dim]`` must equal the SemIDLayer's z_sem head output,
    and ``cond[:, z_sem_dim:]`` must equal its z_id head output.

    Otherwise ``torch.split(cond, [z_sem_dim, z_id_dim], dim=1)`` recovers
    the wrong tensors downstream.
    """
    z_sem_dim, z_id_dim = 16, 8
    layer = SemIDLayer(
        z_sem_dim=z_sem_dim,
        z_id_dim=z_id_dim,
        dims=2,
        in_channels=4,
        kernel_size=1,
    )
    x = torch.randn(2, 4, 1, 1)
    cond = layer(x)
    assert cond.shape == (2, z_sem_dim + z_id_dim)

    expected_z_sem = layer.sem(x).flatten(start_dim=1)
    expected_z_id = layer.id(x).flatten(start_dim=1)

    assert torch.allclose(cond[:, :z_sem_dim], expected_z_sem)
    assert torch.allclose(cond[:, z_sem_dim:], expected_z_id)


def test_split_recovers_heads() -> None:
    """``torch.split`` with ``[z_sem_dim, z_id_dim]`` recovers the heads."""
    z_sem_dim, z_id_dim = 16, 8
    layer = SemIDLayer(
        z_sem_dim=z_sem_dim,
        z_id_dim=z_id_dim,
        dims=2,
        in_channels=4,
        kernel_size=1,
    )
    x = torch.randn(3, 4, 1, 1)
    cond = layer(x)
    z_sem, z_id = torch.split(cond, [z_sem_dim, z_id_dim], dim=1)
    assert z_sem.shape == (3, z_sem_dim)
    assert z_id.shape == (3, z_id_dim)
    assert torch.allclose(z_sem, layer.sem(x).flatten(start_dim=1))
    assert torch.allclose(z_id, layer.id(x).flatten(start_dim=1))


def test_diffae_id_config_dim_invariant() -> None:
    """``DiffAEIDModel`` must reject configs where
    ``enc_z_sem_dim + enc_z_id_dim != cond_channels``.

    We only construct the config and call its validator path via
    instantiation up to the check (which is the very first line of
    ``DiffAEIDModel.__init__`` after ``super().__init__``). To avoid
    paying the cost of the full UNet init, we monkey-patch the parent
    ``__init__`` to a no-op for this assertion test.
    """
    from diffae.model import diffae_id_preserve as mod

    orig_init = mod.DiffAEModel.__init__
    mod.DiffAEModel.__init__ = lambda self, conf: None  # type: ignore
    try:
        cfg = object.__new__(mod.DiffAEIDConfig)
        cfg.enc_z_sem_dim = 200
        cfg.enc_z_id_dim = 100
        cfg.cond_channels = 256  # intentionally wrong: 300 != 256
        # Other fields are accessed only via attributes that the unit
        # under test (the dim check + intensity-aug check) reads.
        cfg.intensity_augs_names = ("gamma",)
        with pytest.raises(ValueError, match="z_id_dim and z_sem_dim"):
            mod.DiffAEIDModel(cfg)  # type: ignore[arg-type]
    finally:
        mod.DiffAEModel.__init__ = orig_init


def test_diffae_id_config_requires_intensity_aug() -> None:
    """Empty ``intensity_augs_names`` is illegal for the disentanglement model."""
    from diffae.model import diffae_id_preserve as mod

    orig_init = mod.DiffAEModel.__init__
    mod.DiffAEModel.__init__ = lambda self, conf: None  # type: ignore
    try:
        cfg = object.__new__(mod.DiffAEIDConfig)
        cfg.enc_z_sem_dim = 128
        cfg.enc_z_id_dim = 128
        cfg.cond_channels = 256
        cfg.intensity_augs_names = ()
        # We need to feed an attribute that the parent class would have
        # set: self.intensity_augs_names. Patch parent init to set it.
        mod.DiffAEModel.__init__ = (  # type: ignore
            lambda self, conf: setattr(self, "intensity_augs_names", conf.intensity_augs_names)
        )
        with pytest.raises(ValueError, match="At least one intensity augmentation"):
            mod.DiffAEIDModel(cfg)  # type: ignore[arg-type]
    finally:
        mod.DiffAEModel.__init__ = orig_init
