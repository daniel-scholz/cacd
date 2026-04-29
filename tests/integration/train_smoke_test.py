"""Integration smoke test: instantiate ``LitModel`` for the paper config.

Constructing ``LitModel`` exercises the full template → ``TrainConfig`` →
``make_model_conf`` → UNet/DiffAE/encoder pipeline. This is the path that
``run.py`` walks before training starts, so a regression here blocks any
training run end-to-end.

The model is ~24M params; this test is marked ``slow`` so default CI runs
can opt out.
"""

from __future__ import annotations

import pytest

from diffae.experiment import LitModel
from diffae.templates import scanner_harm

PAPER_CONFIGS = [
    "oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs",
    "oasis3_ixi_guys-hh_2d_vanilla_diffae",
]


@pytest.mark.slow
@pytest.mark.parametrize("conf_name", PAPER_CONFIGS)
def test_litmodel_instantiates(conf_name: str) -> None:
    conf = scanner_harm(conf_name)
    model = LitModel(conf)
    # Both train and EMA models must be present and parameterised.
    assert sum(p.numel() for p in model.model.parameters()) > 0
    assert sum(p.numel() for p in model.ema_model.parameters()) > 0
    # Samplers built without error.
    assert model.sampler is not None
    assert model.eval_sampler is not None
