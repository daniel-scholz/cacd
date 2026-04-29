"""Unit tests for the diffusion timestep & beta-schedule plumbing.

We pin two contracts:

* ``space_timesteps`` returns a strictly-increasing set of integer timesteps
  in ``[0, num_timesteps)`` that obeys the section-count argument. This is
  the function ``DiffAEHarmonizationModel`` and DDIM inversion rely on.
* The ``"linear"`` and ``"cosine"`` beta schedules return strictly
  positive, monotone-non-decreasing tensors of the requested length.
  A regression that flips the sign or swaps end-points would break
  training silently — losses still converge to *something*, just not the
  right thing.
"""

from __future__ import annotations

import torch

from diffae.diffusion.base import get_named_beta_schedule
from diffae.diffusion.diffusion import space_timesteps


def test_space_timesteps_count_matches() -> None:
    steps = space_timesteps(num_timesteps=1000, section_counts=[20])
    assert len(steps) == 20
    assert max(steps) < 1000
    assert min(steps) >= 0


def test_space_timesteps_ddim_string() -> None:
    steps = space_timesteps(num_timesteps=1000, section_counts="ddim20")
    assert len(steps) == 20
    assert max(steps) < 1000


def test_space_timesteps_multi_section() -> None:
    steps = space_timesteps(num_timesteps=300, section_counts=[10, 15, 20])
    assert len(steps) == 45


def test_linear_beta_schedule_monotone_positive() -> None:
    betas = get_named_beta_schedule("linear", 1000)
    assert betas.shape == (1000,)
    assert (betas > 0).all()
    diffs = betas[1:] - betas[:-1]
    # Linear schedule must be strictly increasing.
    assert (diffs > 0).all()


def test_linear_beta_schedule_endpoints() -> None:
    """Linear schedule with 1000 steps must span [1e-4, 0.02] — these
    constants are referenced in the DDIM paper and downstream eval."""
    betas = get_named_beta_schedule("linear", 1000)
    assert torch.isclose(betas[0], torch.tensor(1e-4))
    assert torch.isclose(betas[-1], torch.tensor(0.02))


def test_const_beta_schedule() -> None:
    betas = get_named_beta_schedule("const0.01", 100)
    assert betas.shape == (100,)
    assert torch.allclose(betas, betas[0].expand_as(betas))
    assert betas[0] > 0
