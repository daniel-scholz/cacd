"""Integration smoke tests for the harmonization baselines in ``harm_model.py``.

The lightweight baselines (``HistogramMatchingModel``, ``UnharmonizeModel``)
have no external weights and must work end-to-end on a tiny synthetic
batch. These guarantee that import paths, tensor shapes, and dtype
conventions across the abstract base class are intact.

The heavy baselines (DiffAE, HACA3, CycleGAN) require pretrained weights
and live data, so they are gated behind ``@pytest.mark.needs_weights`` —
run them locally on a workstation with weights mounted, not in CI.
"""

from __future__ import annotations

import pytest
import torch

from harm_model import BrainMRIHarmonizationModel, HistogramMatchingModel, UnharmonizeModel


def _fake_brain_batch(b: int = 4, h: int = 64, w: int = 64) -> torch.Tensor:
    """A (B, 1, H, W) tensor in [-1, 1] mimicking a normalised T1w slice."""
    return torch.rand(b, 1, h, w) * 2.0 - 1.0


def test_harm_model_abc_contract() -> None:
    """``BrainMRIHarmonizationModel`` requires the ``harmonize`` method."""
    assert hasattr(BrainMRIHarmonizationModel, "harmonize")


def test_unharmonize_returns_input_unchanged() -> None:
    src = _fake_brain_batch()
    model = UnharmonizeModel()
    out = model.harmonize(src)
    assert out.shape == src.shape
    assert torch.equal(out, src)


def test_histogram_matching_preserves_shape_and_finite() -> None:
    src = _fake_brain_batch()
    target = _fake_brain_batch()
    model = HistogramMatchingModel(target_images=target)
    out = model.harmonize(src)
    assert out.shape == src.shape
    assert torch.isfinite(out).all()


def test_histogram_matching_changes_intensity_distribution() -> None:
    """Histogram-matched output should be closer in mean to the target than
    the source is — otherwise the matching is a no-op and we'd not catch
    regressions in scikit-image's API."""
    torch.manual_seed(0)
    src = torch.rand(1, 1, 64, 64) * 0.4 + 0.1  # bright-ish
    target = torch.rand(1, 1, 64, 64) * 0.2 - 0.5  # dark-ish
    model = HistogramMatchingModel(target_images=target)
    out = model.harmonize(src)
    src_target_gap = (src.mean() - target.mean()).abs()
    out_target_gap = (out.mean() - target.mean()).abs()
    assert out_target_gap < src_target_gap


@pytest.mark.needs_weights
def test_haca3_baseline_loads() -> None:
    """Smoke-test for the HACA3 wrapper. Skipped unless HACA3 is installed
    and the pretrained weight is on disk at the expected location."""
    haca3 = pytest.importorskip("haca3")  # noqa: F841
    import os
    from pathlib import Path

    weights = Path(os.environ.get("HACA3_MODEL_PATH", "models/haca3/harmonization_public.pt"))
    if not weights.exists():
        pytest.skip(f"HACA3 weights missing at {weights}")

    from harm_model import HACA3HarmonizationModel

    target = _fake_brain_batch(b=2, h=224, w=224)
    model = HACA3HarmonizationModel(target_images=target, pretrained=True)
    src = _fake_brain_batch(b=1, h=224, w=224)
    out = model.harmonize(src)
    assert out.shape[-2:] == src.shape[-2:]
