"""Shared pytest fixtures for CACD tests."""

from __future__ import annotations

# Filters mirror pyproject.toml's [tool.pytest.ini_options].filterwarnings,
# applied here too because the lightning/SimpleITK warnings fire at import
# time before pytest's own filter parsing kicks in.
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Deprecated call to .pkg_resources\.declare_namespace",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type Swig.*has no __module__",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink.*has no __module__",
    category=DeprecationWarning,
)

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed Python/NumPy/PyTorch RNGs so tests are deterministic."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def fake_image_batch() -> torch.Tensor:
    """Small 2D image batch in [-1, 1]: shape (B=4, C=1, H=64, W=64)."""
    return torch.rand(4, 1, 64, 64) * 2.0 - 1.0
