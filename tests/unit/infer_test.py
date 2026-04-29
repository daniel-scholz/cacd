"""Unit tests for ``infer.py`` — the public-facing harmonization CLI.

Covers the bits we can exercise without weights or a GPU:

- ``_load_tensor`` round-trips for ``.pt`` and ``.npy``.
- The click contract (required flags appear in ``--help``).
- The release pre-flight invariant from ``docs/release_strip.md``: the
  paper config must overlay onto the ``scanner_harm`` template such
  that ``img_size == (192, 224)``, ``net_enc_z_sem_dim == 256``,
  ``slices_around_middle == 10``. If the JSON overlay silently stops
  landing, ``infer.py`` would load the wrong architecture.

The full end-to-end run (real ckpt + real On-Harmony slices) is in
``tests/integration/infer_test.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

import infer
from diffae.templates import templates_dict


def test_load_tensor_pt_roundtrip(tmp_path: Path) -> None:
    src = torch.randn(2, 1, 16, 16)
    fp = tmp_path / "x.pt"
    torch.save(src, fp)
    out = infer._load_tensor(fp)
    assert out.shape == src.shape
    assert out.dtype == torch.float32
    assert torch.equal(out, src.float())


def test_load_tensor_npy_roundtrip(tmp_path: Path) -> None:
    arr = np.random.randn(2, 1, 16, 16).astype(np.float64)  # non-float32 on disk
    fp = tmp_path / "x.npy"
    np.save(fp, arr)
    out = infer._load_tensor(fp)
    assert out.shape == arr.shape
    # ``_load_tensor`` must coerce to float32 regardless of the on-disk dtype.
    assert out.dtype == torch.float32
    assert np.allclose(out.numpy(), arr.astype(np.float32))


def test_cli_help_lists_required_flags() -> None:
    """Catch silent breakage of the click signature documented in the
    module docstring (``--config / --checkpoint / --source / --target /
    --output``)."""
    runner = CliRunner()
    result = runner.invoke(infer.main, ["--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--config", "--checkpoint", "--source", "--target", "--output"):
        assert flag in result.output, f"{flag} missing from --help"


def test_paper_config_overlay_lands() -> None:
    """Pre-flight check from ``docs/release_strip.md`` step 3.

    Loading the paper "Ours" config via the ``scanner_harm`` template
    must yield the architecture the ``1voovf9c`` checkpoint was trained
    with. Mismatch means the JSON overlay silently isn't landing — and
    ``infer.py`` would load the wrong model with ``strict=False`` and
    emit garbage.
    """
    conf = templates_dict["scanner_harm"]("oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs")
    assert tuple(conf.img_size) == (192, 224)
    assert conf.net_enc_z_sem_dim == 256
    assert conf.slices_around_middle == 10
    # Bonus: confirm the CACD split is intact (z_sem + z_id == cond_channels).
    assert conf.net_enc_z_sem_dim + conf.net_enc_z_id_dim == conf.net_cond_channels
