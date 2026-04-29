"""Tests for the pure-function helpers in ``eval_harm_baselines.py``.

The full ``main()`` flow exercises wandb, dataset loading, and rendering —
those need real data and are exercised by ``multi_slurm_eval_harm_baselines.sh``
on the cluster. Here we pin the small utility functions whose contracts
the downstream CSV/plotting steps depend on.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from eval_harm_baselines import calc_mean_metrics, get_image_size


def test_calc_mean_metrics_groups_by_method_and_id() -> None:
    """``calc_mean_metrics`` must group by ``(method, method_specific_name,
    wandb_id, global_step)`` so each (run, checkpoint-step) pair shows up
    as one row. ``global_step`` was added to the keys to keep separate
    checkpoints of the same wandb_id from collapsing together."""
    df = pd.DataFrame(
        {
            "method": ["DiffAE", "DiffAE", "HACA3", "HACA3"],
            "method_specific_name": ["x", "x", None, None],
            "wandb_id": ["1voovf9c", "1voovf9c", None, None],
            "global_step": [100, 100, None, None],
            "PSNR": [25.0, 27.0, 22.0, 24.0],
            "MS-SSIM": [0.9, 0.92, 0.85, 0.87],
        }
    )
    out = calc_mean_metrics(df)
    assert len(out) == 2
    assert out.loc[("DiffAE", "x", "1voovf9c", 100), "PSNR"] == 26.0
    # HACA3 row: pandas groupby converts None → NaN in the index labels.
    haca3_row = out[out.index.get_level_values("method") == "HACA3"]
    assert len(haca3_row) == 1
    ms_ssim_col = cast(pd.Series, haca3_row["MS-SSIM"])
    assert float(ms_ssim_col.iloc[0]) == 0.86


def test_get_image_size_returns_default_for_pretrained() -> None:
    assert get_image_size("pretrained") == (192, 224, 192)
    assert get_image_size(None) == (192, 224, 192)


def test_get_image_size_reads_conf_json() -> None:
    """A real release config must round-trip through ``get_image_size``.
    The 2D paper config has ``img_size`` of (192, 224) (the trailing 192
    is dropped at config-build time for ``dims=2``)."""
    size = get_image_size("oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs")
    assert isinstance(size, tuple)
    assert all(isinstance(s, int) for s in size)
    assert len(size) >= 2
