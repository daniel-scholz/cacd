"""Unit tests for TrainConfig JSON-overlay loading.

The release-relevant configs in ``conf/`` must:

* parse cleanly via ``scanner_harm(name) -> update_config_from_json()``
  (no missing keys, no type-coercion failures);
* satisfy ``net_enc_z_sem_dim + net_enc_z_id_dim == net_cond_channels``.
  CACD configs split this 256+256=512; vanilla configs collapse to
  z_sem=cond_channels and z_id=0.

We deliberately do *not* test every JSON in ``conf/``: many are stale
experimental configs predating the public release. The whitelist below
is the set we publish and want green.
"""

from __future__ import annotations

import pytest

from diffae.templates import scanner_harm

# Configs we publish / care about reproducing:
RELEASE_CONFIGS = [
    # Paper "Ours" run (1voovf9c)
    "oasis3_ixi_guys-hh_2d_cacd_diffae_all_augs",
    # Paper vanilla baseline (72jaa0rm)
    "oasis3_ixi_guys-hh_2d_vanilla_diffae",
    # Gamma-only ablation
    "oasis3_ixi_guys-hh_2d_cacd_diffae_gamma",
    # Gamma + RC ablation
    "oasis3_ixi_guys-hh_2d_cacd_diffae_gamma_rc",
    # Vanilla + intensity augs
    "oasis3_ixi_guys-hh_2d_vanilla_diffae_augs",
]


@pytest.mark.parametrize("conf_name", RELEASE_CONFIGS)
def test_release_conf_loads(conf_name: str) -> None:
    """Loading the conf via the template must not raise."""
    conf = scanner_harm(conf_name)
    assert conf.name == conf_name
    assert conf.model_conf is not None


@pytest.mark.parametrize("conf_name", RELEASE_CONFIGS)
def test_release_conf_dim_invariant(conf_name: str) -> None:
    """``z_sem_dim + z_id_dim`` must equal ``cond_channels``. Holds for
    CACD (split 256+256=512) and for vanilla DiffAE (z_sem=512, z_id=0)
    alike."""
    conf = scanner_harm(conf_name)
    total = conf.net_enc_z_sem_dim + conf.net_enc_z_id_dim
    assert total == conf.net_cond_channels, (
        f"{conf_name}: enc_z_sem_dim ({conf.net_enc_z_sem_dim}) + "
        f"enc_z_id_dim ({conf.net_enc_z_id_dim}) != net_cond_channels "
        f"({conf.net_cond_channels})"
    )
