"""Unit tests for the InfoNCE losses used to disentangle z_a and z_c.

Loss math (Eq. 1 in the CACD paper, supervised contrastive form):

    ℓ(z_i, P_i, N_i) = -log( Σ_{k∈P} exp(sim(z_i, z_k)/τ)
                           / Σ_{k∈P∪N} exp(sim(z_i, z_k)/τ) )

Tests check shape, finiteness, monotonicity (aligned positives lower the
loss), and the negative-masking semantics of ``MaskedNegativesInfoNCELoss``.
"""

from __future__ import annotations

import torch

from diffae.loss import InfoNCELoss, MaskedNegativesInfoNCELoss


def _balanced_labels(n_groups: int, group_size: int) -> torch.Tensor:
    """Return a (B, B) {0,1} matrix where entries within the same group are 1.

    The diagonal is 1 too (will be stripped by the loss before indexing).
    """
    g = torch.arange(n_groups).repeat_interleave(group_size)
    return (g[:, None] == g[None, :]).long()


def test_infonce_forward_shape_and_finite() -> None:
    loss_fn = InfoNCELoss(temperature=0.1)
    features = torch.randn(8, 16)
    labels = _balanced_labels(n_groups=4, group_size=2)
    loss = loss_fn(features, labels)
    assert loss.shape == (8,)
    assert torch.isfinite(loss).all()


def test_infonce_aligned_positives_lower_than_random() -> None:
    """Aligned positives + dissimilar negatives → smaller loss than random."""
    loss_fn = InfoNCELoss(temperature=0.1)
    n_groups, group_size = 4, 2
    labels = _balanced_labels(n_groups, group_size)

    # Random features
    random_features = torch.randn(n_groups * group_size, 16)
    random_loss = loss_fn(random_features, labels).mean()

    # Aligned: all members of a group share a basis vector; groups orthogonal.
    aligned = torch.zeros(n_groups * group_size, n_groups)
    g = torch.arange(n_groups).repeat_interleave(group_size)
    aligned[torch.arange(n_groups * group_size), g] = 1.0
    aligned_loss = loss_fn(aligned, labels).mean()

    assert aligned_loss < random_loss


def test_infonce_temperature_monotone() -> None:
    """Lower temperature sharpens the softmax; for aligned positives this
    pushes loss further down."""
    n_groups, group_size = 3, 3
    labels = _balanced_labels(n_groups, group_size)
    aligned = torch.zeros(n_groups * group_size, n_groups)
    g = torch.arange(n_groups).repeat_interleave(group_size)
    aligned[torch.arange(n_groups * group_size), g] = 1.0

    loss_hot = InfoNCELoss(temperature=0.05)(aligned, labels).mean()
    loss_warm = InfoNCELoss(temperature=1.0)(aligned, labels).mean()
    assert loss_hot < loss_warm


def test_masked_negatives_excludes_masked_pairs() -> None:
    """Masking out all negatives should drop the loss compared to keeping them.

    Removing dissimilar negatives makes the denominator smaller relative to
    the positives' numerator, so log(num/den) is larger and ``-log`` is
    smaller — i.e., loss decreases.
    """
    torch.manual_seed(0)
    n_groups, group_size = 4, 2
    labels = _balanced_labels(n_groups, group_size)
    features = torch.randn(n_groups * group_size, 16)

    # All non-self pairs eligible as negatives (matches plain InfoNCE):
    full_mask = torch.ones_like(labels, dtype=torch.bool)
    masked_loss_fn = MaskedNegativesInfoNCELoss(temperature=0.1)
    full_neg_loss = masked_loss_fn(features, labels.bool(), full_mask).mean()

    # Drop *every* negative — only positives count → loss ≈ 0.
    no_neg_mask = labels.bool().clone()
    empty_neg_loss = masked_loss_fn(features, labels.bool(), no_neg_mask).mean()

    assert empty_neg_loss < full_neg_loss
    assert empty_neg_loss.abs() < 1e-4


def test_masked_negatives_matches_plain_when_full_mask() -> None:
    """With a full (all-True) negative mask, MaskedNegativesInfoNCELoss should
    reproduce the plain InfoNCELoss result up to numerical noise."""
    n_groups, group_size = 4, 2
    labels = _balanced_labels(n_groups, group_size)
    features = torch.randn(n_groups * group_size, 16)

    plain = InfoNCELoss(temperature=0.1)(features, labels)
    masked = MaskedNegativesInfoNCELoss(temperature=0.1)(
        features, labels.bool(), torch.ones_like(labels, dtype=torch.bool)
    )
    assert torch.allclose(plain, masked, atol=1e-5)
