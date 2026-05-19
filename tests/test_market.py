"""Market construction (spec §2)."""
from __future__ import annotations

import numpy as np

from ucb_corruption.market import Market
from ucb_corruption.rng import spawn_streams


def test_mu_rows_are_permutation_of_reward_levels():
    streams = spawn_streams(42)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=streams.perm, rng_arm=streams.arm)
    expected = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    for i in range(5):
        np.testing.assert_allclose(np.sort(m.mu[i])[::-1], expected)


def test_adjacent_gap_exactly_delta():
    streams = spawn_streams(7)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=streams.perm, rng_arm=streams.arm)
    for i in range(5):
        sorted_desc = np.sort(m.mu[i])[::-1]
        diffs = sorted_desc[:-1] - sorted_desc[1:]
        np.testing.assert_allclose(diffs, np.full(4, 0.2))


def test_arm_prefs_are_permutations():
    streams = spawn_streams(13)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=streams.perm, rng_arm=streams.arm)
    for j in range(5):
        assert sorted(m.arm_prefs[j].tolist()) == [0, 1, 2, 3, 4]


def test_player_prefs_sorted_descending_by_mu():
    streams = spawn_streams(13)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=streams.perm, rng_arm=streams.arm)
    for i in range(5):
        vals = m.mu[i, m.player_prefs[i]]
        assert np.all(vals[:-1] >= vals[1:])


def test_m_star_is_a_permutation():
    streams = spawn_streams(99)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=streams.perm, rng_arm=streams.arm)
    assert sorted(m.m_star.tolist()) == [0, 1, 2, 3, 4]


def test_sample_clean_clipped_gaussian_bounds():
    """All samples must lie in [0, 1]; sigma=0.0 should match mu exactly."""
    streams = spawn_streams(0)
    m = Market.build(N=5, K=5, delta=0.2,
                     rng_perm=streams.perm, rng_arm=streams.arm,
                     sigma=1.0)
    for _ in range(200):
        rS = m.sample_clean(streams.reward)
        assert rS.shape == (5, 5)
        assert np.all(rS >= 0.0)
        assert np.all(rS <= 1.0)


def test_sample_clean_zero_sigma_returns_mu():
    streams = spawn_streams(1)
    m = Market.build(N=5, K=5, delta=0.2,
                     rng_perm=streams.perm, rng_arm=streams.arm,
                     sigma=0.0)
    rS = m.sample_clean(streams.reward)
    # mu in [0.1, 0.9], so clipping is a no-op; rS should equal mu exactly.
    np.testing.assert_allclose(rS, m.mu)


def test_clipped_gaussian_distribution_properties():
    """clip(N(0.7, 1), 0, 1):
        P(=0) ≈ Φ(-0.7) ≈ 0.242
        P(=1) ≈ 1 - Φ(0.3) ≈ 0.382
        E    ≈ 0.382 + 0.376·0.516 ≈ 0.576
    (User-supplied spec §4 quoted 0.61, which is incorrect — derived from a
    sign-flipped truncated-normal mean.)"""
    rng = np.random.default_rng(0)
    samples = np.clip(rng.normal(0.7, 1.0, size=10_000), 0.0, 1.0)
    assert 0.20 < (samples == 0).mean() < 0.28
    assert 0.34 < (samples == 1).mean() < 0.42
    assert 0.55 < samples.mean() < 0.60
