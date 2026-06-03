"""Tests for EmpiricalMeanTracker, rank_by_empirical_mean, rank_by_ucb."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import EmpiricalMeanTracker, rank_by_empirical_mean, rank_by_ucb


def test_tracker_incremental_mean():
    t = EmpiricalMeanTracker(3)
    rewards = [1.0, 3.0, 5.0]
    for r in rewards:
        t.update(1, r)
    mu = t.empirical_mean()
    assert mu[1] == np.mean(rewards)
    assert mu[0] == 0.0
    assert mu[2] == 0.0
    assert t.counts[1] == 3
    assert t.counts[0] == 0


def test_rank_by_ucb_all_inf_tie_break_random():
    t = EmpiricalMeanTracker(5)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)
    r1 = rank_by_ucb(t, t=1, rng=rng1)
    r2 = rank_by_ucb(t, t=1, rng=rng2)
    assert sorted(r1.tolist()) == list(range(5))
    assert sorted(r2.tolist()) == list(range(5))
    assert not np.array_equal(r1, r2)  # different rngs => different rankings


def test_rank_by_ucb_is_permutation_for_all_K():
    t = EmpiricalMeanTracker(7)
    rng = np.random.default_rng(0)
    for arm, reward in [(0, 1.0), (2, 2.0), (5, 0.5)]:
        t.update(arm, reward)
    ranking = rank_by_ucb(t, t=10, rng=rng)
    assert sorted(ranking.tolist()) == list(range(7))


def test_rank_by_ucb_converges_to_true_ordering():
    """After many pulls, UCB ranking should approximate the true ranking."""
    K = 4
    true_means = np.array([0.1, 0.9, 0.5, 0.3])
    t = EmpiricalMeanTracker(K)
    rng = np.random.default_rng(42)
    n_pulls = 2000
    for arm in range(K):
        for _ in range(n_pulls):
            r = rng.normal(true_means[arm], 1.0)
            t.update(arm, r)
    ranking = rank_by_ucb(t, t=K * n_pulls + 1, rng=rng)
    # Top ranked should be arm 1 (highest mean)
    assert ranking[0] == 1
    # Bottom ranked should be arm 0 (lowest mean)
    assert ranking[-1] == 0


def test_rank_by_empirical_mean_basic():
    t = EmpiricalMeanTracker(3)
    rng = np.random.default_rng(0)
    t.update(0, 1.0)
    t.update(1, 5.0)
    t.update(2, 3.0)
    r = rank_by_empirical_mean(t, rng)
    assert r[0] == 1  # highest mean
    assert r[-1] == 0  # lowest mean
