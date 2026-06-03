"""Tests for regret computation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.market import Market
from src.regret import agent_optimal_regret, agent_pessimal_regret


def test_optimal_matching_zero_expected_regret():
    m = Market.from_example_6(p3_gap=0.05)
    opt = m.optimal_stable_matching()
    horizon = 50
    matchings = np.tile(opt, (horizon, 1))
    rewards = np.zeros((horizon, m.N))  # unused when use_expected=True
    cum = agent_optimal_regret(m, matchings, rewards, use_expected=True)
    np.testing.assert_allclose(cum, 0.0)


def test_expected_regret_matches_realized_on_average():
    """With many trials, trial-averaged realized regret should approach expected regret."""
    m = Market.from_example_2(delta=0.5)
    opt = m.optimal_stable_matching()
    horizon = 100
    N = m.N
    n_trials = 500

    # Play optimal matching every round
    matchings = np.tile(opt, (horizon, 1))
    rng = np.random.default_rng(0)

    # Accumulate realized regret across trials
    realized_mean = np.zeros((horizon, N))
    for _ in range(n_trials):
        rewards = np.zeros((horizon, N))
        for i in range(N):
            for t in range(horizon):
                rewards[t, i] = m.sample_reward(i, opt[i], rng)
        realized_mean += agent_optimal_regret(m, matchings, rewards, use_expected=False)
    realized_mean /= n_trials

    expected = agent_optimal_regret(m, matchings, np.zeros((horizon, N)), use_expected=True)
    # Expected regret with optimal matching is 0; realized should average near 0
    # At horizon=100, n_trials=500, std=1: SEM ~ sqrt(100)/sqrt(500) ≈ 0.45
    # Use 3-sigma tolerance of 1.5 so test is flake-free.
    np.testing.assert_allclose(realized_mean[-1], 0.0, atol=1.5)
    np.testing.assert_allclose(expected[-1], 0.0)


def test_regret_is_cumulative_sum():
    m = Market.from_example_2(delta=0.4)
    horizon = 10
    # Always play arm 0 (for both agents): p1 optimal is a1 (idx 0), p2 optimal is a2 (idx 1)
    matchings = np.zeros((horizon, m.N), dtype=int)
    rewards = np.zeros((horizon, m.N))
    cum = agent_optimal_regret(m, matchings, rewards, use_expected=True)
    # p1 plays optimal (arm 0) every round => regret 0
    np.testing.assert_allclose(cum[:, 0], 0.0)
    # p2 plays arm 0 (mu=0) vs optimal arm 1 (mu=1) => regret = 1 each round
    np.testing.assert_allclose(cum[:, 1], np.arange(1, horizon + 1, dtype=float))


def test_pessimal_regret_in_example_6():
    m = Market.from_example_6(p3_gap=0.05)
    pess = m.pessimal_stable_matching()
    horizon = 20
    matchings = np.tile(pess, (horizon, 1))
    rewards = np.zeros((horizon, m.N))
    cum = agent_pessimal_regret(m, matchings, rewards, use_expected=True)
    # Playing the pessimal matching => pessimal regret is 0
    np.testing.assert_allclose(cum, 0.0)
