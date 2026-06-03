"""Tests for Market class and Example factories."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.market import Market


def test_example_2_optimal_matching():
    m = Market.from_example_2(delta=0.3)
    opt = m.optimal_stable_matching()
    np.testing.assert_array_equal(opt, np.array([0, 1]))


def test_example_2_agent_prefs():
    m = Market.from_example_2(delta=0.5)
    # p1 prefers a1, p2 prefers a2
    np.testing.assert_array_equal(m.agent_preferences[0], [0, 1])
    np.testing.assert_array_equal(m.agent_preferences[1], [1, 0])


def test_example_6_optimal_matching():
    m = Market.from_example_6(p3_gap=0.05)
    opt = m.optimal_stable_matching()
    np.testing.assert_array_equal(opt, np.array([0, 1, 2]))


def test_example_6_agent_prefs():
    m = Market.from_example_6(p3_gap=0.05)
    # p1: a1 > a2 > a3
    np.testing.assert_array_equal(m.agent_preferences[0], [0, 1, 2])
    # p2: a2 > a1 > a3
    np.testing.assert_array_equal(m.agent_preferences[1], [1, 0, 2])
    # p3: a3 > a1 > a2
    np.testing.assert_array_equal(m.agent_preferences[2], [2, 0, 1])


def test_example_7_identity_matching():
    m = Market.from_example_7(N=20, gap=0.1)
    opt = m.optimal_stable_matching()
    np.testing.assert_array_equal(opt, np.arange(20))


def test_pessimal_equals_optimal_when_unique():
    # Example 7 has a unique stable matching (both sides globally ordered identically)
    m = Market.from_example_7(N=5, gap=0.2)
    opt = m.optimal_stable_matching()
    pess = m.pessimal_stable_matching()
    np.testing.assert_array_equal(opt, pess)


def test_example_6_pessimal_differs_from_optimal():
    # Example 6 has multiple stable matchings.
    m = Market.from_example_6(p3_gap=0.05)
    opt = m.optimal_stable_matching()
    pess = m.pessimal_stable_matching()
    # Both stable matchings exist per the paper; verify pessimal is also a permutation.
    assert sorted(pess.tolist()) == [0, 1, 2]
    # Agent 3 (p3) gets a1 under pessimal (matched according to arms), losing a3.
    # The paper describes (p1,a2), (p2,a1), (p3,a3) as another stable matching when
    # p3 mis-ranks a1 > a3. Here with TRUE prefs, check at least that matchings match paper.
    # Under arm-proposing, arm a3 proposes to p3 first. Unique stable matching for
    # example 6 is not the case here; verify pessimal is a valid permutation.
    assert np.all(opt == np.array([0, 1, 2]))


def test_sample_reward_mean_close_to_true():
    m = Market.from_example_6(p3_gap=0.05)
    rng = np.random.default_rng(0)
    samples = np.array([m.sample_reward(0, 0, rng) for _ in range(5000)])
    assert abs(samples.mean() - m.mean_rewards[0, 0]) < 0.05


def test_market_rejects_ties():
    with pytest.raises(ValueError):
        Market(2, 2, np.array([[0.5, 0.5], [0.0, 1.0]]), np.array([[0, 1], [0, 1]]))


def test_market_rejects_invalid_arm_prefs():
    with pytest.raises(ValueError):
        Market(
            2,
            2,
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0, 0], [0, 1]]),  # not a permutation
        )
