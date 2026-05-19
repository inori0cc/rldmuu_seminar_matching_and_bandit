"""Metric helpers."""
from __future__ import annotations

import numpy as np

from ucb_corruption.metrics import clean_pseudo_regret, downsample, max_regret, stable_match_rate


def test_clean_pseudo_regret_monotone_nondecreasing():
    T, N, K = 100, 3, 4
    mu = np.array([[0.9, 0.5, 0.3, 0.1],
                   [0.1, 0.9, 0.5, 0.3],
                   [0.3, 0.1, 0.9, 0.5]])
    m_star = np.array([0, 1, 2])
    rng = np.random.default_rng(0)
    matchings = rng.integers(0, K, size=(T, N))
    R = clean_pseudo_regret(matchings, mu, m_star)
    diffs = np.diff(R, axis=0)
    assert np.all(diffs >= -1e-12)


def test_max_regret_shape_and_value():
    R = np.array([[1.0, 2.0, 0.5], [2.0, 2.0, 3.0]])
    mr = max_regret(R)
    assert mr.shape == (2,)
    np.testing.assert_allclose(mr, [2.0, 3.0])


def test_downsample_count_and_endpoints():
    curve = np.arange(100_000)
    idx, vals = downsample(curve, n=200)
    assert idx.shape == (200,)
    assert idx[0] == 0
    assert idx[-1] == 99_999
    np.testing.assert_array_equal(vals, curve[idx])


def test_downsample_n_geq_T_returns_full():
    curve = np.arange(50)
    idx, vals = downsample(curve, n=200)
    assert idx.shape == (50,)
    np.testing.assert_array_equal(vals, curve)


def test_stable_match_rate_full_match():
    T, N = 100, 3
    m_star = np.array([0, 1, 2])
    matchings = np.tile(m_star, (T, 1))
    assert stable_match_rate(matchings, m_star, tail_frac=0.1) == 1.0


def test_stable_match_rate_no_match():
    T, N = 100, 3
    m_star = np.array([0, 1, 2])
    matchings = np.tile(np.array([1, 0, 1]), (T, 1))
    assert stable_match_rate(matchings, m_star, tail_frac=0.1) == 0.0
