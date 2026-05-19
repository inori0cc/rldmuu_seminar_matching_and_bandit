"""Tracker + UCB low-level checks."""
from __future__ import annotations

import numpy as np

from ucb_corruption.algorithms import VanillaUCB, _log_t
from ucb_corruption.tracker import EmpiricalMeanTracker


def test_log_guard_at_t_1():
    assert _log_t(1) == np.log(2)
    assert _log_t(2) == np.log(2)
    assert _log_t(10) == np.log(10)


def test_tracker_mean_is_zero_when_unpulled():
    tr = EmpiricalMeanTracker(K=5)
    np.testing.assert_array_equal(tr.mean, np.zeros(5))
    tr.update(2, 1.0)
    assert tr.mean[2] == 1.0
    assert tr.mean[0] == 0.0
    assert tr.unpulled_mask[0]
    assert not tr.unpulled_mask[2]


def test_bernoulli_sample_mean_converges():
    tr = EmpiricalMeanTracker(K=2)
    rng = np.random.default_rng(0)
    p = 0.7
    for _ in range(5000):
        tr.update(0, float(rng.random() < p))
    assert abs(tr.mean[0] - p) < 0.02


def test_vanilla_ucb_unpulled_first_ascending():
    alg = VanillaUCB(K=5)
    # Only pull arm 2 once with high reward
    alg.update(2, 1.0)
    ranking = alg.rank(t=10)
    # Unpulled arms (0,1,3,4) come first in ascending order; pulled (2) last
    assert ranking.tolist()[:4] == [0, 1, 3, 4]
    assert ranking.tolist()[4] == 2


def test_ucb_score_monotone_decreasing_in_N():
    """For fixed mu_hat and t, increasing the count of an arm strictly decreases its UCB width."""
    K = 3
    a = VanillaUCB(K=K)
    # Pull arm 0 once with reward 0.5
    a.update(0, 0.5)
    s1 = a._scores(t=100)[0]
    a.update(0, 0.5)
    s2 = a._scores(t=100)[0]
    a.update(0, 0.5)
    s3 = a._scores(t=100)[0]
    assert s1 > s2 > s3
