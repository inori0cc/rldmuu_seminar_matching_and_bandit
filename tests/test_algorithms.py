"""Algorithm-level invariants."""
from __future__ import annotations

import numpy as np

from ucb_corruption.algorithms import MoMGSUCB, _rank_descending


def test_rank_descending_unpulled_first():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    unpulled = np.array([False, True, False, True, False])
    out = _rank_descending(scores, unpulled).tolist()
    # Unpulled (1, 3) come first in ascending order; pulled by descending score
    # pulled: scores [0.1, 0.5, 0.3] for arms [0, 2, 4] → desc = [2, 4, 0]
    assert out[:2] == [1, 3]
    assert out[2:] == [2, 4, 0]


def test_rank_descending_pulled_tie_break_ascending():
    scores = np.array([0.5, 0.5, 0.5])
    unpulled = np.array([False, False, False])
    out = _rank_descending(scores, unpulled).tolist()
    assert out == [0, 1, 2]


def test_rank_descending_all_unpulled():
    scores = np.zeros(4)
    unpulled = np.array([True, True, True, True])
    out = _rank_descending(scores, unpulled).tolist()
    assert out == [0, 1, 2, 3]


# ---- MoM-GS-UCB ---------------------------------------------------------


def test_mom_cold_start_ascending():
    """All unpulled → ranking [0,1,2,3,4] (UCB=+inf, asc index tie)."""
    alg = MoMGSUCB(K=5, T=1000)
    np.testing.assert_array_equal(alg.rank(t=10), [0, 1, 2, 3, 4])


def test_mom_small_N_fallback_to_mean():
    """N < 3: mu_mom = sample mean of available observations."""
    alg = MoMGSUCB(K=3, T=1000)
    alg.update(0, 0.5)
    alg.update(0, 0.7)
    assert abs(alg.mu_mom_cache[0] - 0.6) < 1e-9


def test_mom_robust_to_outliers():
    """MoM resists tail outliers that would skew the sample mean."""
    alg = MoMGSUCB(K=1, T=1000)
    # 20 normal readings (0.5) + 5 huge outliers (100.0).
    for _ in range(20):
        alg.update(0, 0.5)
    for _ in range(5):
        alg.update(0, 100.0)
    # Sample mean ≈ 20.4. New k_cap = 2*ceil(log 1000)+1 = 15, capped by
    # floor(sqrt(25)) = 5 → k=5, group_size=5. Chronological groups:
    # four groups of 0.5, one group of 100.0 → median = 0.5.
    assert abs(alg.mu_mom_cache[0] - 0.5) < 1e-9


def test_mom_returns_full_permutation():
    rng = np.random.default_rng(0)
    alg = MoMGSUCB(K=5, T=1000)
    for _ in range(30):
        alg.update(int(rng.integers(5)), float(rng.random()))
    r = alg.rank(t=100).tolist()
    assert sorted(r) == [0, 1, 2, 3, 4]


def test_mom_choose_k_caps_and_odd():
    """k = min(2 ceil(log T) + 1, floor(sqrt(N))), made odd."""
    # T=1000 → k_cap = 2*7+1 = 15. C and eps no longer affect k.
    alg = MoMGSUCB(K=1, T=1000)
    assert alg._choose_k(N=50) == 7      # floor(sqrt(50))=7, odd
    assert alg._choose_k(N=200) == 13    # floor(sqrt(200))=14 → 13 (odd)
    assert alg._choose_k(N=400) == 15    # floor(sqrt(400))=20, but k_cap=15
    assert alg._choose_k(N=2) == 1       # floor(sqrt(2))=1


def test_mom_constructor_takes_no_C():
    """MoMGSUCB is truly unknown-C: constructor accepts neither C nor eps."""
    import inspect
    params = inspect.signature(MoMGSUCB.__init__).parameters
    assert "C" not in params
    assert "eps" not in params
    # But it does require K, T (used for k_cap), and alpha_MoM.
    for name in ("K", "T", "alpha_MoM"):
        assert name in params
