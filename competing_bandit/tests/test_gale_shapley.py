"""Tests for Gale-Shapley agent-proposing DA."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gale_shapley import gale_shapley


def is_stable(
    matching: np.ndarray,
    proposer_prefs: np.ndarray,
    receiver_prefs: np.ndarray,
) -> bool:
    """No blocking pair exists."""
    N, K = proposer_prefs.shape
    # Rank lookups
    p_rank = np.full((N, K), K, dtype=int)
    for i in range(N):
        for r, j in enumerate(proposer_prefs[i]):
            p_rank[i, j] = r
    r_rank = np.full((K, N), N, dtype=int)
    for j in range(K):
        for r, i in enumerate(receiver_prefs[j]):
            r_rank[j, i] = r

    inverse = {int(j): i for i, j in enumerate(matching) if j != -1}

    for i in range(N):
        current = matching[i]
        for j in range(K):
            if j == current:
                continue
            # proposer i prefers j over current (unmatched counts as worst)
            i_prefers_j = current == -1 or p_rank[i, j] < p_rank[i, current]
            if not i_prefers_j:
                continue
            holder = inverse.get(int(j), -1)
            j_prefers_i = holder == -1 or r_rank[j, i] < r_rank[j, holder]
            if j_prefers_i:
                return False
    return True


def test_gs_textbook_3x3():
    men_prefs = np.array([[0, 1, 2], [1, 0, 2], [0, 1, 2]])
    women_prefs = np.array([[2, 0, 1], [0, 2, 1], [0, 1, 2]])
    matching = gale_shapley(men_prefs, women_prefs)
    assert is_stable(matching, men_prefs, women_prefs)


def test_gs_identity_global_prefs():
    N = 10
    prefs = np.tile(np.arange(N), (N, 1))
    matching = gale_shapley(prefs, prefs)
    np.testing.assert_array_equal(matching, np.arange(N))


def test_gs_unique_partner_each():
    # Each proposer i ranks receiver i first and vice versa
    N = 6
    prefs_p = np.array([np.roll(np.arange(N), -i) for i in range(N)])
    prefs_r = np.array([np.roll(np.arange(N), -j) for j in range(N)])
    matching = gale_shapley(prefs_p, prefs_r)
    np.testing.assert_array_equal(matching, np.arange(N))


@pytest.mark.parametrize("seed", range(50))
def test_gs_random_instances_are_stable(seed):
    rng = np.random.default_rng(seed)
    N = 5
    K = 5
    prefs_p = np.stack([rng.permutation(K) for _ in range(N)])
    prefs_r = np.stack([rng.permutation(N) for _ in range(K)])
    matching = gale_shapley(prefs_p, prefs_r)
    assert is_stable(matching, prefs_p, prefs_r)


def test_gs_proposer_optimality():
    # Classic 2-stable-matchings example (Gusfield-Irving): 2x2 where flipping
    # proposer sides yields two distinct stable matchings.
    # men 0 prefers w0 > w1; men 1 prefers w1 > w0
    # w0 prefers m1 > m0; w1 prefers m0 > m1
    men_prefs = np.array([[0, 1], [1, 0]])
    women_prefs = np.array([[1, 0], [0, 1]])
    m_men = gale_shapley(men_prefs, women_prefs)
    m_women = gale_shapley(women_prefs, men_prefs)
    # men-proposing should give men their top choice
    np.testing.assert_array_equal(m_men, np.array([0, 1]))
    # women-proposing: woman 0 -> m1, woman 1 -> m0 => for men: m0 matched to w1, m1 to w0
    # Invert women's result to express as men's matching
    men_match_under_women = np.full(2, -1, dtype=int)
    for w, m in enumerate(m_women):
        men_match_under_women[m] = w
    np.testing.assert_array_equal(men_match_under_women, np.array([1, 0]))


def test_gs_N_less_than_K():
    # N=2 proposers, K=3 receivers. Some receiver stays unmatched.
    prefs_p = np.array([[0, 1, 2], [0, 1, 2]])
    prefs_r = np.array([[0, 1], [1, 0], [0, 1]])
    matching = gale_shapley(prefs_p, prefs_r)
    assert matching.shape == (2,)
    assert len(set(matching.tolist())) == 2  # no duplicates among matched proposers
    assert is_stable(matching, prefs_p, prefs_r)
