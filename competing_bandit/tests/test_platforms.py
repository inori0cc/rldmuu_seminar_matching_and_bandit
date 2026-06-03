"""Tests for ETCPlatform and GSPlatform."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gale_shapley import gale_shapley
from src.platforms import ETCPlatform, GSPlatform
from tests.test_gale_shapley import is_stable


def test_etc_cyclic_covers_each_pair_exactly_h_times():
    N, K, h = 3, 4, 5
    arm_prefs = np.tile(np.arange(N), (K, 1))
    p = ETCPlatform(N, K, h, arm_prefs)
    counter: Counter = Counter()
    for t in range(1, h * K + 1):
        a = p.assign(t, rankings=np.zeros((N, K), dtype=int))
        for i_py in range(N):
            counter[(i_py, int(a[i_py]))] += 1
    # Each (agent, arm) visited exactly h times
    assert all(counter[(i, j)] == h for i in range(N) for j in range(K)), counter


def test_etc_no_collision_each_round():
    N, K, h = 4, 4, 3
    arm_prefs = np.tile(np.arange(N), (K, 1))
    p = ETCPlatform(N, K, h, arm_prefs)
    for t in range(1, h * K + 1):
        a = p.assign(t, rankings=np.zeros((N, K), dtype=int))
        assert len(set(a.tolist())) == N  # injective


def test_etc_gs_invoked_at_hk_plus_1():
    N, K, h = 2, 2, 3
    arm_prefs = np.array([[0, 1], [0, 1]])
    p = ETCPlatform(N, K, h, arm_prefs)
    # exhaust cyclic stage
    for t in range(1, h * K + 1):
        p.assign(t, rankings=np.zeros((N, K), dtype=int))
    # At t = hK+1, submit rankings and check GS is used
    rankings = np.array([[0, 1], [1, 0]])  # p1 prefers a1, p2 prefers a2
    committed = p.assign(h * K + 1, rankings=rankings)
    expected = gale_shapley(rankings, arm_prefs)
    np.testing.assert_array_equal(committed, expected)


def test_etc_commit_stage_is_constant():
    N, K, h = 2, 2, 2
    arm_prefs = np.array([[0, 1], [0, 1]])
    p = ETCPlatform(N, K, h, arm_prefs)
    for t in range(1, h * K + 1):
        p.assign(t, rankings=np.zeros((N, K), dtype=int))
    rankings = np.array([[0, 1], [1, 0]])
    committed = p.assign(h * K + 1, rankings=rankings)
    for t in range(h * K + 2, h * K + 10):
        a = p.assign(t, rankings=np.zeros((N, K), dtype=int))
        np.testing.assert_array_equal(a, committed)


def test_gs_platform_outputs_stable():
    rng = np.random.default_rng(0)
    N, K = 4, 4
    arm_prefs = np.stack([rng.permutation(N) for _ in range(K)])
    p = GSPlatform(N, K, arm_prefs)
    agent_rankings = np.stack([rng.permutation(K) for _ in range(N)])
    match = p.assign(t=1, rankings=agent_rankings)
    assert is_stable(match, agent_rankings, arm_prefs)


def test_gs_platform_no_collision():
    rng = np.random.default_rng(0)
    N, K = 5, 5
    arm_prefs = np.stack([rng.permutation(N) for _ in range(K)])
    p = GSPlatform(N, K, arm_prefs)
    for seed in range(20):
        sub_rng = np.random.default_rng(seed)
        rankings = np.stack([sub_rng.permutation(K) for _ in range(N)])
        match = p.assign(t=seed + 1, rankings=rankings)
        assert len(set(match.tolist())) == N
