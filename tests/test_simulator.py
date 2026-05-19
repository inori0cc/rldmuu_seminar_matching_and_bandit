"""End-to-end simulator checks at small T."""
from __future__ import annotations

import numpy as np

from ucb_corruption.adversaries import A2, B, NoOp
from ucb_corruption.adversaries import C as AdvC
from ucb_corruption.algorithms import MoMGSUCB, VanillaUCB
from ucb_corruption.market import Market
from ucb_corruption.rng import spawn_streams
from ucb_corruption.simulator import run_trial


def test_run_trial_deterministic_with_seed():
    s1 = spawn_streams(42)
    s2 = spawn_streams(42)
    m1 = Market.build(N=5, K=5, delta=0.2, rng_perm=s1.perm, rng_arm=s1.arm)
    m2 = Market.build(N=5, K=5, delta=0.2, rng_perm=s2.perm, rng_arm=s2.arm)
    np.testing.assert_array_equal(m1.mu, m2.mu)

    a1 = [VanillaUCB(K=5) for _ in range(5)]
    a2 = [VanillaUCB(K=5) for _ in range(5)]
    r1 = run_trial(m1, a1, NoOp(5, 5), T=200, rng_reward=s1.reward, C=0.0, eps=0.2)
    r2 = run_trial(m2, a2, NoOp(5, 5), T=200, rng_reward=s2.reward, C=0.0, eps=0.2)
    np.testing.assert_array_equal(r1.matchings, r2.matchings)


def test_budget_invariant_in_simulator():
    """Adversary A2 with budget C=50 must respect total_cost <= 50."""
    s = spawn_streams(11)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=s.perm, rng_arm=s.arm)
    algs = [VanillaUCB(K=5) for _ in range(5)]
    adv = A2(N=5, K=5, C=50.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    result = run_trial(m, algs, adv, T=2000, rng_reward=s.reward, C=50.0, eps=0.2)
    assert np.all(result.total_cost <= 50.0 + 1e-6)


def test_run_trial_with_strong_adversary_C():
    """Strong post-matching adversary C runs end-to-end and respects budget."""
    s = spawn_streams(13)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=s.perm, rng_arm=s.arm)
    algs = [VanillaUCB(K=5) for _ in range(5)]
    adv = AdvC(N=5, K=5, C=20.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    result = run_trial(m, algs, adv, T=2000, rng_reward=s.reward, C=20.0, eps=0.2)
    assert np.all(result.total_cost <= 20.0 + 1e-6)
    # Sanity: adversary C reports at least one active round in 2000 rounds.
    assert result.active_rounds.sum() > 0


def test_run_trial_with_mom_algorithm():
    """MoM end-to-end smoke: budget invariant respected, no crashes."""
    s = spawn_streams(30)
    m = Market.build(N=5, K=5, delta=0.2, rng_perm=s.perm, rng_arm=s.arm)
    algs = [MoMGSUCB(K=5, T=2000) for _ in range(5)]
    adv = A2(N=5, K=5, C=20.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    result = run_trial(m, algs, adv, T=2000, rng_reward=s.reward, C=20.0, eps=0.2)
    assert np.all(result.total_cost <= 20.0 + 1e-6)
