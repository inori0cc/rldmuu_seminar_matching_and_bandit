"""Adversary invariants (spec §4)."""
from __future__ import annotations

import inspect

import numpy as np

from ucb_corruption.adversaries import A2, B, NoOp, _argmax_asc
from ucb_corruption.adversaries import C as AdvC
from ucb_corruption.market import Market
from ucb_corruption.rng import spawn_streams


def _market(seed: int = 0) -> Market:
    s = spawn_streams(seed)
    return Market.build(N=5, K=5, delta=0.2, rng_perm=s.perm, rng_arm=s.arm)


def test_corrupt_signature_excludes_m_t():
    """Structural enforcement of spec §3 timing constraint."""
    for cls in (NoOp, A2, B):
        params = inspect.signature(cls.corrupt).parameters
        assert "m_t" not in params, f"{cls.__name__}.corrupt must not see m_t"


def test_argmax_asc_breaks_ties_ascending():
    v = np.array([0.5, 0.5, 0.5, 0.5])
    assert _argmax_asc(v) == 0
    assert _argmax_asc(v, exclude=0) == 1


def test_noop_zero_cost():
    m = _market()
    s = spawn_streams(0)
    adv = NoOp(N=5, K=5)
    rS = m.sample_clean(s.reward)
    r, cost = adv.corrupt(1, rS)
    np.testing.assert_array_equal(r, rS)
    np.testing.assert_array_equal(cost, np.zeros(5))


def test_a2_target_fixed_and_not_m_star():
    m = _market(1)
    adv = A2(N=5, K=5, C=1000.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    # Target arm must differ from m_star for each player
    assert np.all(adv.j_sub != m.m_star)
    # Save initial j_sub
    initial = adv.j_sub.copy()
    # Run 50 rounds with random rS — j_sub must not change
    s = spawn_streams(1)
    for t in range(1, 51):
        rS = m.sample_clean(s.reward)
        adv.corrupt(t, rS)
        adv.observe(t, np.zeros(5, dtype=np.int64), np.zeros(5), rS, rS)
    np.testing.assert_array_equal(adv.j_sub, initial)


def test_a2_target_is_best_non_stable():
    m = _market(2)
    adv = A2(N=5, K=5, C=100.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    for i in range(5):
        excluded = int(m.m_star[i])
        masked = m.mu[i].copy()
        masked[excluded] = -np.inf
        expected = int(np.argmax(masked))
        assert int(adv.j_sub[i]) == expected


def test_budget_invariant_a2():
    m = _market(3)
    C = 5.0
    eps = 0.2
    adv = A2(N=5, K=5, C=C, eps=eps, m_star=m.m_star, mu=m.mu)
    s = spawn_streams(3)
    total = np.zeros(5)
    for t in range(1, 1001):
        rS = m.sample_clean(s.reward)
        r, cost = adv.corrupt(t, rS)
        # L_inf identity
        linf = np.max(np.abs(r - rS), axis=1)
        np.testing.assert_allclose(linf, cost, atol=1e-12)
        total += cost
        adv.observe(t, np.zeros(5, dtype=np.int64), r[np.arange(5), np.zeros(5, dtype=np.int64)], rS, r)
    assert np.all(total <= C + 1e-9)


def test_a2_at_C_zero_no_corruption():
    m = _market(4)
    adv = A2(N=5, K=5, C=0.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    s = spawn_streams(4)
    for t in range(1, 201):
        rS = m.sample_clean(s.reward)
        r, cost = adv.corrupt(t, rS)
        np.testing.assert_array_equal(r, rS)
        np.testing.assert_array_equal(cost, np.zeros(5))


def test_b_warmup_zero_until_all_arms_pulled():
    m = _market(5)
    adv = B(N=5, K=5, C=100.0, eps=0.2, m_star=m.m_star)
    s = spawn_streams(5)
    # First few rounds: mirror is empty → all unpulled → cost==0
    for t in range(1, 4):
        rS = m.sample_clean(s.reward)
        r, cost = adv.corrupt(t, rS)
        np.testing.assert_array_equal(r, rS)
        np.testing.assert_array_equal(cost, np.zeros(5))
        # Observe a matching that doesn't cover all arms — keeps warm-up on
        m_t = np.zeros(5, dtype=np.int64)
        X = r[np.arange(5), m_t]
        adv.observe(t, m_t, X, rS, r)
        # still some arms unpulled per player


def test_b_at_C_zero_no_corruption():
    m = _market(6)
    adv = B(N=5, K=5, C=0.0, eps=0.2, m_star=m.m_star)
    s = spawn_streams(6)
    # Force mirror to be fully warm by pre-populating
    for i in range(5):
        for j in range(5):
            adv.mirrors[i].update(j, 0.5)
    for t in range(1, 201):
        rS = m.sample_clean(s.reward)
        r, cost = adv.corrupt(t, rS)
        np.testing.assert_array_equal(r, rS)
        np.testing.assert_array_equal(cost, np.zeros(5))


def test_b_target_when_j_top_equals_m_star():
    m = _market(7)
    adv = B(N=5, K=5, C=100.0, eps=0.2, m_star=m.m_star)
    # Pre-populate mirrors so j_top == m_star[i] for each player, and the
    # runner-up is well-defined.
    for i in range(5):
        ms = int(m.m_star[i])
        for j in range(5):
            adv.mirrors[i].update(j, 1.0 if j == ms else 0.5)
        # Make arm `runner` slightly above the rest non-ms arms.
        runner = (ms + 1) % 5
        adv.mirrors[i].update(runner, 0.8)
    # Now pick target; should be runner-up (the non-ms with highest mu_hat).
    for i in range(5):
        ms = int(m.m_star[i])
        runner = (ms + 1) % 5
        target = adv._pick_target(i)
        assert target == runner, f"player {i}: expected {runner}, got {target}"


def test_b_target_when_j_top_differs_from_m_star():
    m = _market(8)
    adv = B(N=5, K=5, C=100.0, eps=0.2, m_star=m.m_star)
    # Force j_top != m_star: make one non-ms arm dominate.
    for i in range(5):
        ms = int(m.m_star[i])
        wrong = (ms + 2) % 5
        for j in range(5):
            adv.mirrors[i].update(j, 0.9 if j == wrong else 0.1)
    for i in range(5):
        wrong = (int(m.m_star[i]) + 2) % 5
        target = adv._pick_target(i)
        assert target == wrong


def test_budget_invariant_b():
    m = _market(9)
    C = 5.0
    eps = 0.2
    adv = B(N=5, K=5, C=C, eps=eps, m_star=m.m_star)
    s = spawn_streams(9)
    total = np.zeros(5)
    for t in range(1, 1001):
        rS = m.sample_clean(s.reward)
        r, cost = adv.corrupt(t, rS)
        linf = np.max(np.abs(r - rS), axis=1)
        np.testing.assert_allclose(linf, cost, atol=1e-12)
        total += cost
        m_t = np.array([t % 5 for _ in range(5)], dtype=np.int64)  # rotate arms so warm-up ends
        X = r[np.arange(5), m_t]
        adv.observe(t, m_t, X, rS, r)
    assert np.all(total <= C + 1e-9)


# ---- Adversary C (strong post-matching) ---------------------------------


def test_C_kind_is_post():
    m = _market(0)
    adv = AdvC(N=5, K=5, C=100.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    assert adv.kind == "post"


def test_C_attack_observation_sees_m_t():
    """Structural confirmation: C's attack DOES receive m_t (the point of post-matching)."""
    params = inspect.signature(AdvC.attack_observation).parameters
    assert "m_t" in params
    assert "rS" in params


def test_C_corrupt_method_absent():
    """C does not implement the pre-matching corrupt(t, rS) method."""
    assert not hasattr(AdvC, "corrupt")


def test_C_target_only_when_matched():
    """Cost > 0 only for players whose m_t hits j_sub."""
    m = _market(2)
    adv = AdvC(N=5, K=5, C=100.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    s = spawn_streams(2)
    rS = m.sample_clean(s.reward)
    # Build m_t: player 0 → j_sub[0] (attacked); others → m_star (not attacked)
    m_t = m.m_star.astype(np.int64).copy()
    m_t[0] = int(adv.j_sub[0])
    X, cost = adv.attack_observation(1, rS, m_t)
    # Player 0 may or may not have nonzero cost depending on headroom; if it does,
    # all OTHER players must have zero cost.
    assert np.all(cost[1:] == 0.0), f"cost on non-matched players should be 0, got {cost}"
    # Players 1..4 should observe their m_star coord unchanged
    for i in range(1, 5):
        assert X[i] == rS[i, m_t[i]]


def test_C_target_attacked_when_matched():
    """When m_t[i] == j_sub[i] and there's headroom + budget, X[i] > rS[i, j_sub]."""
    m = _market(3)
    adv = AdvC(N=5, K=5, C=100.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    # Force a low rS at j_sub for headroom
    rS = np.zeros((5, 5), dtype=np.float64)
    m_t = adv.j_sub.astype(np.int64).copy()  # everyone matches their j_sub
    X, cost = adv.attack_observation(1, rS, m_t)
    np.testing.assert_allclose(cost, 0.2)         # δ_t = min(eps=0.2, 1, 100) = 0.2
    np.testing.assert_allclose(X, 0.2)            # rS=0 + 0.2 = 0.2


def test_budget_invariant_C():
    m = _market(4)
    C_budget = 5.0
    adv = AdvC(N=5, K=5, C=C_budget, eps=0.2, m_star=m.m_star, mu=m.mu)
    s = spawn_streams(4)
    total = np.zeros(5)
    for t in range(1, 1001):
        rS = m.sample_clean(s.reward)
        # Random matchings for each player
        m_t = s.adv.integers(0, 5, size=5).astype(np.int64)
        X, cost = adv.attack_observation(t, rS, m_t)
        total += cost
    assert np.all(total <= C_budget + 1e-9), f"budget violated: {total}"


def test_C_at_C_zero_no_corruption():
    m = _market(5)
    adv = AdvC(N=5, K=5, C=0.0, eps=0.2, m_star=m.m_star, mu=m.mu)
    s = spawn_streams(5)
    for t in range(1, 201):
        rS = m.sample_clean(s.reward)
        m_t = adv.j_sub.astype(np.int64).copy()  # always match j_sub
        X, cost = adv.attack_observation(t, rS, m_t)
        np.testing.assert_array_equal(cost, np.zeros(5))
        # X should equal rS at the matched (j_sub) coord, untouched
        for i in range(5):
            assert X[i] == rS[i, m_t[i]]
