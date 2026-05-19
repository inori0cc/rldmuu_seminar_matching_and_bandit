"""Per-round simulation loop (spec §3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adversaries import Adversary
from .algorithms import Algorithm
from .gale_shapley import gale_shapley
from .market import Market


@dataclass
class TrialResult:
    matchings: np.ndarray            # (T, N) int8
    cost: np.ndarray                 # (T, N) float32
    first_exhaust_round: np.ndarray  # (N,) int64, -1 if never
    total_cost: np.ndarray           # (N,) float64
    zero_rounds: np.ndarray          # (N,) int64
    active_rounds: np.ndarray        # (N,) int64
    saturated_rounds: np.ndarray     # (N,) int64, rounds where delta_t == eps
    avg_cost_per_active: np.ndarray  # (N,) float64
    m_star: np.ndarray               # (N,) int64
    mu: np.ndarray                   # (N, K) float64
    T: int
    N: int
    K: int


def run_trial(
    market: Market,
    algorithms: list[Algorithm],
    adversary: Adversary,
    T: int,
    rng_reward: np.random.Generator,
    C: float,
    eps: float,
) -> TrialResult:
    """Run T rounds of (clean sample → adversary corrupt → ranks → GS → update → observe).

    `C` is the per-player budget used only for the post-loop assertion and
    logging. The adversary already enforces its own budget internally.
    """
    N = market.N
    K = market.K
    assert len(algorithms) == N

    matchings = np.empty((T, N), dtype=np.int8)
    cost = np.empty((T, N), dtype=np.float32)
    first_exhaust = np.full(N, -1, dtype=np.int64)
    total_cost = np.zeros(N, dtype=np.float64)
    zero_rounds = np.zeros(N, dtype=np.int64)
    active_rounds = np.zeros(N, dtype=np.int64)
    saturated_rounds = np.zeros(N, dtype=np.int64)

    player_idx = np.arange(N)

    adv_kind = getattr(adversary, "kind", "pre")

    for t_idx in range(T):
        t = t_idx + 1  # spec uses 1-indexed t

        rS = market.sample_clean(rng_reward)                  # (N, K)

        if adv_kind == "pre":
            r, cost_t = adversary.corrupt(t, rS)              # before m_t
            rankings = np.stack([alg.rank(t) for alg in algorithms])  # (N, K)
            m_t = gale_shapley(rankings, market.arm_prefs)    # (N,)
            X = r[player_idx, m_t]                            # (N,) observed corrupted
        else:  # "post" — strong adversary acts after m_t is known
            rankings = np.stack([alg.rank(t) for alg in algorithms])
            m_t = gale_shapley(rankings, market.arm_prefs)
            X, cost_t = adversary.attack_observation(t, rS, m_t)
            r = rS  # placeholder for observe(); post-matching adversaries don't read it

        for i in range(N):
            algorithms[i].update(int(m_t[i]), float(X[i]))

        adversary.observe(t, m_t, X, rS, r)

        matchings[t_idx] = m_t.astype(np.int8)
        cost[t_idx] = cost_t.astype(np.float32)
        total_cost += cost_t

        # budget accounting
        active_mask = cost_t > 0
        zero_mask = ~active_mask
        saturated_mask = np.isclose(cost_t, eps, atol=1e-12)
        active_rounds += active_mask.astype(np.int64)
        zero_rounds += zero_mask.astype(np.int64)
        saturated_rounds += saturated_mask.astype(np.int64)
        # first time total_cost exhausts C (within numerical slack)
        if C > 0:
            for i in range(N):
                if first_exhaust[i] == -1 and total_cost[i] >= C - 1e-9:
                    first_exhaust[i] = t

    assert np.all(total_cost <= C + 1e-6), (
        f"budget violated: total_cost={total_cost}, C={C}"
    )

    avg_cost_per_active = total_cost / np.maximum(active_rounds, 1).astype(np.float64)

    return TrialResult(
        matchings=matchings,
        cost=cost,
        first_exhaust_round=first_exhaust,
        total_cost=total_cost,
        zero_rounds=zero_rounds,
        active_rounds=active_rounds,
        saturated_rounds=saturated_rounds,
        avg_cost_per_active=avg_cost_per_active,
        m_star=market.m_star.astype(np.int64),
        mu=market.mu.astype(np.float64),
        T=T,
        N=N,
        K=K,
    )
