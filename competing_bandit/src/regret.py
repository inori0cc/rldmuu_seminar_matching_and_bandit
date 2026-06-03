"""Cumulative regret computation per paper Eqs. (1) and (2).

Agent-optimal:  R_i(n) = n * mu_i(m_opt(i)) - sum_t E[X_{i, m_t(i)}(t)]
Agent-pessimal: R_i(n) = n * mu_i(m_pess(i)) - sum_t E[X_{i, m_t(i)}(t)]

With expected=True we subtract mean_rewards (variance-reduced estimator);
with expected=False we subtract the realized rewards.
"""
from __future__ import annotations

import numpy as np

from src.market import Market


def _cumulative_regret(
    baseline_matching: np.ndarray,
    market: Market,
    matchings: np.ndarray,
    rewards: np.ndarray,
    use_expected: bool,
) -> np.ndarray:
    N = market.N
    horizon = matchings.shape[0]
    assert matchings.shape == (horizon, N)
    baseline_mu = market.mean_rewards[np.arange(N), baseline_matching]  # (N,)
    if use_expected:
        played_mu = market.mean_rewards[
            np.arange(N)[None, :].repeat(horizon, axis=0), matchings
        ]
        instantaneous = baseline_mu[None, :] - played_mu
    else:
        instantaneous = baseline_mu[None, :] - rewards
    return np.cumsum(instantaneous, axis=0)


def agent_optimal_regret(
    market: Market,
    matchings: np.ndarray,
    rewards: np.ndarray,
    use_expected: bool = True,
) -> np.ndarray:
    return _cumulative_regret(
        market.optimal_stable_matching(), market, matchings, rewards, use_expected
    )


def agent_pessimal_regret(
    market: Market,
    matchings: np.ndarray,
    rewards: np.ndarray,
    use_expected: bool = True,
) -> np.ndarray:
    return _cumulative_regret(
        market.pessimal_stable_matching(), market, matchings, rewards, use_expected
    )
