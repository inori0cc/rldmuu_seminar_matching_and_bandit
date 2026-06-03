"""Per-agent empirical state and ranking strategies.

Notes on UCB formula (paper Eq. 6):
    u_{i,j}(t) = inf                                       if T_{i,j} == 0
               = mu_hat_{i,j}(t) + sqrt(3 log(t) / (2 T))  otherwise
The paper writes T_i(t-1) but means T_{i,j}(t-1) (per-pair count); we use the
current stored count (updated only on successful pulls), which at the start of
round t equals T_{i,j}(t-1) in the paper's time convention.
"""
from __future__ import annotations

import numpy as np


class EmpiricalMeanTracker:
    def __init__(self, K: int):
        self.K = K
        self.counts = np.zeros(K, dtype=np.int64)
        self.sum_rewards = np.zeros(K, dtype=np.float64)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward

    def empirical_mean(self) -> np.ndarray:
        out = np.zeros(self.K, dtype=np.float64)
        pulled = self.counts > 0
        out[pulled] = self.sum_rewards[pulled] / self.counts[pulled]
        return out


_TIE_NOISE = 1e-10


def rank_by_empirical_mean(
    tracker: EmpiricalMeanTracker, rng: np.random.Generator
) -> np.ndarray:
    """Return ranking: descending by empirical mean, random tie-break."""
    mu_hat = tracker.empirical_mean()
    noise = rng.uniform(0, _TIE_NOISE, size=tracker.K)
    return np.argsort(-(mu_hat + noise))


def rank_by_ucb(
    tracker: EmpiricalMeanTracker, t: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ranking: descending by UCB (Eq. 6). Unpulled arms (UCB = +inf)
    are placed first in a uniformly random order. Pulled arms follow, sorted
    by UCB descending with noise tie-break."""
    K = tracker.K
    pulled = tracker.counts > 0
    unpulled_idx = np.flatnonzero(~pulled)
    pulled_idx = np.flatnonzero(pulled)

    unpulled_ranking = rng.permutation(unpulled_idx)

    if pulled_idx.size > 0:
        log_t = np.log(max(t, 1))
        mu_hat = tracker.sum_rewards[pulled] / tracker.counts[pulled]
        width = np.sqrt(3.0 * log_t / (2.0 * tracker.counts[pulled]))
        ucb_vals = mu_hat + width
        noise = rng.uniform(0, _TIE_NOISE, size=pulled_idx.size)
        order = np.argsort(-(ucb_vals + noise))
        pulled_ranking = pulled_idx[order]
    else:
        pulled_ranking = np.array([], dtype=np.int64)

    return np.concatenate([unpulled_ranking, pulled_ranking]).astype(np.int64)
