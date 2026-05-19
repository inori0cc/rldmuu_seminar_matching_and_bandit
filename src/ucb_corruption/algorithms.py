"""UCB algorithms (spec §5).

Both algorithms share `_rank_descending`, the single source of truth for the
ascending-arm-index tie-break (spec §9.3). Unpulled arms (N=0) are ranked
before pulled arms in ascending index order; pulled arms are sorted by
descending UCB score with ascending-index tie-break.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .tracker import ChronoHistoryTracker, EmpiricalMeanTracker


def _log_t(t: int) -> float:
    """log(max(t, 2)) per spec §9.2."""
    return float(np.log(max(t, 2)))


def _rank_descending(scores: np.ndarray, unpulled_mask: np.ndarray) -> np.ndarray:
    """Return arm indices sorted: unpulled first (asc index), then pulled by
    descending score with ascending-index tie-break."""
    unpulled = np.flatnonzero(unpulled_mask)               # already ascending
    pulled = np.flatnonzero(~unpulled_mask)
    if pulled.size == 0:
        return unpulled.astype(np.int64)
    # lexsort: last key is primary. Primary -score ascending == score descending.
    # Secondary `pulled` (arm index) ascending breaks ties.
    order = np.lexsort((pulled, -scores[pulled]))
    return np.concatenate([unpulled, pulled[order]]).astype(np.int64)


class Algorithm(Protocol):
    K: int
    tracker: EmpiricalMeanTracker

    def rank(self, t: int) -> np.ndarray: ...
    def update(self, j: int, x: float) -> None: ...


class VanillaUCB:
    """U_{i,j}(t) = mu_hat + sqrt(alpha * log(max(t,2)) / max(N,1)); +inf if N=0."""

    def __init__(self, K: int, alpha: float = 1.5) -> None:
        self.K = K
        self.alpha = float(alpha)
        self.tracker = EmpiricalMeanTracker(K)

    def _scores(self, t: int) -> np.ndarray:
        counts = self.tracker.counts
        safe = np.maximum(counts, 1)
        width = np.sqrt(self.alpha * _log_t(t) / safe)
        return self.tracker.mean + width

    def rank(self, t: int) -> np.ndarray:
        return _rank_descending(self._scores(t), self.tracker.unpulled_mask)

    def update(self, j: int, x: float) -> None:
        self.tracker.update(j, x)


class MoMGSUCB:
    """Median-of-Means GS-UCB (Bubeck-Cesa-Bianchi-Lugosi 2013 style).

    True unknown-C robust learner: the constructor takes only `K`, the horizon
    `T` (used to set the BCBL13 group-count cap k_cap = 2⌈log T⌉+1), and the
    UCB coefficient `alpha_MoM`. No `C`, no `eps` — the corruption budget is
    not visible to the algorithm.

    Estimator: μ̂ = median of k chronological group means.

    Group count: k_cap = 2⌈log T⌉+1, further capped to ⌊√N⌋ so each group
    holds ~√N samples (avoids the group_size=1 collapse). Made odd. For
    N < 3 or k < 3 we fall back to the sample mean.

    Radius: sqrt(alpha_MoM * k * log(max(t,2)) / N). The √k factor matches
    the honest CI for MoM (group means have variance ∝ k/N).
    """

    def __init__(
        self,
        K: int,
        T: int,
        alpha_MoM: float = 8.0,
        n_min_for_mom: int = 3,
    ) -> None:
        self.K = K
        self.T = int(T)
        self.alpha_MoM = float(alpha_MoM)
        # Cold-start fallback: until N >= n_min_for_mom, use sample mean.
        self.n_min_for_mom = int(n_min_for_mom)
        self.tracker = ChronoHistoryTracker(K)
        self.mu_mom_cache = np.zeros(K, dtype=np.float64)
        self._last_samples_used = np.zeros(K, dtype=np.int64)
        # BCBL13: k = Θ(log(1/δ)) with δ ~ 1/T → k = O(log T).
        self._k_cap = 2 * int(np.ceil(np.log(max(self.T, 2)))) + 1
        if self._k_cap % 2 == 0:
            self._k_cap -= 1

    def _choose_k(self, N: int) -> int:
        # Cap so each group has at least ~sqrt(N) samples → group_size >= sqrt(N).
        sqrtN = int(np.floor(np.sqrt(N)))
        k = min(self._k_cap, sqrtN)
        if k % 2 == 0:
            k -= 1
        return max(k, 1)

    def _compute_mom(self, j: int) -> float:
        H = self.tracker.view(j)
        N = H.size
        if N == 0:
            return 0.0
        if N < self.n_min_for_mom:
            return float(np.mean(H))
        k = self._choose_k(N)
        if k < 3:
            return float(np.mean(H))
        group_size = N // k
        samples_used = group_size * k
        # Saturated-regime shortcut: if samples_used unchanged from last
        # recompute on this arm, the partition (and hence MoM) is identical.
        if samples_used == int(self._last_samples_used[j]):
            return float(self.mu_mom_cache[j])
        group_means = H[:samples_used].reshape(k, group_size).mean(axis=1)
        self._last_samples_used[j] = samples_used
        return float(np.median(group_means))

    def rank(self, t: int) -> np.ndarray:
        counts = self.tracker.counts
        safe = np.maximum(counts, 1)
        # Per-arm k for honest CI; matches the k used in _compute_mom.
        k_per_arm = np.array(
            [self._choose_k(int(n)) if n >= 3 else 1 for n in counts],
            dtype=np.float64,
        )
        radius = np.sqrt(self.alpha_MoM * k_per_arm * _log_t(t) / safe)
        scores = self.mu_mom_cache + radius
        return _rank_descending(scores, self.tracker.unpulled_mask)

    def update(self, j: int, x: float) -> None:
        self.tracker.update(j, x)
        self.mu_mom_cache[j] = self._compute_mom(j)


