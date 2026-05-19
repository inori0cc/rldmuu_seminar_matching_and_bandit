"""Per-player sufficient statistics: N_{i,j}, S_{i,j}, mu_hat (spec §5.3)."""
from __future__ import annotations

import numpy as np


class EmpiricalMeanTracker:
    """Counts and reward sums for one player over K arms.

    `mean` is 0 where counts == 0 (the +inf sentinel belongs to UCB scores,
    not to mu_hat itself; cf. spec §4.2 Important convention).
    """

    __slots__ = ("K", "counts", "sums")

    def __init__(self, K: int) -> None:
        self.K = K
        self.counts = np.zeros(K, dtype=np.int64)
        self.sums = np.zeros(K, dtype=np.float64)

    def update(self, j: int, x: float) -> None:
        self.counts[j] += 1
        self.sums[j] += x

    @property
    def mean(self) -> np.ndarray:
        safe = np.maximum(self.counts, 1)
        out = self.sums / safe
        out[self.counts == 0] = 0.0
        return out

    @property
    def unpulled_mask(self) -> np.ndarray:
        return self.counts == 0


class ChronoHistoryTracker:
    """Full chronological observation history per arm.

    `history[j]` is a numpy buffer of capacity >= counts[j]; valid samples
    are at `history[j][:counts[j]]` (zero-copy view). Capacity doubles on
    demand for amortized O(1) append. Reading the valid prefix as
    `tracker.history[j][:tracker.counts[j]]` avoids any Python-list to
    numpy-array copy at MoM-recompute time — critical for large-T runs.
    """

    __slots__ = ("K", "history", "counts", "_capacity")

    def __init__(self, K: int, initial_cap: int = 16) -> None:
        self.K = K
        self.history: list[np.ndarray] = [
            np.empty(initial_cap, dtype=np.float64) for _ in range(K)
        ]
        self._capacity: list[int] = [initial_cap] * K
        self.counts = np.zeros(K, dtype=np.int64)

    def update(self, j: int, x: float) -> None:
        n = int(self.counts[j])
        if n >= self._capacity[j]:
            new_cap = self._capacity[j] * 2
            new_arr = np.empty(new_cap, dtype=np.float64)
            new_arr[:n] = self.history[j][:n]
            self.history[j] = new_arr
            self._capacity[j] = new_cap
        self.history[j][n] = x
        self.counts[j] += 1

    def view(self, j: int) -> np.ndarray:
        """Zero-copy view of the valid samples for arm j."""
        return self.history[j][: int(self.counts[j])]

    @property
    def unpulled_mask(self) -> np.ndarray:
        return self.counts == 0
