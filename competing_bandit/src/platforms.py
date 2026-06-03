"""Platforms: ETCPlatform and GSPlatform (paper Table 1).

ETC cyclic formula: paper writes m_t(i) = a_{(t + i - 1) mod K + 1} in 1-indexed
notation. In 0-indexed Python (t_py = t-1, i_py = i-1), this is:
    arm = (t_py + i_py + 1) % K
Any cyclic shift that produces an injective assignment each round and covers
every (agent, arm) pair exactly h times over hK rounds is functionally
equivalent for Layer 1 purposes; we follow the paper's formula.
"""
from __future__ import annotations

import numpy as np

from src.gale_shapley import gale_shapley


class ETCPlatform:
    def __init__(self, N: int, K: int, h: int, arm_preferences: np.ndarray):
        assert K >= N, f"need K >= N, got N={N}, K={K}"
        assert h >= 1
        self.N = N
        self.K = K
        self.h = h
        self.arm_preferences = np.asarray(arm_preferences, dtype=np.int64)
        self._committed: np.ndarray | None = None

    def assign(self, t: int, rankings: np.ndarray) -> np.ndarray:
        """Return assignment (shape (N,)) for round t (1-indexed).

        rankings: shape (N, K). Only used at t = h*K + 1 when GS is invoked.
        """
        hK = self.h * self.K
        if t <= hK:
            t_py = t - 1
            assignment = np.array(
                [(t_py + i_py + 1) % self.K for i_py in range(self.N)],
                dtype=np.int64,
            )
            return assignment
        if t == hK + 1:
            self._committed = gale_shapley(rankings, self.arm_preferences)
            return self._committed.copy()
        # t > hK + 1 => commit phase
        assert self._committed is not None, "ETCPlatform: commit stage before GS ran"
        return self._committed.copy()


class GSPlatform:
    def __init__(self, N: int, K: int, arm_preferences: np.ndarray):
        assert K >= N, f"need K >= N, got N={N}, K={K}"
        self.N = N
        self.K = K
        self.arm_preferences = np.asarray(arm_preferences, dtype=np.int64)

    def assign(self, t: int, rankings: np.ndarray) -> np.ndarray:
        return gale_shapley(rankings, self.arm_preferences)
