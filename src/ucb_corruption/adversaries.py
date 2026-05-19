"""Per-player adversaries (spec §4).

The `corrupt(t, rS)` signature deliberately omits `m_t`: this structurally
enforces the timing constraint r^S → r → m_t → X (spec §3). `observe` is
called after matching+feedback for B's mirror trackers.
"""
from __future__ import annotations

from typing import ClassVar, Protocol

import numpy as np

from .tracker import EmpiricalMeanTracker


def _argmax_asc(values: np.ndarray, exclude: int | None = None) -> int:
    """Argmax with ascending-arm-index tie-break (spec §9.3).

    `np.argmax` returns the first occurrence; iterating values in ascending
    index order yields the lowest-index maximizer naturally.
    """
    if exclude is None:
        return int(np.argmax(values))
    masked = values.copy()
    masked[exclude] = -np.inf
    return int(np.argmax(masked))


class Adversary(Protocol):
    """All adversaries enforce per-player budget C internally.

    NOTE: `corrupt` signature has no `m_t` parameter (timing constraint).
    """

    def corrupt(self, t: int, rS: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def observe(
        self,
        t: int,
        m_t: np.ndarray,
        X: np.ndarray,
        rS: np.ndarray,
        r: np.ndarray,
    ) -> None: ...


class NoOp:
    """Identity adversary. Used in unit/smoke tests; NOT part of the experiment grid."""

    kind: ClassVar[str] = "pre"

    def __init__(self, N: int, K: int) -> None:
        self.N = N
        self.K = K

    def corrupt(self, t: int, rS: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rS, np.zeros(self.N, dtype=np.float64)

    def observe(self, t, m_t, X, rS, r) -> None:  # noqa: ARG002
        return None


class A2:
    """A2: oblivious anchored suboptimal-arm boosting (spec §4.1).

    Target arm j_sub[i] = argmax_{j != m_star[i]} mu[i, j] is fixed at
    construction; corrupted coordinate is rS[i, j_sub[i]] + delta_t.
    """

    kind: ClassVar[str] = "pre"

    def __init__(
        self,
        N: int,
        K: int,
        C: float,
        eps: float,
        m_star: np.ndarray,
        mu: np.ndarray,
    ) -> None:
        self.N = N
        self.K = K
        self.C = float(C)
        self.eps = float(eps)
        self.m_star = np.asarray(m_star, dtype=np.int64)
        self.j_sub = np.array(
            [_argmax_asc(mu[i], exclude=int(self.m_star[i])) for i in range(N)],
            dtype=np.int64,
        )
        self.budget_rem = np.full(N, self.C, dtype=np.float64)

    def corrupt(self, t: int, rS: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = rS.copy()
        cost = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            j = int(self.j_sub[i])
            headroom = 1.0 - float(rS[i, j])
            delta = min(self.eps, headroom, float(self.budget_rem[i]))
            if delta > 0.0:
                r[i, j] = rS[i, j] + delta
                cost[i] = delta
                self.budget_rem[i] -= delta
        return r, cost

    def observe(self, t, m_t, X, rS, r) -> None:  # noqa: ARG002
        return None


class B:
    """B: adaptive Wu-style medium adversary (spec §4.2).

    Maintains a mirror of each player's sufficient statistics from the observed
    corrupted match rewards X_i(s). Warm-up gating: while any arm is unpulled
    for player i, no budget is spent. Dynamic target: empirical runner-up if
    j_top == m_star[i], else j_top.
    """

    kind: ClassVar[str] = "pre"

    def __init__(
        self,
        N: int,
        K: int,
        C: float,
        eps: float,
        m_star: np.ndarray,
    ) -> None:
        self.N = N
        self.K = K
        self.C = float(C)
        self.eps = float(eps)
        self.m_star = np.asarray(m_star, dtype=np.int64)
        self.budget_rem = np.full(N, self.C, dtype=np.float64)
        self.mirrors: list[EmpiricalMeanTracker] = [EmpiricalMeanTracker(K) for _ in range(N)]

    def _pick_target(self, i: int) -> int:
        mirror = self.mirrors[i]
        mu_hat = mirror.mean
        j_top = _argmax_asc(mu_hat)
        if j_top == int(self.m_star[i]):
            return _argmax_asc(mu_hat, exclude=int(self.m_star[i]))
        return j_top

    def corrupt(self, t: int, rS: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = rS.copy()
        cost = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            mirror = self.mirrors[i]
            if mirror.unpulled_mask.any():
                # warm-up: dominate-by-+inf-unpulled means corruption is moot
                continue
            j = self._pick_target(i)
            headroom = 1.0 - float(rS[i, j])
            delta = min(self.eps, headroom, float(self.budget_rem[i]))
            if delta > 0.0:
                r[i, j] = rS[i, j] + delta
                cost[i] = delta
                self.budget_rem[i] -= delta
        return r, cost

    def observe(self, t, m_t, X, rS, r) -> None:  # noqa: ARG002
        # Mirror tracks what the player sees: the corrupted match reward X.
        for i in range(self.N):
            self.mirrors[i].update(int(m_t[i]), float(X[i]))


class C:
    """C: strong post-matching adversary (Wang et al. style).

    Observes m_t before acting and modifies only the realized observation
    X_i(t) = r_tilde_{i, m_t(i)}(t). Targeted strong boosting: spends budget
    only when the player happens to match its target j_sub[i]. Same C is
    therefore more efficient than A2 (which boosts j_sub every round even
    when the player never observes that coordinate).

    NOTE: this class implements `attack_observation(t, rS, m_t)`, NOT
    `corrupt(t, rS)`. The simulator dispatches via `kind == "post"`.
    """

    kind: ClassVar[str] = "post"

    def __init__(
        self,
        N: int,
        K: int,
        C: float,
        eps: float,
        m_star: np.ndarray,
        mu: np.ndarray,
    ) -> None:
        self.N = N
        self.K = K
        self.C = float(C)
        self.eps = float(eps)
        self.m_star = np.asarray(m_star, dtype=np.int64)
        self.j_sub = np.array(
            [_argmax_asc(mu[i], exclude=int(self.m_star[i])) for i in range(N)],
            dtype=np.int64,
        )
        self.budget_rem = np.full(N, self.C, dtype=np.float64)

    def attack_observation(
        self,
        t: int,
        rS: np.ndarray,
        m_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X (N,), cost (N,)).

        Default X[i] = rS[i, m_t[i]]. If m_t[i] == j_sub[i], boost the
        observation by delta = min(eps, 1 - rS[i, j_sub], budget_rem[i]).
        """
        N = self.N
        player_idx = np.arange(N)
        X = rS[player_idx, m_t.astype(np.int64)].astype(np.float64).copy()
        cost = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if int(m_t[i]) != int(self.j_sub[i]):
                continue
            j = int(self.j_sub[i])
            headroom = 1.0 - float(rS[i, j])
            delta = min(self.eps, headroom, float(self.budget_rem[i]))
            if delta > 0.0:
                X[i] = float(np.clip(rS[i, j] + delta, 0.0, 1.0))
                cost[i] = delta
                self.budget_rem[i] -= delta
        return X, cost

    def observe(self, t, m_t, X, rS, r) -> None:  # noqa: ARG002
        # Stateless across rounds (j_sub fixed, budget tracked inside attack).
        return None
