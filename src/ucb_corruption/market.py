"""Matching-market instance: mean matrix mu, arm preferences, m* (spec §2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gale_shapley import gale_shapley


@dataclass(frozen=True)
class Market:
    N: int
    K: int
    mu: np.ndarray              # (N, K) float64 in [0, 1]
    arm_prefs: np.ndarray       # (K, N) int64, strict permutations of [N]
    player_prefs: np.ndarray    # (N, K) int64, ranking of arms in descending mu
    m_star: np.ndarray          # (N,) int64, player-optimal stable matching on true mu
    sigma: float = 1.0          # std-dev of clipped Gaussian reward noise (Wu et al. 2025)

    @classmethod
    def build(
        cls,
        N: int,
        K: int,
        delta: float,
        rng_perm: np.random.Generator,
        rng_arm: np.random.Generator,
        top_reward: float = 0.9,
        sigma: float = 1.0,
    ) -> "Market":
        """Construct a market per spec §2.

        For each player i, sample a random arm-permutation rho_i and assign
            mu[i, rho_i(k)] = top_reward - delta * k   (k = 0, .., K-1).
        For each arm j, sample a strict ranking pi_j over players.
        """
        if delta <= 0:
            raise ValueError("delta must be positive")
        if top_reward - delta * (K - 1) < 0:
            raise ValueError("reward levels fall below 0; lower K or delta")

        levels = top_reward - delta * np.arange(K)         # (K,)
        mu = np.empty((N, K), dtype=np.float64)
        for i in range(N):
            rho_i = rng_perm.permutation(K)
            # rho_i[k] = arm ranked k-th by player i → assigns the k-th reward level
            mu[i, rho_i] = levels

        arm_prefs = np.stack([rng_arm.permutation(N) for _ in range(K)]).astype(np.int64)

        # Player rankings: descending mu, with ascending-index tie-break (stable sort).
        # mu values are unique per row by construction, but keep the guard explicit.
        player_prefs = np.argsort(-mu, axis=1, kind="stable").astype(np.int64)

        m_star = gale_shapley(player_prefs, arm_prefs)

        return cls(
            N=N,
            K=K,
            mu=mu,
            arm_prefs=arm_prefs,
            player_prefs=player_prefs,
            m_star=m_star,
            sigma=float(sigma),
        )

    def sample_clean(self, rng: np.random.Generator) -> np.ndarray:
        """Return r^S of shape (N, K) with each entry = clip(N(mu[i,j], sigma), 0, 1).

        Wu et al. 2025 §5 uses Gaussian rewards with sigma=1. We additionally
        clip to [0, 1] so the per-round adversary cost stays bounded and the
        `1 - rS` headroom in A2/B remains non-negative.
        """
        z = rng.normal(loc=self.mu, scale=self.sigma, size=(self.N, self.K))
        return np.clip(z, 0.0, 1.0).astype(np.float64)
