"""Market: preferences + reward distributions + ground-truth stable matchings."""
from __future__ import annotations

import numpy as np

from src.gale_shapley import gale_shapley


class Market:
    def __init__(
        self,
        N: int,
        K: int,
        mean_rewards: np.ndarray,
        arm_preferences: np.ndarray,
        reward_std: float = 1.0,
    ):
        assert K >= N, f"need K >= N, got N={N}, K={K}"
        mean_rewards = np.asarray(mean_rewards, dtype=np.float64)
        arm_preferences = np.asarray(arm_preferences, dtype=np.int64)
        assert mean_rewards.shape == (N, K), (
            f"mean_rewards shape {mean_rewards.shape} != ({N}, {K})"
        )
        assert arm_preferences.shape == (K, N), (
            f"arm_preferences shape {arm_preferences.shape} != ({K}, {N})"
        )
        for i in range(N):
            if len(np.unique(mean_rewards[i])) != K:
                raise ValueError(
                    f"mean_rewards[{i}] has ties; strict preferences required."
                )
        for j in range(K):
            if sorted(arm_preferences[j].tolist()) != list(range(N)):
                raise ValueError(
                    f"arm_preferences[{j}] is not a permutation of [0..{N - 1}]"
                )

        self.N = N
        self.K = K
        self.mean_rewards = mean_rewards
        self.arm_preferences = arm_preferences
        self.reward_std = float(reward_std)

    @property
    def agent_preferences(self) -> np.ndarray:
        """shape (N, K), agent i's ranking over arms (0 = most preferred)."""
        return np.argsort(-self.mean_rewards, axis=1).astype(np.int64)

    def sample_reward(self, agent: int, arm: int, rng: np.random.Generator) -> float:
        mu = self.mean_rewards[agent, arm]
        return float(rng.normal(loc=mu, scale=self.reward_std))

    def optimal_stable_matching(self) -> np.ndarray:
        """Agent-proposing GS on true preferences => agent-optimal stable matching."""
        return gale_shapley(self.agent_preferences, self.arm_preferences)

    def pessimal_stable_matching(self) -> np.ndarray:
        """Arm-proposing GS on true preferences => agent-pessimal stable matching.

        Arm-proposing yields an arm-optimal matching, which for agents (the
        non-proposing side) is the pessimal stable matching.
        """
        arm_match = gale_shapley(self.arm_preferences, self.agent_preferences)
        # arm_match[j] = agent matched to arm j; invert to agent -> arm.
        agent_match = np.full(self.N, -1, dtype=np.int64)
        for j, i in enumerate(arm_match):
            if i != -1:
                agent_match[i] = j
        return agent_match

    # ---------- Factory methods for paper Examples 2, 6, 7 ----------

    @classmethod
    def from_example_2(cls, delta: float, reward_std: float = 1.0) -> "Market":
        """Example 2: N=K=2, p1 has gap delta, both arms prefer p1.

        p1: a1 > a2 with gap delta  =>  mu_1 = [delta, 0]
        p2: a2 > a1 with gap 1      =>  mu_2 = [0, 1]
        a1: p1 > p2;  a2: p1 > p2
        """
        if delta <= 0:
            raise ValueError("delta must be > 0 for strict preferences.")
        N, K = 2, 2
        mean_rewards = np.array([[delta, 0.0], [0.0, 1.0]])
        arm_preferences = np.array([[0, 1], [0, 1]])
        return cls(N, K, mean_rewards, arm_preferences, reward_std)

    @classmethod
    def from_example_6(cls, p3_gap: float = 0.05, reward_std: float = 1.0) -> "Market":
        """Example 6: N=K=3.

        Agent prefs:
          p1: a1 > a2 > a3   mu_1 = [2, 1, 0]   (consecutive gap 1)
          p2: a2 > a1 > a3   mu_2 = [1, 2, 0]   (consecutive gap 1)
          p3: a3 > a1 > a2   mu_3 = [1, 0, 1 + p3_gap]   (a3 - a1 gap = p3_gap)
        Arm prefs:
          a1: p2 > p3 > p1
          a2: p1 > p2 > p3
          a3: p3 > p1 > p2
        """
        if p3_gap <= 0:
            raise ValueError("p3_gap must be > 0.")
        N, K = 3, 3
        mean_rewards = np.array(
            [
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 0.0, 1.0 + p3_gap],
            ]
        )
        arm_preferences = np.array(
            [
                [1, 2, 0],  # a1: p2 > p3 > p1
                [0, 1, 2],  # a2: p1 > p2 > p3
                [2, 0, 1],  # a3: p3 > p1 > p2
            ]
        )
        return cls(N, K, mean_rewards, arm_preferences, reward_std)

    @classmethod
    def from_example_7(
        cls, N: int = 20, gap: float = 0.1, reward_std: float = 1.0
    ) -> "Market":
        """Example 7: N=K, global preferences.

        All agents share ranking a_0 > a_1 > ... > a_{N-1}.
        All arms share ranking p_0 > p_1 > ... > p_{N-1}.
        mu_i(j) = (N - 1 - j) * gap, identical across agents.
        """
        if gap <= 0:
            raise ValueError("gap must be > 0.")
        K = N
        row = np.array([(N - 1 - j) * gap for j in range(K)], dtype=np.float64)
        mean_rewards = np.tile(row, (N, 1))
        arm_preferences = np.tile(np.arange(N, dtype=np.int64), (K, 1))
        return cls(N, K, mean_rewards, arm_preferences, reward_std)
