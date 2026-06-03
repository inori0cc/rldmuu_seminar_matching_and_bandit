from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

from .matching import gale_shapley


@dataclass
class MarketInstance:
    """
    Market instance.

    mu[i, j] = true utility/reward mean of player i for arm j.
    nu[j, i] = true utility/reward mean of arm j for player i.

    arm_prefs is the true arm ranking over players.
    It is derived from nu when nu is available.
    """
    mu: np.ndarray
    arm_prefs: List[List[int]]
    nu: Optional[np.ndarray] = None


def player_rankings_from_mu(mu: np.ndarray) -> List[List[int]]:
    """True player rankings induced by mu."""
    N, _ = mu.shape
    return [list(np.argsort(-mu[i])) for i in range(N)]


def arm_rankings_from_nu(nu: np.ndarray) -> List[List[int]]:
    """True arm rankings induced by nu."""
    K, _ = nu.shape
    return [list(np.argsort(-nu[j])) for j in range(K)]


def rankings_to_scores(
    rankings: List[List[int]],
    n_candidates: int,
    high: float = 0.9,
    low: float = 0.1,
) -> np.ndarray:
    """
    Convert strict rankings into utility scores.

    Example:
        ranking [2, 0, 1] with high=0.9, low=0.1 gives:
        candidate 2 -> 0.9
        candidate 0 -> 0.5
        candidate 1 -> 0.1
    """
    values = np.linspace(high, low, n_candidates)
    scores = np.zeros((len(rankings), n_candidates), dtype=float)

    for row, ranking in enumerate(rankings):
        for pos, candidate in enumerate(ranking):
            scores[row, candidate] = values[pos]

    return scores


def generate_instance(
    N: int,
    K: int,
    rng: np.random.Generator,
    reward_model: str = "bernoulli",
) -> MarketInstance:
    """
    Generate a random two-sided market.

    For Bernoulli rewards, utilities are kept inside (0, 1).
    For Gaussian rewards, utilities can be arbitrary real values.
    """
    if reward_model == "bernoulli":
        mu = rng.uniform(0.05, 0.95, size=(N, K))
        nu = rng.uniform(0.05, 0.95, size=(K, N))
    elif reward_model == "gaussian":
        mu = rng.normal(loc=0.0, scale=1.0, size=(N, K))
        nu = rng.normal(loc=0.0, scale=1.0, size=(K, N))
    else:
        raise ValueError(f"Unsupported reward_model: {reward_model}")

    arm_prefs = arm_rankings_from_nu(nu)
    return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)


def generate_handcrafted_instance(
    name: str = "multi_stable_3x3",
    delta: float | None = None,
) -> MarketInstance:
    """
    Handcrafted instances used for reproduction and extension experiments.

    Original reproduction examples:
        - example2
        - example6
        - example7
        - multi_stable_3x3

    Extension example:
        - two_sided_gap_3x3
        - conflict_arm_gap_3x3
    """
    if name == "example2":
        if delta is None:
            raise ValueError("example2 requires a delta value.")

        mu = np.array([
            [float(delta), 0.0],
            [0.0, 1.0],
        ], dtype=float)

        arm_prefs = [
            [0, 1],
            [0, 1],
        ]

        nu = rankings_to_scores(arm_prefs, n_candidates=2)
        return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)

    if name == "example6":
        mu = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.95, 0.0, 1.0],
        ], dtype=float)

        arm_prefs = [
            [1, 2, 0],
            [0, 1, 2],
            [2, 0, 1],
        ]

        nu = rankings_to_scores(arm_prefs, n_candidates=3)
        return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)

    if name == "example7":
        N = 20
        K = 20
        gap = 0.1

        # Global player preferences: all players rank a1 > a2 > ... > aK.
        mu = np.zeros((N, K), dtype=float)
        for i in range(N):
            for j in range(K):
                mu[i, j] = -gap * j

        # Global arm preferences: all arms rank p1 > p2 > ... > pN.
        arm_prefs = [list(range(N)) for _ in range(K)]
        nu = rankings_to_scores(arm_prefs, n_candidates=N)

        return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)

    if name == "multi_stable_3x3":
        mu = np.array([
            [0.9, 0.6, 0.3],
            [0.3, 0.9, 0.6],
            [0.6, 0.3, 0.9],
        ], dtype=float)

        arm_prefs = [
            [1, 0, 2],
            [2, 1, 0],
            [0, 2, 1],
        ]

        nu = rankings_to_scores(arm_prefs, n_candidates=3)
        return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)

    if name == "two_sided_gap_3x3":
        """
        Simple extension instance.

        Players have clear favorite arms:
            p1 -> a1
            p2 -> a2
            p3 -> a3

        Arms also have corresponding favorite players:
            a1 -> p1
            a2 -> p2
            a3 -> p3

        delta controls how hard arm preferences are to learn.
        Smaller delta = harder two-sided stability learning.
        """
        gap = 0.1 if delta is None else float(delta)
        if not (0.0 < gap < 0.45):
            raise ValueError("For two_sided_gap_3x3, delta should be in (0, 0.45).")

        mu = np.array([
            [0.90, 0.30, 0.10],
            [0.10, 0.90, 0.30],
            [0.30, 0.10, 0.90],
        ], dtype=float)

        arm_rankings = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        ]

        nu = np.zeros((3, 3), dtype=float)
        values = [0.5 + gap, 0.5, 0.5 - gap]

        for arm, ranking in enumerate(arm_rankings):
            for pos, player in enumerate(ranking):
                nu[arm, player] = values[pos]

        arm_prefs = arm_rankings_from_nu(nu)
        return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)

    if name == "conflict_arm_gap_3x3":
        """
        Conflict-heavy two-sided learning instance.

        All players prefer the same arm order:
            a1 > a2 > a3

        All arms prefer the same player order:
            p1 > p2 > p3

        True stable matching:
            p1 -> a1
            p2 -> a2
            p3 -> a3

        This instance makes arm-side learning important because all players
        compete for the best arms. If arms learn their player rankings slowly,
        the platform may output matchings that are unstable under true preferences.

        delta controls how hard arm preferences are to learn:
            large delta = easy arm ranking learning
            small delta = hard arm ranking learning
        """
        gap = 0.1 if delta is None else float(delta)
        if not (0.0 < gap < 0.45):
            raise ValueError("For conflict_arm_gap_3x3, delta should be in (0, 0.45).")

        # All players have the same clear preference order:
        # a1 > a2 > a3
        mu = np.array([
            [0.90, 0.60, 0.30],
            [0.90, 0.60, 0.30],
            [0.90, 0.60, 0.30],
        ], dtype=float)

        # All arms have the same true preference order:
        # p1 > p2 > p3
        arm_rankings = [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ]

        # Arm utility values become harder to distinguish when gap is small.
        values = [0.5 + gap, 0.5, 0.5 - gap]

        nu = np.zeros((3, 3), dtype=float)
        for arm, ranking in enumerate(arm_rankings):
            for pos, player in enumerate(ranking):
                nu[arm, player] = values[pos]

        arm_prefs = arm_rankings_from_nu(nu)
        return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)
    
    raise ValueError(f"Unknown handcrafted instance: {name}")


def sample_reward(
    mean: float,
    rng: np.random.Generator,
    reward_model: str = "bernoulli",
    sigma: float = 1.0,
) -> float:
    """Sample one stochastic reward."""
    if reward_model == "bernoulli":
        if mean < 0.0 or mean > 1.0:
            raise ValueError(
                f"Bernoulli mean must be in [0, 1], got {mean}. "
                "Use reward_model='gaussian' for unrestricted utilities."
            )
        return float(rng.binomial(1, mean))

    if reward_model == "gaussian":
        return float(rng.normal(loc=mean, scale=sigma))

    raise ValueError(f"Unsupported reward_model: {reward_model}")


def run_round(
    instance: MarketInstance,
    players_rankings: List[List[int]],
    rng: np.random.Generator,
    reward_model: str = "bernoulli",
    reward_sigma: float = 1.0,
    arm_rankings: Optional[List[List[int]]] = None,
    sample_arm_rewards: bool = False,
) -> Tuple[List[Optional[int]], Dict[int, float]] | Tuple[List[Optional[int]], Dict[int, float], Dict[int, float]]:
    """
    One centralized market round.

    By default, this reproduces the original setting:
        - players submit learned rankings;
        - platform uses true arm rankings.

    If arm_rankings is provided, the platform uses those estimated arm rankings.
    If sample_arm_rewards=True, arms also observe rewards from matched players.
    """
    rankings_for_arms = instance.arm_prefs if arm_rankings is None else arm_rankings
    match_p2a = gale_shapley(players_rankings, rankings_for_arms)

    player_rewards: Dict[int, float] = {}
    arm_rewards: Dict[int, float] = {}

    for i, j in enumerate(match_p2a):
        if j is None:
            continue

        player_rewards[i] = sample_reward(
            instance.mu[i, j],
            rng=rng,
            reward_model=reward_model,
            sigma=reward_sigma,
        )

        if sample_arm_rewards:
            if instance.nu is None:
                raise ValueError("sample_arm_rewards=True requires instance.nu.")
            arm_rewards[j] = sample_reward(
                instance.nu[j, i],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )

    if sample_arm_rewards:
        return match_p2a, player_rewards, arm_rewards

    return match_p2a, player_rewards