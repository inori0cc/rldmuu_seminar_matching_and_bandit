from __future__ import annotations

"""
Learning algorithms for the matching-bandits project.

This file contains two groups of algorithms.

1. Paper-style one-sided learning algorithms:
   - run_ucb_gs
   - run_etc

   In these algorithms, players learn their preferences over arms.
   Arm preferences are assumed to be known by the platform, as in the paper.

2. Extension algorithms:
   - run_two_sided_ucb_gs
   - run_two_sided_etc

   In these algorithms, both sides learn:
   - players learn mu[i, j]: how much player i likes arm j;
   - arms learn nu[j, i]: how much arm j likes player i.

All algorithms return match_history, which is later used by metrics.py to evaluate
reward regret, binary stable regret, blocking pairs, and time-to-stability.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from .market import MarketInstance, run_round, sample_reward
from .matching import gale_shapley


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class UCBSummary:
    """
    Output of the original one-sided UCB+GS algorithm.

    mu_hat[i, j]:
        current estimate of player i's mean reward for arm j.

    counts[i, j]:
        number of times player i successfully observed arm j.

    match_history[t]:
        matching produced at round t, represented as player -> arm.

    reward_history[t]:
        player rewards observed at round t.
    """
    mu_hat: np.ndarray
    counts: np.ndarray
    match_history: List[List[Optional[int]]]
    reward_history: List[Dict[int, float]]


@dataclass
class ETCSummary:
    """
    Output of the original one-sided Explore-Then-Commit algorithm.

    committed_match:
        final matching chosen after the exploration phase.
    """
    mu_hat: np.ndarray
    counts: np.ndarray
    committed_match: List[Optional[int]]
    match_history: List[List[Optional[int]]]
    reward_history: List[Dict[int, float]]


@dataclass
class TwoSidedUCBSummary:
    """
    Output of the two-sided UCB+GS extension.

    mu_hat and mu_counts track player-side learning.
    nu_hat and nu_counts track arm-side learning.

    This is the main difference with the original paper-style setting:
    arm preferences are not given to the platform anymore; they are estimated.
    """
    mu_hat: np.ndarray
    mu_counts: np.ndarray
    nu_hat: np.ndarray
    nu_counts: np.ndarray
    match_history: List[List[Optional[int]]]
    player_reward_history: List[Dict[int, float]]
    arm_reward_history: List[Dict[int, float]]


@dataclass
class TwoSidedETCSummary:
    """
    Output of the two-sided Explore-Then-Commit extension.

    After forced exploration, the algorithm commits to one matching computed
    from estimated player rankings and estimated arm rankings.
    """
    mu_hat: np.ndarray
    mu_counts: np.ndarray
    nu_hat: np.ndarray
    nu_counts: np.ndarray
    committed_match: List[Optional[int]]
    match_history: List[List[Optional[int]]]
    player_reward_history: List[Dict[int, float]]
    arm_reward_history: List[Dict[int, float]]


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _ucb_scores(mu_hat_row: np.ndarray, counts_row: np.ndarray, t: int) -> np.ndarray:
    """
    Compute UCB scores for one decision-maker.

    The same formula is used for both sides:
    - for a player, candidates are arms;
    - for an arm, candidates are players.

    UCB intuition:
        score = empirical mean + uncertainty bonus

    A candidate that has never been observed gets +infinity. This forces
    every side to keep unexplored candidates attractive at the beginning.
    """
    scores = np.full_like(mu_hat_row, np.inf, dtype=float)
    seen = counts_row > 0

    if np.any(seen):
        scores[seen] = mu_hat_row[seen] + np.sqrt(
            (3.0 * np.log(max(t, 2))) / (2.0 * counts_row[seen])
        )

    return scores


def _ranking_from_scores(
    scores: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Convert numerical scores into a ranking.

    Higher score = more preferred candidate.

    When several candidates have the same score, random tie-breaking avoids
    deterministic artifacts. This matters especially at the beginning, when
    many UCB scores are +infinity.
    """
    if rng is None:
        return list(np.argsort(-scores))

    tie_break = rng.random(scores.shape[0])
    return list(np.lexsort((tie_break, -scores)))


def _round_robin_matching(N: int, K: int, t: int) -> List[int]:
    """
    Deterministic exploration schedule used by ETC.

    At each exploration round, every player is assigned to one arm.
    If N <= K, this produces no collisions and ensures that every player
    samples every arm once every K rounds.

    Example with N = K = 3:
        round 1: p0->a1, p1->a2, p2->a0
        round 2: p0->a2, p1->a0, p2->a1
        round 3: p0->a0, p1->a1, p2->a2
    """
    return [(i + t) % K for i in range(N)]


# ---------------------------------------------------------------------------
# Original paper-style algorithms: one-sided learning
# ---------------------------------------------------------------------------

def run_ucb_gs(
    instance: MarketInstance,
    T: int,
    rng: np.random.Generator,
    reward_model: str = "bernoulli",
    reward_sigma: float = 1.0,
) -> UCBSummary:
    """
    Original paper-style centralized UCB + Gale-Shapley.

    This is the known-arm-preference baseline.

    What is learned?
        Players learn mu[i, j], their reward value for each arm.

    What is assumed known?
        Arm preferences are known to the platform through instance.arm_prefs.

    At each round:
        1. each player ranks arms using UCB scores;
        2. the platform runs Gale-Shapley using:
             - estimated player rankings;
             - true arm rankings;
        3. matched players observe rewards;
        4. player-side estimates are updated.

    This baseline is important because it isolates the cost of adding
    arm-side learning in the extension.
    """
    N, K = instance.mu.shape

    # Player-side estimates only.
    mu_hat = np.zeros((N, K), dtype=float)
    counts = np.zeros((N, K), dtype=int)

    match_history: List[List[Optional[int]]] = []
    reward_history: List[Dict[int, float]] = []

    for t in range(1, T + 1):
        players_rankings: List[List[int]] = []

        # Each player builds a ranking over arms from UCB scores.
        for i in range(N):
            scores = _ucb_scores(mu_hat[i], counts[i], t=t)
            ranking = _ranking_from_scores(scores, rng=rng)
            players_rankings.append(ranking)

        # run_round uses true arm preferences by default.
        match, rewards = run_round(
            instance,
            players_rankings,
            rng=rng,
            reward_model=reward_model,
            reward_sigma=reward_sigma,
        )

        # Only successful matched pairs generate observations.
        for i, j in enumerate(match):
            if j is None:
                continue

            r = rewards[i]
            counts[i, j] += 1

            # Incremental mean update:
            # new_mean = old_mean + (new_observation - old_mean) / count
            mu_hat[i, j] += (r - mu_hat[i, j]) / counts[i, j]

        match_history.append(match)
        reward_history.append(rewards)

    return UCBSummary(
        mu_hat=mu_hat,
        counts=counts,
        match_history=match_history,
        reward_history=reward_history,
    )


def run_etc(
    instance: MarketInstance,
    T: int,
    rng: np.random.Generator,
    h: int,
    reward_model: str = "bernoulli",
    reward_sigma: float = 1.0,
) -> ETCSummary:
    """
    Original paper-style Explore-Then-Commit.

    This is the one-sided ETC baseline.

    Exploration phase:
        every player samples every arm h times using a round-robin schedule.

    Commitment phase:
        players rank arms using empirical means;
        the platform computes one Gale-Shapley matching using true arm prefs;
        the same matching is repeated until the horizon.

    This is useful as a simple baseline where exploration and exploitation
    are separated.
    """
    N, K = instance.mu.shape

    mu_hat = np.zeros((N, K), dtype=float)
    counts = np.zeros((N, K), dtype=int)

    match_history: List[List[Optional[int]]] = []
    reward_history: List[Dict[int, float]] = []

    explore_steps = min(h * K, T)

    # -------------------------
    # Phase 1: forced exploration
    # -------------------------
    for t in range(1, explore_steps + 1):
        match = _round_robin_matching(N, K, t)
        rewards: Dict[int, float] = {}

        for i, j in enumerate(match):
            r = sample_reward(
                instance.mu[i, j],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )
            rewards[i] = r

            counts[i, j] += 1
            mu_hat[i, j] += (r - mu_hat[i, j]) / counts[i, j]

        match_history.append(match)
        reward_history.append(rewards)

    # In a full exploration phase, every player-arm pair should be sampled h times.
    if explore_steps == h * K:
        assert np.all(counts == h), (
            "Exploration did not sample each player-arm pair exactly h times:\n"
            f"{counts}"
        )

    # -------------------------
    # Phase 2: commit
    # -------------------------
    players_rankings: List[List[int]] = []
    for i in range(N):
        scores = mu_hat[i].copy()

        # If T is too small and some arms were never sampled, put them last.
        scores[counts[i] == 0] = -np.inf
        players_rankings.append(_ranking_from_scores(scores))

    # Arm preferences are still true/known in the one-sided baseline.
    committed_match = gale_shapley(players_rankings, instance.arm_prefs)

    # Repeat the committed matching until the horizon.
    for _ in range(explore_steps + 1, T + 1):
        rewards: Dict[int, float] = {}

        for i, j in enumerate(committed_match):
            if j is None:
                continue

            r = sample_reward(
                instance.mu[i, j],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )
            rewards[i] = r

            counts[i, j] += 1
            mu_hat[i, j] += (r - mu_hat[i, j]) / counts[i, j]

        match_history.append(committed_match)
        reward_history.append(rewards)

    return ETCSummary(
        mu_hat=mu_hat,
        counts=counts,
        committed_match=committed_match,
        match_history=match_history,
        reward_history=reward_history,
    )


# ---------------------------------------------------------------------------
# Extension algorithms: two-sided learning
# ---------------------------------------------------------------------------

def run_two_sided_ucb_gs(
    instance: MarketInstance,
    T: int,
    rng: np.random.Generator,
    reward_model: str = "bernoulli",
    reward_sigma: float = 1.0,
) -> TwoSidedUCBSummary:
    """
    Extension: two-sided UCB + Gale-Shapley.

    This is the main extension algorithm.

    Difference from run_ucb_gs:
        run_ucb_gs learns only player preferences and uses true arm rankings.
        run_two_sided_ucb_gs learns both player preferences and arm preferences.

    What is learned?
        Players learn:
            mu[i, j] = player i's utility for arm j.

        Arms learn:
            nu[j, i] = arm j's utility for player i.

    At each round:
        1. players rank arms using UCB over mu_hat;
        2. arms rank players using UCB over nu_hat;
        3. Gale-Shapley runs using estimated rankings on both sides;
        4. matched pairs observe two rewards:
             - a player reward sampled from mu[i, j];
             - an arm reward sampled from nu[j, i];
        5. both estimates are updated.

    Important:
        Evaluation is done later using the true preferences. Therefore, a
        matching can be stable according to estimated rankings but unstable
        according to true rankings. This is exactly what binary stable regret
        measures.
    """
    if instance.nu is None:
        raise ValueError("Two-sided learning requires instance.nu.")

    N, K = instance.mu.shape

    # Player-side learning tables:
    # mu_hat[i, j] estimates how much player i values arm j.
    mu_hat = np.zeros((N, K), dtype=float)
    mu_counts = np.zeros((N, K), dtype=int)

    # Arm-side learning tables:
    # nu_hat[j, i] estimates how much arm j values player i.
    nu_hat = np.zeros((K, N), dtype=float)
    nu_counts = np.zeros((K, N), dtype=int)

    match_history: List[List[Optional[int]]] = []
    player_reward_history: List[Dict[int, float]] = []
    arm_reward_history: List[Dict[int, float]] = []

    for t in range(1, T + 1):
        players_rankings: List[List[int]] = []
        arms_rankings: List[List[int]] = []

        # Players rank arms using their current optimistic estimates.
        for i in range(N):
            scores = _ucb_scores(mu_hat[i], mu_counts[i], t=t)
            players_rankings.append(_ranking_from_scores(scores, rng=rng))

        # Arms rank players using their current optimistic estimates.
        # This replaces the true arm rankings used in the original setting.
        for j in range(K):
            scores = _ucb_scores(nu_hat[j], nu_counts[j], t=t)
            arms_rankings.append(_ranking_from_scores(scores, rng=rng))

        # The platform now uses estimated rankings on both sides.
        match = gale_shapley(players_rankings, arms_rankings)

        player_rewards: Dict[int, float] = {}
        arm_rewards: Dict[int, float] = {}

        # Matched pairs generate observations for both sides.
        for i, j in enumerate(match):
            if j is None:
                continue

            player_reward = sample_reward(
                instance.mu[i, j],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )
            arm_reward = sample_reward(
                instance.nu[j, i],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )

            player_rewards[i] = player_reward
            arm_rewards[j] = arm_reward

            # Update player-side estimate for pair (i, j).
            mu_counts[i, j] += 1
            mu_hat[i, j] += (player_reward - mu_hat[i, j]) / mu_counts[i, j]

            # Update arm-side estimate for pair (j, i).
            nu_counts[j, i] += 1
            nu_hat[j, i] += (arm_reward - nu_hat[j, i]) / nu_counts[j, i]

        match_history.append(match)
        player_reward_history.append(player_rewards)
        arm_reward_history.append(arm_rewards)

    return TwoSidedUCBSummary(
        mu_hat=mu_hat,
        mu_counts=mu_counts,
        nu_hat=nu_hat,
        nu_counts=nu_counts,
        match_history=match_history,
        player_reward_history=player_reward_history,
        arm_reward_history=arm_reward_history,
    )


def run_two_sided_etc(
    instance: MarketInstance,
    T: int,
    rng: np.random.Generator,
    h: int,
    reward_model: str = "bernoulli",
    reward_sigma: float = 1.0,
) -> TwoSidedETCSummary:
    """
    Extension: two-sided Explore-Then-Commit.

    This is the controlled two-sided baseline.

    Difference from run_etc:
        run_etc explores only player-side rewards and uses true arm prefs.
        run_two_sided_etc explores both player-side and arm-side rewards.

    Exploration phase:
        every player-arm pair is sampled h times;
        both mu_hat and nu_hat are updated.

    Commitment phase:
        players rank arms using estimated mu_hat;
        arms rank players using estimated nu_hat;
        Gale-Shapley is run once;
        the resulting matching is repeated until the horizon.

    Why this algorithm is useful:
        It lets us study how large h must be before the market reliably commits
        to a truly stable matching. This is what the ETC h-sweep evaluates.
    """
    if instance.nu is None:
        raise ValueError("Two-sided learning requires instance.nu.")

    N, K = instance.mu.shape

    mu_hat = np.zeros((N, K), dtype=float)
    mu_counts = np.zeros((N, K), dtype=int)

    nu_hat = np.zeros((K, N), dtype=float)
    nu_counts = np.zeros((K, N), dtype=int)

    match_history: List[List[Optional[int]]] = []
    player_reward_history: List[Dict[int, float]] = []
    arm_reward_history: List[Dict[int, float]] = []

    explore_steps = min(h * K, T)

    # -------------------------
    # Phase 1: forced exploration
    # -------------------------
    for t in range(1, explore_steps + 1):
        match = _round_robin_matching(N, K, t)

        player_rewards: Dict[int, float] = {}
        arm_rewards: Dict[int, float] = {}

        for i, j in enumerate(match):
            # The player observes a reward about arm j.
            player_reward = sample_reward(
                instance.mu[i, j],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )

            # The arm observes a reward about player i.
            arm_reward = sample_reward(
                instance.nu[j, i],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )

            player_rewards[i] = player_reward
            arm_rewards[j] = arm_reward

            mu_counts[i, j] += 1
            mu_hat[i, j] += (player_reward - mu_hat[i, j]) / mu_counts[i, j]

            nu_counts[j, i] += 1
            nu_hat[j, i] += (arm_reward - nu_hat[j, i]) / nu_counts[j, i]

        match_history.append(match)
        player_reward_history.append(player_rewards)
        arm_reward_history.append(arm_rewards)

    # If the exploration phase completed normally, every pair should have
    # exactly h observations on both sides.
    if explore_steps == h * K:
        assert np.all(mu_counts == h), (
            "Player-side exploration did not sample each pair h times:\n"
            f"{mu_counts}"
        )
        assert np.all(nu_counts == h), (
            "Arm-side exploration did not sample each pair h times:\n"
            f"{nu_counts}"
        )

    # -------------------------
    # Phase 2: build estimated rankings
    # -------------------------
    players_rankings: List[List[int]] = []
    arms_rankings: List[List[int]] = []

    for i in range(N):
        scores = mu_hat[i].copy()
        scores[mu_counts[i] == 0] = -np.inf
        players_rankings.append(_ranking_from_scores(scores))

    for j in range(K):
        scores = nu_hat[j].copy()
        scores[nu_counts[j] == 0] = -np.inf
        arms_rankings.append(_ranking_from_scores(scores))

    # The committed matching is based only on estimated rankings.
    # It may or may not be stable under the true preferences.
    committed_match = gale_shapley(players_rankings, arms_rankings)

    # -------------------------
    # Phase 3: exploit committed matching
    # -------------------------
    for _ in range(explore_steps + 1, T + 1):
        player_rewards: Dict[int, float] = {}
        arm_rewards: Dict[int, float] = {}

        for i, j in enumerate(committed_match):
            if j is None:
                continue

            player_reward = sample_reward(
                instance.mu[i, j],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )
            arm_reward = sample_reward(
                instance.nu[j, i],
                rng=rng,
                reward_model=reward_model,
                sigma=reward_sigma,
            )

            player_rewards[i] = player_reward
            arm_rewards[j] = arm_reward

            # We keep updating estimates during exploitation, even though ETC
            # does not change the committed matching afterward. This preserves
            # complete logs and does not affect the matching decisions.
            mu_counts[i, j] += 1
            mu_hat[i, j] += (player_reward - mu_hat[i, j]) / mu_counts[i, j]

            nu_counts[j, i] += 1
            nu_hat[j, i] += (arm_reward - nu_hat[j, i]) / nu_counts[j, i]

        match_history.append(committed_match)
        player_reward_history.append(player_rewards)
        arm_reward_history.append(arm_rewards)

    return TwoSidedETCSummary(
        mu_hat=mu_hat,
        mu_counts=mu_counts,
        nu_hat=nu_hat,
        nu_counts=nu_counts,
        committed_match=committed_match,
        match_history=match_history,
        player_reward_history=player_reward_history,
        arm_reward_history=arm_reward_history,
    )