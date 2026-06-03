from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .matching import gs_players_propose, gs_arms_propose


def true_player_rankings(mu: np.ndarray) -> List[List[int]]:
    """Player i ranks arms by descending true mean reward mu[i, j]."""
    N, _ = mu.shape
    return [list(np.argsort(-mu[i])) for i in range(N)]


def stable_matching_player_optimal(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
) -> List[Optional[int]]:
    """Player-optimal true stable matching."""
    players_rankings = true_player_rankings(mu)
    return gs_players_propose(players_rankings, arm_prefs)


def stable_matching_player_pessimal(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
) -> List[Optional[int]]:
    """Player-pessimal true stable matching."""
    players_rankings = true_player_rankings(mu)
    return gs_arms_propose(players_rankings, arm_prefs)


def regret_vs_matching(
    mu: np.ndarray,
    match_history: List[List[Optional[int]]],
    comparator_match: List[Optional[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Summed reward regret over all players.
    """
    N, _ = mu.shape
    T = len(match_history)

    comp_reward = np.zeros(N, dtype=float)
    for i, j in enumerate(comparator_match):
        comp_reward[i] = 0.0 if j is None else float(mu[i, j])

    per_round = np.zeros(T, dtype=float)

    for t, match_t in enumerate(match_history):
        total = 0.0
        for i, j in enumerate(match_t):
            actual = 0.0 if j is None else float(mu[i, j])
            total += comp_reward[i] - actual
        per_round[t] = total

    return per_round, np.cumsum(per_round)


def regret_per_player_vs_matching(
    mu: np.ndarray,
    match_history: List[List[Optional[int]]],
    comparator_match: List[Optional[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-player reward regret.
    """
    N, _ = mu.shape
    T = len(match_history)

    comp_reward = np.zeros(N, dtype=float)
    for i, j in enumerate(comparator_match):
        comp_reward[i] = 0.0 if j is None else float(mu[i, j])

    per_round = np.zeros((T, N), dtype=float)

    for t, match_t in enumerate(match_history):
        for i, j in enumerate(match_t):
            actual = 0.0 if j is None else float(mu[i, j])
            per_round[t, i] = comp_reward[i] - actual

    return per_round, np.cumsum(per_round, axis=0)


def stable_regret_curves_per_player(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_history: List[List[Optional[int]]],
) -> Dict[str, Any]:
    """
    Per-player cumulative regrets against:
        - player-optimal true stable matching;
        - player-pessimal true stable matching.
    """
    comp_opt = stable_matching_player_optimal(mu, arm_prefs)
    comp_pess = stable_matching_player_pessimal(mu, arm_prefs)

    _, cum_opt = regret_per_player_vs_matching(mu, match_history, comp_opt)
    _, cum_pess = regret_per_player_vs_matching(mu, match_history, comp_pess)

    return {
        "comp_opt": comp_opt,
        "comp_pess": comp_pess,
        "cum_opt": cum_opt,
        "cum_pess": cum_pess,
    }


def _rank_positions(rankings: List[List[int]]) -> List[Dict[int, int]]:
    """Convert rankings into candidate -> rank position maps."""
    return [
        {candidate: pos for pos, candidate in enumerate(ranking)}
        for ranking in rankings
    ]


def count_blocking_pairs(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_p2a: List[Optional[int]],
) -> int:
    """
    Count true blocking pairs in a matching.

    A pair (player i, arm j) blocks the matching if:
        - player i prefers j over their current match;
        - arm j prefers i over its current matched player.
    """
    N, K = mu.shape

    player_rankings = true_player_rankings(mu)
    player_pos = _rank_positions(player_rankings)
    arm_pos = _rank_positions(arm_prefs)

    arm_to_player: List[Optional[int]] = [None] * K
    for i, j in enumerate(match_p2a):
        if j is not None:
            arm_to_player[j] = i

    blocking_pairs = 0

    for i in range(N):
        current_arm = match_p2a[i]

        for j in range(K):
            if current_arm == j:
                continue

            current_player = arm_to_player[j]

            if current_arm is None:
                player_prefers_j = True
            else:
                player_prefers_j = player_pos[i][j] < player_pos[i][current_arm]

            if current_player is None:
                arm_prefers_i = True
            else:
                arm_prefers_i = arm_pos[j][i] < arm_pos[j][current_player]

            if player_prefers_j and arm_prefers_i:
                blocking_pairs += 1

    return blocking_pairs


def is_stable_matching(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_p2a: List[Optional[int]],
) -> bool:
    """Return True iff the matching is stable under true preferences."""
    return count_blocking_pairs(mu, arm_prefs, match_p2a) == 0


def binary_stable_regret_curve(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_history: List[List[Optional[int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary stable regret.

    per_round[t] = 1 if match_t is unstable under true preferences, else 0.
    cumulative[t] = total number of unstable rounds up to t.
    """
    unstable = np.array([
        0 if is_stable_matching(mu, arm_prefs, match_t) else 1
        for match_t in match_history
    ], dtype=float)

    return unstable, np.cumsum(unstable)


def blocking_pair_curve(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_history: List[List[Optional[int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Number of true blocking pairs per round and cumulatively.
    """
    counts = np.array([
        count_blocking_pairs(mu, arm_prefs, match_t)
        for match_t in match_history
    ], dtype=float)

    return counts, np.cumsum(counts)


def first_stable_time(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_history: List[List[Optional[int]]],
) -> Optional[int]:
    """
    First round index, 1-based, where the produced matching is stable.

    Returns None if no stable matching appears.
    """
    for t, match_t in enumerate(match_history, start=1):
        if is_stable_matching(mu, arm_prefs, match_t):
            return t
    return None


def time_to_permanent_stability(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_history: List[List[Optional[int]]],
) -> Optional[int]:
    """
    First round index, 1-based, after which all matchings are stable.

    Returns None if the algorithm never remains stable permanently.
    """
    unstable, _ = binary_stable_regret_curve(mu, arm_prefs, match_history)

    for idx in range(len(unstable)):
        if np.all(unstable[idx:] == 0):
            return idx + 1

    return None


def stability_curves(
    mu: np.ndarray,
    arm_prefs: List[List[int]],
    match_history: List[List[Optional[int]]],
) -> Dict[str, Any]:
    """
    Convenience wrapper for all stability-based metrics.
    """
    unstable, cum_binary = binary_stable_regret_curve(mu, arm_prefs, match_history)
    blocking, cum_blocking = blocking_pair_curve(mu, arm_prefs, match_history)

    return {
        "unstable": unstable,
        "cum_binary_stable_regret": cum_binary,
        "blocking_pairs": blocking,
        "cum_blocking_pairs": cum_blocking,
        "first_stable_time": first_stable_time(mu, arm_prefs, match_history),
        "time_to_permanent_stability": time_to_permanent_stability(mu, arm_prefs, match_history),
    }