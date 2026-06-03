from __future__ import annotations
from typing import List, Optional, Literal, Dict


ProposerSide = Literal["players", "arms"]


def _build_rank_pos(rankings: List[List[int]]) -> List[Dict[int, int]]:
    """
    rankings[x] is a list of candidates in preference order.
    Returns list of dicts mapping candidate -> rank position (lower is better).
    """
    pos_list: List[Dict[int, int]] = []
    for pref_list in rankings:
        pos_list.append({cand: r for r, cand in enumerate(pref_list)})
    return pos_list


def gale_shapley(
    players_rankings: List[List[int]],
    arms_rankings: List[List[int]],
    proposer: ProposerSide = "players",
) -> List[Optional[int]]:
    """
    Gale–Shapley / Deferred Acceptance algorithm.

    players_rankings[i]: arms in decreasing preference order (best first)
    arms_rankings[j]: players in decreasing preference order (best first)

    proposer="players": players propose to arms (player-optimal stable matching)
    proposer="arms": arms propose to players (player-pessimal stable matching for players)

    Returns:
        match_p2a: list length N where match_p2a[i] is matched arm for player i (or None).
    """
    N = len(players_rankings)
    K = len(arms_rankings)

    if proposer == "players":
        # proposers: players i -> propose to arms
        proposer_prefs = players_rankings
        receiver_prefs = arms_rankings
        receiver_rank_pos = _build_rank_pos(receiver_prefs)  # for each arm: player->rank

        next_proposal = [0] * N
        receiver_match: List[Optional[int]] = [None] * K  # arm -> player
        free_proposers = list(range(N))

        while free_proposers:
            i = free_proposers.pop()
            if next_proposal[i] >= len(proposer_prefs[i]):
                continue

            j = proposer_prefs[i][next_proposal[i]]
            next_proposal[i] += 1

            current = receiver_match[j]
            if current is None:
                receiver_match[j] = i
            else:
                pos_i = receiver_rank_pos[j].get(i, float("inf"))
                pos_c = receiver_rank_pos[j].get(current, float("inf"))
                if pos_i < pos_c:
                    receiver_match[j] = i
                    free_proposers.append(current)
                else:
                    free_proposers.append(i)

        match_p2a: List[Optional[int]] = [None] * N
        for j in range(K):
            i = receiver_match[j]
            if i is not None:
                match_p2a[i] = j
        return match_p2a

    elif proposer == "arms":
        # proposers: arms j -> propose to players
        proposer_prefs = arms_rankings
        receiver_prefs = players_rankings
        receiver_rank_pos = _build_rank_pos(receiver_prefs)  # for each player: arm->rank

        next_proposal = [0] * K
        receiver_match: List[Optional[int]] = [None] * N  # player -> arm
        free_proposers = list(range(K))

        while free_proposers:
            j = free_proposers.pop()
            if next_proposal[j] >= len(proposer_prefs[j]):
                continue

            i = proposer_prefs[j][next_proposal[j]]
            next_proposal[j] += 1

            current = receiver_match[i]
            if current is None:
                receiver_match[i] = j
            else:
                pos_j = receiver_rank_pos[i].get(j, float("inf"))
                pos_c = receiver_rank_pos[i].get(current, float("inf"))
                if pos_j < pos_c:
                    receiver_match[i] = j
                    free_proposers.append(current)
                else:
                    free_proposers.append(j)

        return receiver_match

    else:
        raise ValueError("proposer must be 'players' or 'arms'")


def gs_players_propose(
    players_rankings: List[List[int]],
    arms_rankings: List[List[int]],
) -> List[Optional[int]]:
    """Convenience wrapper: player-optimal stable matching."""
    return gale_shapley(players_rankings, arms_rankings, proposer="players")


def gs_arms_propose(
    players_rankings: List[List[int]],
    arms_rankings: List[List[int]],
) -> List[Optional[int]]:
    """Convenience wrapper: player-pessimal stable matching."""
    return gale_shapley(players_rankings, arms_rankings, proposer="arms")