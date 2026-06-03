"""Agent-proposing deferred acceptance (Gale-Shapley)."""
from __future__ import annotations

from collections import deque

import numpy as np


def gale_shapley(
    proposer_prefs: np.ndarray,
    receiver_prefs: np.ndarray,
) -> np.ndarray:
    """Return the proposer-optimal stable matching.

    Args:
        proposer_prefs: shape (N, K). proposer_prefs[i, r] is the receiver at
            rank r (0 = most preferred) for proposer i.
        receiver_prefs: shape (K, N). receiver_prefs[j, r] is the proposer at
            rank r (0 = most preferred) for receiver j.

    Returns:
        matching: shape (N,). matching[i] is the receiver matched to proposer i,
            or -1 if proposer i is unmatched (only possible if N > K).
    """
    proposer_prefs = np.asarray(proposer_prefs, dtype=np.int64)
    receiver_prefs = np.asarray(receiver_prefs, dtype=np.int64)

    N, K = proposer_prefs.shape
    assert receiver_prefs.shape == (K, N), (
        f"shape mismatch: proposer {proposer_prefs.shape}, receiver {receiver_prefs.shape}"
    )

    receiver_rank = np.full((K, N), N, dtype=np.int64)
    for j in range(K):
        for r, i in enumerate(receiver_prefs[j]):
            receiver_rank[j, i] = r

    next_proposal = np.zeros(N, dtype=np.int64)
    current_match = np.full(K, -1, dtype=np.int64)
    proposer_match = np.full(N, -1, dtype=np.int64)

    free = deque(range(N))

    while free:
        i = free.popleft()
        if next_proposal[i] >= K:
            continue
        j = int(proposer_prefs[i, next_proposal[i]])
        next_proposal[i] += 1

        held = int(current_match[j])
        if held == -1:
            current_match[j] = i
            proposer_match[i] = j
        elif receiver_rank[j, i] < receiver_rank[j, held]:
            current_match[j] = i
            proposer_match[i] = j
            proposer_match[held] = -1
            free.append(held)
        else:
            free.append(i)

    return proposer_match
