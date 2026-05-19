"""GS stability on a small hand-checked instance."""
from __future__ import annotations

import numpy as np

from ucb_corruption.gale_shapley import gale_shapley


def is_stable(matching, proposer_prefs, receiver_prefs):
    N, K = proposer_prefs.shape
    # Build receiver rank lookup
    rrank = np.full((K, N), N + 1, dtype=int)
    for j in range(K):
        for r, i in enumerate(receiver_prefs[j]):
            rrank[j, i] = r
    prank = np.full((N, K), K + 1, dtype=int)
    for i in range(N):
        for r, j in enumerate(proposer_prefs[i]):
            prank[i, j] = r
    # Check no blocking pair
    for i in range(N):
        for j in range(K):
            cur_j = matching[i]
            # i prefers j over its match
            if cur_j == -1 or prank[i, j] < prank[i, cur_j]:
                # j prefers i over its current holder
                held = -1
                for ip in range(N):
                    if matching[ip] == j:
                        held = ip
                        break
                if held == -1 or rrank[j, i] < rrank[j, held]:
                    return False
    return True


def test_gs_5x5_known_case():
    # Hand example: all proposers prefer arm 0; arm 0 prefers proposer 0.
    proposer_prefs = np.array(
        [[0, 1, 2, 3, 4]] * 5,
        dtype=np.int64,
    )
    # Each arm j prefers proposer j first, then ascending
    receiver_prefs = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 0, 1, 3, 4],
            [3, 0, 1, 2, 4],
            [4, 0, 1, 2, 3],
        ],
        dtype=np.int64,
    )
    m = gale_shapley(proposer_prefs, receiver_prefs)
    # Proposer 0 gets arm 0 (its top, and arm 0 wants it).
    assert m[0] == 0
    assert is_stable(m, proposer_prefs, receiver_prefs)


def test_gs_random_5x5_stability():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.stack([rng.permutation(5) for _ in range(5)]).astype(np.int64)
        r = np.stack([rng.permutation(5) for _ in range(5)]).astype(np.int64)
        m = gale_shapley(p, r)
        assert m.shape == (5,)
        assert np.all(m >= 0)
        assert len(set(m.tolist())) == 5
        assert is_stable(m, p, r)
