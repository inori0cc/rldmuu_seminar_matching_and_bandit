"""RNG management utilities.

All randomness in this project flows through numpy.random.Generator instances
created via SeedSequence, so trials are independently reproducible and never
touch the legacy global np.random state.
"""
from __future__ import annotations

import numpy as np


def make_rng(seed: int | np.random.SeedSequence) -> np.random.Generator:
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    return np.random.default_rng(np.random.SeedSequence(seed))


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n)
    return [np.random.default_rng(c) for c in children]
