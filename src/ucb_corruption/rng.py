"""Four-stream RNG bundle (spec §9.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RNGStreams:
    perm: np.random.Generator    # player-side permutations rho_i
    arm: np.random.Generator     # arm-side rankings pi_j
    reward: np.random.Generator  # Bernoulli reward sampling
    adv: np.random.Generator     # reserved for adversary randomness


def spawn_streams(master_seed: int) -> RNGStreams:
    ss = np.random.SeedSequence(master_seed)
    s_perm, s_arm, s_reward, s_adv = ss.spawn(4)
    return RNGStreams(
        perm=np.random.default_rng(s_perm),
        arm=np.random.default_rng(s_arm),
        reward=np.random.default_rng(s_reward),
        adv=np.random.default_rng(s_adv),
    )
