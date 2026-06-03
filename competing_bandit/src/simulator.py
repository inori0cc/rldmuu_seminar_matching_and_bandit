"""Single-trial and multi-trial drivers."""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.agents import EmpiricalMeanTracker, rank_by_empirical_mean, rank_by_ucb
from src.market import Market


def run_trial(
    market: Market,
    platform,
    ranking_strategy: str,
    horizon: int,
    rng: np.random.Generator,
) -> dict:
    N, K = market.N, market.K
    trackers = [EmpiricalMeanTracker(K) for _ in range(N)]

    matchings = np.zeros((horizon, N), dtype=np.int64)
    rewards = np.zeros((horizon, N), dtype=np.float64)
    successful = np.zeros((horizon, N), dtype=bool)

    for t_py in range(horizon):
        t = t_py + 1

        rankings = np.zeros((N, K), dtype=np.int64)
        for i in range(N):
            if ranking_strategy == "ucb":
                rankings[i] = rank_by_ucb(trackers[i], t, rng)
            elif ranking_strategy == "empirical_mean":
                rankings[i] = rank_by_empirical_mean(trackers[i], rng)
            else:
                raise ValueError(f"unknown ranking_strategy: {ranking_strategy}")

        assignment = platform.assign(t, rankings)
        matchings[t_py] = assignment

        for i in range(N):
            arm = int(assignment[i])
            if arm < 0:
                continue
            r = market.sample_reward(i, arm, rng)
            rewards[t_py, i] = r
            successful[t_py, i] = True
            trackers[i].update(arm, r)

    return {
        "matchings": matchings,
        "rewards": rewards,
        "successful": successful,
    }


def run_experiment(
    market_factory: Callable[[], Market],
    platform_factory: Callable[[Market], object],
    ranking_strategy: str,
    horizon: int,
    n_trials: int,
    seed: int,
    progress: bool = False,
) -> list[dict]:
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_trials)

    iterator = range(n_trials)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=n_trials, leave=False)
        except ImportError:
            pass

    results = []
    for trial_idx in iterator:
        rng = np.random.default_rng(child_seeds[trial_idx])
        market = market_factory()
        platform = platform_factory(market)
        results.append(run_trial(market, platform, ranking_strategy, horizon, rng))
    return results
