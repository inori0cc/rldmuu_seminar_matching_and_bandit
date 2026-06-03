"""Reproduce Figure 1b (Example 6): centralized UCB on GS Platform,
cumulative agent-optimal regret vs. time, horizon 200, 100 trials.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market import Market
from src.platforms import GSPlatform
from src.regret import agent_optimal_regret
from src.simulator import run_experiment


def main(config_path: str, smoke: bool) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    horizon = 50 if smoke else cfg["horizon"]
    n_trials = 5 if smoke else cfg["n_trials"]

    market = Market.from_example_6(p3_gap=cfg["p3_gap"])

    def platform_factory(m: Market) -> GSPlatform:
        return GSPlatform(m.N, m.K, m.arm_preferences)

    start_time = time.time()
    results = run_experiment(
        market_factory=lambda m=market: m,
        platform_factory=platform_factory,
        ranking_strategy="ucb",
        horizon=horizon,
        n_trials=n_trials,
        seed=cfg["seed"],
        progress=True,
    )

    N = market.N
    all_cum = np.zeros((n_trials, horizon, N))
    for trial_idx, res in enumerate(results):
        all_cum[trial_idx] = agent_optimal_regret(
            market, res["matchings"], res["rewards"], use_expected=True
        )
    mean = all_cum.mean(axis=0)
    sem = all_cum.std(axis=0, ddof=1) / np.sqrt(n_trials)

    elapsed = time.time() - start_time
    out = ROOT / "data" / ("figure_1b_smoke.npz" if smoke else "figure_1b.npz")
    np.savez(
        out,
        cum_regret_mean=mean,
        cum_regret_sem=sem,
        horizon=horizon,
        n_trials=n_trials,
    )
    print(
        f"Figure 1b {'(smoke) ' if smoke else ''}saved to {out}; "
        f"final regrets = {mean[-1]}; elapsed {elapsed:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
