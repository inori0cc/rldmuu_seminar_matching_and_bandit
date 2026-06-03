"""Reproduce Figure 1a (Example 2): centralized UCB on GS Platform,
expected regret vs. Delta, horizon 400, 100 trials.
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
    horizon = 20 if smoke else cfg["horizon"]
    n_trials = 3 if smoke else cfg["n_trials"]
    deltas = np.linspace(
        cfg["delta_grid"]["start"],
        cfg["delta_grid"]["stop"],
        cfg["delta_grid"]["num"],
    )

    mean_regrets = np.zeros((len(deltas), 2))
    sem_regrets = np.zeros((len(deltas), 2))

    start_time = time.time()
    for d_idx, delta in enumerate(deltas):
        market = Market.from_example_2(delta=float(delta))

        def platform_factory(m: Market, _market=market) -> GSPlatform:
            return GSPlatform(m.N, m.K, m.arm_preferences)

        results = run_experiment(
            market_factory=lambda m=market: m,
            platform_factory=platform_factory,
            ranking_strategy="ucb",
            horizon=horizon,
            n_trials=n_trials,
            seed=cfg["seed"] + d_idx,
        )
        final = np.zeros((n_trials, 2))
        for trial_idx, res in enumerate(results):
            cum = agent_optimal_regret(
                market, res["matchings"], res["rewards"], use_expected=True
            )
            final[trial_idx] = cum[-1]
        mean_regrets[d_idx] = final.mean(axis=0)
        sem_regrets[d_idx] = final.std(axis=0, ddof=1) / np.sqrt(n_trials)
        print(
            f"  delta={delta:.3f}  mean_regret={mean_regrets[d_idx]}  "
            f"sem={sem_regrets[d_idx]}"
        )

    elapsed = time.time() - start_time
    out = ROOT / "data" / ("figure_1a_smoke.npz" if smoke else "figure_1a.npz")
    np.savez(
        out,
        deltas=deltas,
        mean_regrets=mean_regrets,
        sem_regrets=sem_regrets,
        horizon=horizon,
        n_trials=n_trials,
    )
    print(f"Figure 1a {'(smoke) ' if smoke else ''}saved to {out}; elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
