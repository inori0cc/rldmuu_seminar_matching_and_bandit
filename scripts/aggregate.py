"""Aggregate per-seed npz results into per-config curves (spec §9.5)."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ucb_corruption.io import load_trial
from ucb_corruption.metrics import (
    clean_pseudo_regret,
    downsample,
    max_regret,
    stable_match_rate,
)

NAME_RE = re.compile(r"algo=(?P<algo>\w+)__adv=(?P<adv>\w+)__C=(?P<C>\d+)__seed=(?P<seed>\d+)\.npz$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-points", type=int, default=200)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in sorted(args.inp.glob("*.npz")):
        m = NAME_RE.search(path.name)
        if not m:
            continue
        key = (m["algo"], m["adv"], int(m["C"]))
        groups[key].append(path)

    for (algo, adv, C), paths in sorted(groups.items()):
        maxreg_curves = []
        per_player_curves = []
        match_rates = []
        total_costs = []
        first_exhausts = []
        for path in paths:
            data = load_trial(path)
            R = clean_pseudo_regret(data["matchings"], data["mu"], data["m_star"])
            maxreg_curves.append(max_regret(R))
            per_player_curves.append(R)
            match_rates.append(stable_match_rate(data["matchings"], data["m_star"]))
            total_costs.append(data["total_cost"])
            first_exhausts.append(data["first_exhaust_round"])

        maxreg_stack = np.stack(maxreg_curves)             # (n_seeds, T)
        per_player_stack = np.stack(per_player_curves)     # (n_seeds, T, N)
        n = len(paths)

        idx, _ = downsample(maxreg_stack[0], n=args.n_points)
        mean = maxreg_stack.mean(axis=0)[idx]
        se = (maxreg_stack.std(axis=0, ddof=1) / np.sqrt(n))[idx] if n > 1 else np.zeros_like(mean)
        per_player_mean = per_player_stack.mean(axis=0)[idx]                # (n_points, N)
        per_player_se = (
            per_player_stack.std(axis=0, ddof=1)[idx] / np.sqrt(n)
            if n > 1 else np.zeros_like(per_player_mean)
        )

        out_path = args.out / f"{algo}_{adv}_C{C}.npz"
        np.savez(
            out_path,
            t_idx=idx,
            maxreg_mean=mean,
            maxreg_se=se,
            per_player_mean=per_player_mean,
            per_player_se=per_player_se,
            match_rate_mean=np.mean(match_rates),
            match_rate_se=(np.std(match_rates, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            total_cost_mean=np.mean(total_costs, axis=0),
            first_exhaust_mean=np.mean(first_exhausts, axis=0),
            n_seeds=n,
        )
        print(f"{algo:7s} {adv:3s} C={C:5d}  n={n:2d}  MaxReg(T)={mean[-1]:8.2f} ± {se[-1]:.2f}")


if __name__ == "__main__":
    main()
