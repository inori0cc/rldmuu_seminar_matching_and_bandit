"""Generate a runs.tsv for the 200-run experimental grid (spec §7).

Grid: algos {vanilla, robust} × adversaries {a2, b} × C {0, 200, 500, 800, 1000}
       × 10 seeds = 200 rows.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

ALGOS = ("vanilla", "robust")
ADVS = ("a2", "b")
CS = (0, 200, 500, 800, 1000)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", type=Path, default=Path("runs.tsv"))
    p.add_argument("--T", type=int, default=100_000)
    p.add_argument("--out", type=Path, default=Path("results/"))
    p.add_argument("--n-seeds", type=int, default=10,
                   help="number of seeds per (algo, adv, C); seed indices 0..n-1")
    args = p.parse_args()

    seeds = tuple(range(args.n_seeds))
    rows = []
    for algo, adv, C, seed in product(ALGOS, ADVS, CS, seeds):
        rows.append(
            f"{algo}\t{adv}\t{C}\t{args.T}\t{seed}\t{args.out}"
        )
    args.tsv.write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} rows to {args.tsv}")


if __name__ == "__main__":
    main()
