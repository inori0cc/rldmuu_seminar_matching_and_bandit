"""Run one (algo, adversary, C, seed) trial and save an npz.

Usage:
    python -m scripts.run_single --algo mom --adversary a2 --C 500 \
        --T 100000 --seed 0 --out results/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ucb_corruption.adversaries import A2, B, NoOp
from ucb_corruption.adversaries import C as AdvC
from ucb_corruption.algorithms import MoMGSUCB, VanillaUCB
from ucb_corruption.io import save_trial
from ucb_corruption.market import Market
from ucb_corruption.rng import spawn_streams
from ucb_corruption.simulator import run_trial


def build_algorithms(name: str, N: int, K: int, alpha: float,
                     T: int | None = None,
                     alpha_mom: float = 8.0,
                     n_min_for_mom: int = 50):
    if name == "vanilla":
        return [VanillaUCB(K=K, alpha=alpha) for _ in range(N)]
    if name == "mom":
        if T is None:
            raise ValueError("MoMGSUCB requires T")
        return [
            MoMGSUCB(K=K, T=T, alpha_MoM=alpha_mom, n_min_for_mom=n_min_for_mom)
            for _ in range(N)
        ]
    raise ValueError(f"unknown algorithm: {name}")


def build_adversary(name: str, N: int, K: int, C: float, eps: float, market: Market):
    if name == "noop":
        return NoOp(N=N, K=K)
    if name == "a2":
        return A2(N=N, K=K, C=C, eps=eps, m_star=market.m_star, mu=market.mu)
    if name == "b":
        return B(N=N, K=K, C=C, eps=eps, m_star=market.m_star)
    if name == "c":
        return AdvC(N=N, K=K, C=C, eps=eps, m_star=market.m_star, mu=market.mu)
    raise ValueError(f"unknown adversary: {name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["vanilla", "mom"], required=True)
    p.add_argument("--adversary", choices=["noop", "a2", "b", "c"], required=True)
    p.add_argument("--C", type=float, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--delta", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--sigma", type=float, default=1.0,
                   help="std-dev of clipped Gaussian reward noise (Wu et al. 2025: 1.0)")
    p.add_argument("--alpha-mom", type=float, default=8.0,
                   help="MoM-UCB coefficient (radius = sqrt(alpha_mom * k * log t / N))")
    p.add_argument("--n-min-for-mom", type=int, default=3,
                   help="MoMGSUCB cold-start cutoff: use sample mean while N < this")
    args = p.parse_args()

    streams = spawn_streams(args.seed)
    market = Market.build(
        N=args.N, K=args.K, delta=args.delta,
        rng_perm=streams.perm, rng_arm=streams.arm,
        sigma=args.sigma,
    )
    algorithms = build_algorithms(
        args.algo, N=args.N, K=args.K,
        alpha=args.alpha,
        T=args.T, alpha_mom=args.alpha_mom,
        n_min_for_mom=args.n_min_for_mom,
    )
    adversary = build_adversary(args.adversary, N=args.N, K=args.K, C=args.C, eps=args.eps, market=market)

    result = run_trial(
        market=market,
        algorithms=algorithms,
        adversary=adversary,
        T=args.T,
        rng_reward=streams.reward,
        C=args.C,
        eps=args.eps,
    )

    path = save_trial(
        result, args.out,
        algo=args.algo, adversary=args.adversary,
        C=args.C, seed=args.seed,
        alpha=args.alpha, eps=args.eps,
        sigma=args.sigma,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
