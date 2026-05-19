"""Figures 1–4 from aggregated curves (spec §8)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CS = (0, 200, 500, 800, 1000)
ADVS = ("a2", "b")
ALGOS = ("vanilla", "robust")
COLORS = {"vanilla": "#377eb8", "robust": "#e41a1c"}
LINESTYLES = {"a2": "-", "b": "--"}


def _load(agg_dir: Path, algo: str, adv: str, C: int) -> dict:
    path = agg_dir / f"{algo}_{adv}_C{C}.npz"
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def fig1(agg_dir: Path, out_dir: Path) -> None:
    """One subplot per C, four curves each."""
    fig, axes = plt.subplots(1, len(CS), figsize=(4 * len(CS), 3.5), sharey=True)
    for ax, C in zip(axes, CS):
        for algo in ALGOS:
            for adv in ADVS:
                d = _load(agg_dir, algo, adv, C)
                label = f"{algo}+{adv}"
                ax.plot(
                    d["t_idx"], d["maxreg_mean"],
                    color=COLORS[algo], linestyle=LINESTYLES[adv], label=label,
                )
                ax.fill_between(
                    d["t_idx"],
                    d["maxreg_mean"] - d["maxreg_se"],
                    d["maxreg_mean"] + d["maxreg_se"],
                    color=COLORS[algo], alpha=0.15,
                )
        ax.set_title(f"C={C}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("MaxReg(t)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_main_regret.pdf")
    fig.savefig(out_dir / "fig1_main_regret.png", dpi=150)
    plt.close(fig)


def fig2(agg_dir: Path, out_dir: Path) -> None:
    """Robust GS-UCB at all C, two panels (A2, B)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(CS)))
    for ax, adv in zip(axes, ADVS):
        for color, C in zip(cmap, CS):
            d = _load(agg_dir, "robust", adv, C)
            ax.plot(d["t_idx"], d["maxreg_mean"], color=color, label=f"C={C}")
            ax.fill_between(
                d["t_idx"],
                d["maxreg_mean"] - d["maxreg_se"],
                d["maxreg_mean"] + d["maxreg_se"],
                color=color, alpha=0.15,
            )
        ax.set_title(f"Robust GS-UCB under {adv.upper()}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("MaxReg(t)")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_C_effect.pdf")
    fig.savefig(out_dir / "fig2_C_effect.png", dpi=150)
    plt.close(fig)


def fig3(agg_dir: Path, out_dir: Path) -> None:
    """Final MaxReg(T) vs C, four lines."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo in ALGOS:
        for adv in ADVS:
            ys = []
            es = []
            for C in CS:
                d = _load(agg_dir, algo, adv, C)
                ys.append(d["maxreg_mean"][-1])
                es.append(d["maxreg_se"][-1])
            ax.errorbar(
                CS, ys, yerr=es,
                color=COLORS[algo], linestyle=LINESTYLES[adv], marker="o",
                label=f"{algo}+{adv}",
            )
    ax.set_xlabel("per-player budget C")
    ax.set_ylabel("MaxReg(T)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_final_vs_C.pdf")
    fig.savefig(out_dir / "fig3_final_vs_C.png", dpi=150)
    plt.close(fig)


def fig4(agg_dir: Path, out_dir: Path) -> None:
    """C=0 clean sanity check: Robust must coincide with Vanilla."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, adv in zip(axes, ADVS):
        for algo in ALGOS:
            d = _load(agg_dir, algo, adv, 0)
            ax.plot(d["t_idx"], d["maxreg_mean"], color=COLORS[algo], linestyle=LINESTYLES[adv], label=algo)
            ax.fill_between(
                d["t_idx"],
                d["maxreg_mean"] - d["maxreg_se"],
                d["maxreg_mean"] + d["maxreg_se"],
                color=COLORS[algo], alpha=0.15,
            )
        ax.set_title(f"C=0 under {adv.upper()}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("MaxReg(t)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_clean_sanity.pdf")
    fig.savefig(out_dir / "fig4_clean_sanity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fig", type=int, choices=[1, 2, 3, 4, 0], default=0, help="0 = all")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.fig in (0, 1): fig1(args.inp, args.out)
    if args.fig in (0, 2): fig2(args.inp, args.out)
    if args.fig in (0, 3): fig3(args.inp, args.out)
    if args.fig in (0, 4): fig4(args.inp, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
