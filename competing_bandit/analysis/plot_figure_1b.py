"""Plot Figure 1b: cumulative expected regret vs. time."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._style import apply


def main() -> None:
    apply()
    data = np.load(ROOT / "data" / "figure_1b.npz")
    mean = data["cum_regret_mean"]  # (T, N)
    sem = data["cum_regret_sem"]
    T, N = mean.shape
    times = np.arange(1, T + 1)

    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(times, mean[:, i], label=f"Agent {i + 1}")
        ax.fill_between(times, mean[:, i] - sem[:, i], mean[:, i] + sem[:, i], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Expected Regret")
    ax.legend()
    ax.grid(alpha=0.3)
    out = ROOT / "figures" / "figure_1b.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
