"""Plot Figure 1c: cumulative expected regret for 5 representative agents."""
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
    data = np.load(ROOT / "data" / "figure_1c.npz")
    mean = data["cum_regret_mean"]
    sem = data["cum_regret_sem"]
    agents = data["agents_to_show"].tolist()
    T = mean.shape[0]
    times = np.arange(1, T + 1)

    fig, ax = plt.subplots()
    for idx in agents:
        ax.plot(times, mean[:, idx], label=f"Agent {idx + 1}")
        ax.fill_between(
            times,
            mean[:, idx] - sem[:, idx],
            mean[:, idx] + sem[:, idx],
            alpha=0.15,
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Expected Regret")
    ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    out = ROOT / "figures" / "figure_1c.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
