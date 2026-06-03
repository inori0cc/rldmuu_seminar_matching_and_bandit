"""Plot Figure 1a: expected regret vs. Delta, log y-axis."""
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
    data = np.load(ROOT / "data" / "figure_1a.npz")
    deltas = data["deltas"]
    mean = data["mean_regrets"]
    sem = data["sem_regrets"]

    fig, ax = plt.subplots()
    for i, (label, marker) in enumerate(zip(["Agent 1", "Agent 2"], ["o", "s"])):
        ax.errorbar(
            deltas,
            mean[:, i],
            yerr=sem[:, i],
            label=label,
            marker=marker,
            capsize=2,
            markersize=4,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Reward Gap of Agent 1 ($\Delta$)")
    ax.set_ylabel("Expected Regret")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    out = ROOT / "figures" / "figure_1a.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
