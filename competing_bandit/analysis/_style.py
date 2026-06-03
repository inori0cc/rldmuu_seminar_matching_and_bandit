"""Shared matplotlib style for Figure 1 plots."""
from __future__ import annotations

import matplotlib.pyplot as plt


def apply():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
            "figure.figsize": (4.5, 3.0),
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )
