from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


def mean_and_ci(arr: np.ndarray, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    arr: shape (num_seeds, T)
    Returns mean, lower, upper using normal approx.
    """
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    se = std / np.sqrt(max(arr.shape[0], 1))

    # 95% normal approx multiplier ~ 1.96
    z = 1.96 if ci == 0.95 else 1.96
    lower = mean - z * se
    upper = mean + z * se
    return mean, lower, upper


def plot_regret_curves(
    curves: Dict[str, np.ndarray],
    save_path: str,
    title: str = "Cumulative Regret (mean ± 95% CI)",
) -> None:
    """
    curves: dict name -> array shape (num_seeds, T) (cumulative regret)
    """
    plt.figure()
    for name, arr in curves.items():
        mean, lo, hi = mean_and_ci(arr, ci=0.95)
        x = np.arange(1, mean.shape[0] + 1)
        plt.plot(x, mean, label=name)
        plt.fill_between(x, lo, hi, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("Cumulative regret")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_regret_curves_per_player(
    curves: Dict[str, np.ndarray],
    save_path: str,
    player_ids: list[int],
    title: str = "Per-player cumulative regret (mean ± 95% CI)",
) -> None:
    """
    curves: dict name -> array shape (num_seeds, T, N)
    player_ids: which players to plot (e.g., [0,1,2])
    """
    plt.figure()

    for algo_name, arr in curves.items():
        # arr shape: (S, T, N)
        for pid in player_ids:
            series = arr[:, :, pid]  # (S, T)
            mean, lo, hi = mean_and_ci(series, ci=0.95)
            x = np.arange(1, mean.shape[0] + 1)
            label = f"{algo_name} | p{pid+1}"
            plt.plot(x, mean, label=label)
            plt.fill_between(x, lo, hi, alpha=0.15)

    plt.xlabel("t")
    plt.ylabel("Cumulative regret")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_metric_curves(
    curves: Dict[str, np.ndarray],
    save_path: str,
    ylabel: str,
    title: str,
    xlabel: str = "t",
) -> None:
    """
    Generic mean ± 95% CI plot for time-dependent metrics.

    curves:
        dict name -> array of shape (num_seeds, T)

    This uses the same visual style as plot_regret_curves:
        - mean curve over seeds;
        - shaded 95% confidence interval;
        - one curve per algorithm.
    """
    plt.figure()

    for name, arr in curves.items():
        mean, lo, hi = mean_and_ci(arr, ci=0.95)
        x = np.arange(1, mean.shape[0] + 1)

        plt.plot(x, mean, label=name)
        plt.fill_between(x, lo, hi, alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()