from __future__ import annotations

import os
import numpy as np


RESULTS_DIR = "results"


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())

    if values.size <= 1:
        return mean, 0.0

    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, 1.96 * se


def fmt(mean: float, ci: float) -> str:
    return f"{mean:.2f} ± {ci:.2f}"


def add_line(lines: list[str], line: str = "") -> None:
    print(line)
    lines.append(line)


def summarize_gap_sweep(lines: list[str]) -> None:
    path = os.path.join(RESULTS_DIR, "arm_gap_sweep.npz")
    data = np.load(path)

    gaps = data["arm_gaps"]

    known = data["known_arm_ucbgs_binary_stable_regret"]
    two_sided = data["two_sided_ucbgs_binary_stable_regret"]
    etc = data["two_sided_etc_binary_stable_regret"]

    add_line(lines, "# UCB gap sweep: binary stable regret")
    add_line(lines)
    add_line(
        lines,
        "| Arm gap | Known-arm UCB+GS | Two-sided UCB+GS | Two-sided ETC | Relative increase UCB |",
    )
    add_line(lines, "|---:|---:|---:|---:|---:|")

    for idx, gap in enumerate(gaps):
        known_mean, known_ci = mean_ci(known[idx])
        two_mean, two_ci = mean_ci(two_sided[idx])
        etc_mean, etc_ci = mean_ci(etc[idx])

        relative = two_mean / known_mean if known_mean != 0 else np.nan

        add_line(
            lines,
            f"| {gap:.3f} | {fmt(known_mean, known_ci)} | "
            f"{fmt(two_mean, two_ci)} | {fmt(etc_mean, etc_ci)} | "
            f"{relative:.2f}x |",
        )

    add_line(lines)


def summarize_blocking_pairs(lines: list[str]) -> None:
    path = os.path.join(RESULTS_DIR, "arm_gap_sweep.npz")
    data = np.load(path)

    gaps = data["arm_gaps"]

    known = data["known_arm_ucbgs_cumulative_blocking_pairs"]
    two_sided = data["two_sided_ucbgs_cumulative_blocking_pairs"]

    add_line(lines, "# UCB gap sweep: cumulative blocking pairs")
    add_line(lines)
    add_line(lines, "| Arm gap | Known-arm UCB+GS | Two-sided UCB+GS | Relative increase |")
    add_line(lines, "|---:|---:|---:|---:|")

    for idx, gap in enumerate(gaps):
        known_mean, known_ci = mean_ci(known[idx])
        two_mean, two_ci = mean_ci(two_sided[idx])
        relative = two_mean / known_mean if known_mean != 0 else np.nan

        add_line(
            lines,
            f"| {gap:.3f} | {fmt(known_mean, known_ci)} | "
            f"{fmt(two_mean, two_ci)} | {relative:.2f}x |",
        )

    add_line(lines)


def summarize_etc_commit_rate(lines: list[str]) -> None:
    path = os.path.join(RESULTS_DIR, "etc_h_sweep.npz")
    data = np.load(path)

    gaps = data["gaps"]
    h_values = data["h_values"]
    stable_commit = data["stable_commit"]

    add_line(lines, "# Two-sided ETC: stable commitment rate")
    add_line(lines)

    header = "| Arm gap | " + " | ".join([f"h={h}" for h in h_values]) + " |"
    sep = "|---:|" + "|".join(["---:" for _ in h_values]) + "|"

    add_line(lines, header)
    add_line(lines, sep)

    for g_idx, gap in enumerate(gaps):
        row_values = []

        for h_idx in range(len(h_values)):
            mean, ci = mean_ci(stable_commit[g_idx, h_idx])
            row_values.append(fmt(mean, ci))

        add_line(lines, f"| {gap:.3f} | " + " | ".join(row_values) + " |")

    add_line(lines)


def summarize_etc_best_h(lines: list[str]) -> None:
    path = os.path.join(RESULTS_DIR, "etc_h_sweep.npz")
    data = np.load(path)

    gaps = data["gaps"]
    h_values = data["h_values"]

    binary = data["binary_stable_regret"]
    stable_commit = data["stable_commit"]

    add_line(lines, "# Two-sided ETC: best exploration budgets")
    add_line(lines)
    add_line(
        lines,
        "| Arm gap | Best h by binary stable regret | Binary stable regret | Best h by stable commit rate | Stable commit rate |",
    )
    add_line(lines, "|---:|---:|---:|---:|---:|")

    for g_idx, gap in enumerate(gaps):
        binary_means = binary[g_idx].mean(axis=1)
        commit_means = stable_commit[g_idx].mean(axis=1)

        best_binary_idx = int(np.argmin(binary_means))
        best_commit_idx = int(np.argmax(commit_means))

        best_binary_h = int(h_values[best_binary_idx])
        best_commit_h = int(h_values[best_commit_idx])

        binary_mean, binary_ci = mean_ci(binary[g_idx, best_binary_idx])
        commit_mean, commit_ci = mean_ci(stable_commit[g_idx, best_commit_idx])

        add_line(
            lines,
            f"| {gap:.3f} | {best_binary_h} | {fmt(binary_mean, binary_ci)} | "
            f"{best_commit_h} | {fmt(commit_mean, commit_ci)} |",
        )

    add_line(lines)


def main() -> None:
    lines: list[str] = []

    summarize_gap_sweep(lines)
    summarize_blocking_pairs(lines)
    summarize_etc_commit_rate(lines)
    summarize_etc_best_h(lines)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_path = os.path.join(RESULTS_DIR, "final_summary_tables.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved summary tables to: {out_path}")


if __name__ == "__main__":
    main()