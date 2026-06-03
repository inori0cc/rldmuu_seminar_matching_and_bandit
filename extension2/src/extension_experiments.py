"""
Extension experiments for the two-sided learning matching-market project.

This script runs the experiments used to study the question:

    How much harder is it to reach stable matchings when both sides of the
    market must learn their preferences?

Available experiments:
    fixed   Run a fixed-gap comparison between:
            - known-arm UCB+GS
            - two-sided UCB+GS
            - two-sided ETC

    gap     Sweep the arm preference gap and compare how stability changes
            when arm-side preferences become harder to learn.

    time    Plot cumulative binary stable regret over time for:
            - known-arm UCB+GS
            - two-sided UCB+GS
            - two-sided ETC

    etc     Sweep the exploration budget h for two-sided ETC and measure
            how often the algorithm commits to a truly stable matching.

    random
        Generate randomized preference markets at different arm gaps             

    random-time
        Generate randomized preference markets at gap 0.10 and plot
        cumulative binary stable regret over time.
            
    all     Run all experiments above.

Typical usage from the project root:

    python -m src.extension_experiments --experiment all

Run only the arm-gap sweep:

    python -m src.extension_experiments --experiment gap

Run only the stability-over-time plot:

    python -m src.extension_experiments --experiment time

Run only the ETC exploration-budget sweep:

    python -m src.extension_experiments --experiment etc

Run only the randomized stability-over-time robustness plot:

    python -m src.extension_experiments --experiment random-time

Quick test with fewer seeds and a shorter horizon:

    python -m src.extension_experiments --experiment gap --T 100 --num-seeds 5

Results are saved by default in:

    results/

The generated .npz files can then be summarized with:

    python -m src.summarize_extension_results
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

from .market import (
    MarketInstance,
    generate_handcrafted_instance,
    arm_rankings_from_nu,
)
from .algorithms import run_ucb_gs, run_two_sided_ucb_gs, run_two_sided_etc
from .metrics import (
    stability_curves,
    stable_matching_player_optimal,
    regret_vs_matching,
    is_stable_matching,
    count_blocking_pairs,
)
from .plotting import plot_metric_curves


@dataclass
class ExtensionConfig:
    """
    Central configuration for the extension experiments.

    The experiments are intentionally kept small and reproducible:
    - N = K = 3 is defined inside the handcrafted market instance;
    - T controls the horizon;
    - seeds controls the number of repeated trials;
    - arm_gaps controls how hard arm-side preferences are to learn.

    The main instance is "conflict_arm_gap_3x3":
        all players want the same arms,
        all arms prefer the same players,
        and the arm preference gap controls the difficulty of learning
        the true arm rankings.
    """
    instance_name: str = "conflict_arm_gap_3x3"
    
    T: int = 500
    seeds: List[int] = field(default_factory=lambda: list(range(50)))

    reward_model: str = "bernoulli"
    reward_sigma: float = 1.0

    # ETC samples per player-arm pair
    etc_h: int = 10

    # Fixed-gap comparison
    fixed_gap: float = 0.10

    # Stability-over-time plot
    time_gap: float = 0.10

    # Arm-side preference gap sweep
    arm_gaps: List[float] = field(
        default_factory=lambda: [0.30, 0.20, 0.15, 0.10, 0.07, 0.05]
    )

    # Randomized robustness experiment
    random_trials: int = 50
    random_market_gaps: List[float] = field(
        default_factory=lambda: [0.30, 0.20, 0.15, 0.10, 0.07, 0.05]
    )

    # Randomized stability-over-time robustness plot
    random_time_gap: float = 0.10
    random_time_trials: int = 50

    # ETC exploration-budget sweep
    etc_h_values: List[int] = field(default_factory=lambda: [2, 5, 10, 20, 40, 80])
    etc_h_sweep_gaps: List[float] = field(default_factory=lambda: [0.30, 0.10, 0.05])

    results_dir: str = "results"


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    """
    Mean and 95% CI half-width for a 1D array.
    """
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())

    if values.size <= 1:
        return mean, 0.0

    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, 1.96 * se


def _generate_random_preference_instance(
    rng: np.random.Generator,
    arm_gap: float,
    player_gap: float = 0.20,
    N: int = 3,
    K: int = 3,
) -> MarketInstance:
    """
    Generate a fully randomized 3x3 preference market.

    Unlike the conflict-heavy generator, this version randomizes the preference
    order itself.

    For each player:
        we sample a random ranking of arms.

    For each arm:
        we sample a random ranking of players.

    Then we convert those rankings into Bernoulli utility values.

    Why this is useful:
        It is a broader robustness check. It tests whether two-sided learning
        is still harder when the market is not manually designed to be
        conflict-heavy.

    Important interpretation:
        The effect may be weaker than in the conflict-heavy experiment, because
        random markets are sometimes naturally easier. For example, players may
        prefer different arms, creating less competition.
    """
    if N != 3 or K != 3:
        raise ValueError("This helper currently assumes N=K=3.")

    if not (0.0 < arm_gap < 0.45):
        raise ValueError("arm_gap should be in (0, 0.45).")

    if not (0.0 < player_gap < 0.30):
        raise ValueError("player_gap should be in (0, 0.30).")

    # -------------------------
    # Player-side utilities mu
    # -------------------------
    # For each player, draw a random strict ranking of arms.
    # Example: [2, 0, 1] means this player prefers a3 > a1 > a2.
    #
    # We then assign utility levels according to that ranking.
    # player_gap is kept fixed because in this experiment we mainly want
    # the x-axis to control the arm-side learning difficulty.
    mu = np.zeros((N, K), dtype=float)

    for i in range(N):
        ranking = rng.permutation(K)
        center = rng.uniform(0.45, 0.55)

        values = np.array([
            center + player_gap,
            center,
            center - player_gap,
        ])

        for pos, arm in enumerate(ranking):
            mu[i, arm] = values[pos]

    # -------------------------
    # Arm-side utilities nu
    # -------------------------
    # For each arm, draw a random strict ranking of players.
    # Example: [1, 2, 0] means this arm prefers p2 > p3 > p1.
    #
    # arm_gap is the experimental variable:
    # smaller arm_gap = harder to learn the correct arm-side rankings.
    nu = np.zeros((K, N), dtype=float)

    for j in range(K):
        ranking = rng.permutation(N)
        center = rng.uniform(0.45, 0.55)

        values = np.array([
            center + arm_gap,
            center,
            center - arm_gap,
        ])

        for pos, player in enumerate(ranking):
            nu[j, player] = values[pos]

    # Safety for Bernoulli rewards.
    mu = np.clip(mu, 0.01, 0.99)
    nu = np.clip(nu, 0.01, 0.99)

    arm_prefs = arm_rankings_from_nu(nu)

    return MarketInstance(mu=mu, nu=nu, arm_prefs=arm_prefs)


def _final_player_optimal_regret(instance, match_history) -> float:
    """
    Final summed regret against the player-optimal true stable matching.
    """
    comp_opt = stable_matching_player_optimal(instance.mu, instance.arm_prefs)
    _, cum = regret_vs_matching(instance.mu, match_history, comp_opt)
    return float(cum[-1])


def _summarize_run(instance, match_history) -> Dict[str, float]:
    """
    Compute the main evaluation metrics for one algorithm run.

    Important distinction:
        The algorithm may use estimated preferences while running,
        but all metrics here are computed using the TRUE preferences
        stored in the instance.

    This is why binary stable regret is meaningful:
        it checks whether the produced matchings are truly stable,
        not just stable according to the current estimates.
    """
    # Stability-based metrics:
    # - cum_binary_stable_regret counts unstable rounds.
    # - cum_blocking_pairs counts how many blocking-pair violations occurred.
    stability = stability_curves(instance.mu, instance.arm_prefs, match_history)

    final_binary = float(stability["cum_binary_stable_regret"][-1])
    final_blocking = float(stability["cum_blocking_pairs"][-1])

    # Reward regret is kept for comparison with the paper.
    # In our conflict-heavy instance this can be zero even when matchings
    # are unstable, which motivates using stability-specific metrics.
    final_reward_regret = _final_player_optimal_regret(instance, match_history)

    # If the algorithm never becomes permanently stable, we store T+1.
    # This makes the value numeric and easy to average over seeds.
    permanent = stability["time_to_permanent_stability"]
    if permanent is None:
        permanent_value = len(match_history) + 1
    else:
        permanent_value = float(permanent)

    return {
        "binary_stable_regret": final_binary,
        "cumulative_blocking_pairs": final_blocking,
        "player_optimal_reward_regret": final_reward_regret,
        "time_to_permanent_stability": permanent_value,
    }


def _run_algorithms_on_instance(cfg: ExtensionConfig, instance, seed: int) -> Dict[str, Dict[str, float]]:
    """
    Run the three algorithms on the same instance and seed.

    This makes the comparison fair:
        all algorithms face the same true market and the same random seed.

    Algorithms compared:
        Known-arm UCB+GS:
            Paper-style baseline. Players learn mu, but arm preferences are known.

        Two-sided UCB+GS:
            Extension algorithm. Players learn mu and arms learn nu online.

        Two-sided ETC:
            Extension baseline. Both sides explore first, then commit.
    """
    outputs: Dict[str, Any] = {}

    # Baseline: only players learn; arm preferences are true/known.
    rng = np.random.default_rng(seed)
    outputs["Known-arm UCB+GS"] = run_ucb_gs(
        instance,
        T=cfg.T,
        rng=rng,
        reward_model=cfg.reward_model,
        reward_sigma=cfg.reward_sigma,
    )

    # Main extension: both sides learn continuously with UCB.
    rng = np.random.default_rng(seed)
    outputs["Two-sided UCB+GS"] = run_two_sided_ucb_gs(
        instance,
        T=cfg.T,
        rng=rng,
        reward_model=cfg.reward_model,
        reward_sigma=cfg.reward_sigma,
    )

    # Controlled extension baseline: forced exploration, then commitment.
    rng = np.random.default_rng(seed)
    outputs["Two-sided ETC"] = run_two_sided_etc(
        instance,
        T=cfg.T,
        rng=rng,
        h=cfg.etc_h,
        reward_model=cfg.reward_model,
        reward_sigma=cfg.reward_sigma,
    )

    # Convert raw match histories into compact metric summaries.
    summaries = {
        algo_name: _summarize_run(instance, out.match_history)
        for algo_name, out in outputs.items()
    }

    return summaries


def run_fixed_gap_comparison(cfg: ExtensionConfig) -> None:
    """
    Run one direct comparison at a fixed arm preference gap.

    Purpose:
        Give a simple first answer to the question:
        "Does two-sided learning create more instability than the original
        known-arm-preference setting?"

    This experiment is useful for quick inspection, but the stronger result
    comes from run_arm_gap_sweep, where the difficulty of arm-side learning
    is varied systematically.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    algo_names = ["Known-arm UCB+GS", "Two-sided UCB+GS", "Two-sided ETC"]
    metric_names = [
        "binary_stable_regret",
        "cumulative_blocking_pairs",
        "player_optimal_reward_regret",
        "time_to_permanent_stability",
    ]

    results = {
        algo: {metric: [] for metric in metric_names}
        for algo in algo_names
    }

    # Build the same true market for every seed.
    # Only the stochastic reward samples change across seeds.
    instance = generate_handcrafted_instance(cfg.instance_name, delta=cfg.fixed_gap)

    print("\n=== Fixed-gap comparison ===")
    print(f"instance={cfg.instance_name}, gap={cfg.fixed_gap}, T={cfg.T}, seeds={len(cfg.seeds)}")

    for idx, seed in enumerate(cfg.seeds, start=1):
        summaries = _run_algorithms_on_instance(cfg, instance, seed)

        for algo in algo_names:
            for metric in metric_names:
                results[algo][metric].append(summaries[algo][metric])

        print(f"Seed {idx}/{len(cfg.seeds)} done")

    # Save raw values
    save_dict = {}
    for algo in algo_names:
        key_prefix = algo.lower().replace(" ", "_").replace("+", "").replace("-", "_")
        for metric in metric_names:
            save_dict[f"{key_prefix}_{metric}"] = np.asarray(results[algo][metric], dtype=float)

    np.savez(
        os.path.join(cfg.results_dir, "fixed_gap_comparison.npz"),
        fixed_gap=np.asarray([cfg.fixed_gap]),
        seeds=np.asarray(cfg.seeds),
        **save_dict,
    )

    # Print table
    print("\nFixed-gap summary:")
    for metric in metric_names:
        print(f"\n{metric}")
        for algo in algo_names:
            mean, ci = _mean_ci(np.asarray(results[algo][metric]))
            print(f"  {algo:18s}: {mean:.2f} ± {ci:.2f}")

    # Plot main metrics
    for metric in metric_names:
        means = []
        cis = []

        for algo in algo_names:
            mean, ci = _mean_ci(np.asarray(results[algo][metric]))
            means.append(mean)
            cis.append(ci)

        plt.figure(figsize=(7, 4))
        x = np.arange(len(algo_names))
        plt.bar(x, means, yerr=cis, capsize=4)
        plt.xticks(x, algo_names, rotation=20, ha="right")
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"Fixed gap = {cfg.fixed_gap}")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.results_dir, f"fixed_gap_{metric}.png"), dpi=200)
        plt.close()

    print(f"\nSaved fixed-gap results in {cfg.results_dir}")


def run_stability_over_time(cfg: ExtensionConfig) -> None:
    """
    Plot cumulative binary stable regret over time.

    Purpose:
        This is the dynamic version of the stability analysis.
        It is similar in spirit to the paper's regret-over-time plots,
        but uses binary stable regret instead of reward regret.

    Why binary stable regret?
        In the two-sided extension, reward regret can miss unstable matchings.
        Binary stable regret directly counts how often the market output is
        unstable under the true preferences.

    Output:
        A time curve for:
            - Known-arm UCB+GS;
            - Two-sided UCB+GS;
            - Two-sided ETC.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    algo_names = ["Known-arm UCB+GS", "Two-sided UCB+GS", "Two-sided ETC"]

    # Shape: (num_seeds, T) for each algorithm.
    # Each entry is cumulative binary stable regret at time t.
    curves = {
        algo: np.zeros((len(cfg.seeds), cfg.T), dtype=float)
        for algo in algo_names
    }

    instance = generate_handcrafted_instance(cfg.instance_name, delta=cfg.time_gap)

    print("\n=== Stability-over-time plot ===")
    print(f"instance={cfg.instance_name}, gap={cfg.time_gap}, T={cfg.T}, seeds={len(cfg.seeds)}")

    for s_idx, seed in enumerate(cfg.seeds):
        rng = np.random.default_rng(seed)
        out_known = run_ucb_gs(
            instance,
            T=cfg.T,
            rng=rng,
            reward_model=cfg.reward_model,
            reward_sigma=cfg.reward_sigma,
        )

        rng = np.random.default_rng(seed)
        out_two_sided_ucb = run_two_sided_ucb_gs(
            instance,
            T=cfg.T,
            rng=rng,
            reward_model=cfg.reward_model,
            reward_sigma=cfg.reward_sigma,
        )

        rng = np.random.default_rng(seed)
        out_two_sided_etc = run_two_sided_etc(
            instance,
            T=cfg.T,
            rng=rng,
            h=cfg.etc_h,
            reward_model=cfg.reward_model,
            reward_sigma=cfg.reward_sigma,
        )

        outputs = {
            "Known-arm UCB+GS": out_known,
            "Two-sided UCB+GS": out_two_sided_ucb,
            "Two-sided ETC": out_two_sided_etc,
        }

        for algo_name, out in outputs.items():
            stability = stability_curves(instance.mu, instance.arm_prefs, out.match_history)
            curves[algo_name][s_idx] = stability["cum_binary_stable_regret"]

        print(f"Seed {s_idx + 1}/{len(cfg.seeds)} done")

    # Save raw curves for reproducibility.
    save_dict = {
        "time_gap": np.asarray([cfg.time_gap], dtype=float),
        "seeds": np.asarray(cfg.seeds, dtype=int),
    }

    for algo_name, arr in curves.items():
        key = algo_name.lower().replace(" ", "_").replace("+", "").replace("-", "_")
        save_dict[f"{key}_cum_binary_stable_regret"] = arr

    np.savez(
        os.path.join(cfg.results_dir, "stability_over_time.npz"),
        **save_dict,
    )

    plot_metric_curves(
        curves=curves,
        save_path=os.path.join(cfg.results_dir, "stability_over_time_binary_regret.png"),
        ylabel="Cumulative binary stable regret",
        title=f"Binary stable regret over time (gap={cfg.time_gap})",
        xlabel="t",
    )

    print(f"\nSaved stability-over-time results in {cfg.results_dir}")
    print("Main files:")
    print(" - stability_over_time.npz")
    print(" - stability_over_time_binary_regret.png")


def run_arm_gap_sweep(cfg: ExtensionConfig) -> None:
    """
    Main UCB experiment: sweep the arm preference gap.

    Smaller arm gap means that arms receive more similar rewards from
    different players. Therefore, arm-side rankings are harder to learn.

    Expected behavior:
        - Known-arm UCB+GS should be almost flat across gaps, because it
          receives true arm preferences.
        - Two-sided UCB+GS should become less stable as the gap decreases,
          because it must learn arm preferences from noisy observations.
        - Two-sided ETC with fixed h may fail when h is too small for the gap.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    algo_names = ["Known-arm UCB+GS", "Two-sided UCB+GS", "Two-sided ETC"]
    metric_names = [
        "binary_stable_regret",
        "cumulative_blocking_pairs",
        "player_optimal_reward_regret",
        "time_to_permanent_stability",
    ]

    # results[metric][algo] -> shape (num_gaps, num_seeds)
    results = {
        metric: {
            algo: np.zeros((len(cfg.arm_gaps), len(cfg.seeds)), dtype=float)
            for algo in algo_names
        }
        for metric in metric_names
    }

    print("\n=== Arm-gap sweep ===")
    print(f"T={cfg.T}, seeds={len(cfg.seeds)}")
    print("gaps:", cfg.arm_gaps)

    # For each gap, we regenerate the same structural market but change
    # how hard it is for arms to distinguish players.
    for g_idx, gap in enumerate(cfg.arm_gaps):
        instance = generate_handcrafted_instance(cfg.instance_name, delta=gap)
        print(f"\nGap {gap:.3f} ({g_idx + 1}/{len(cfg.arm_gaps)})")

        # Each seed corresponds to a different stochastic realization
        # of the same true market.
        for s_idx, seed in enumerate(cfg.seeds):
            summaries = _run_algorithms_on_instance(cfg, instance, seed)

            for algo in algo_names:
                for metric in metric_names:
                    results[metric][algo][g_idx, s_idx] = summaries[algo][metric]

            print(f"  Seed {s_idx + 1}/{len(cfg.seeds)} done")

    # Save raw arrays
    save_dict = {
        "arm_gaps": np.asarray(cfg.arm_gaps, dtype=float),
        "seeds": np.asarray(cfg.seeds),
    }

    for metric in metric_names:
        for algo in algo_names:
            key_algo = algo.lower().replace(" ", "_").replace("+", "").replace("-", "_")
            save_dict[f"{key_algo}_{metric}"] = results[metric][algo]

    np.savez(
        os.path.join(cfg.results_dir, "arm_gap_sweep.npz"),
        **save_dict,
    )

    # Print compact summary
    print("\nArm-gap sweep summary: mean binary stable regret")
    for g_idx, gap in enumerate(cfg.arm_gaps):
        print(f"\ngap={gap:.3f}")
        for algo in algo_names:
            mean, ci = _mean_ci(results["binary_stable_regret"][algo][g_idx])
            print(f"  {algo:18s}: {mean:.2f} ± {ci:.2f}")

    # Plot each metric as function of arm gap
    for metric in metric_names:
        plt.figure(figsize=(7, 4))

        for algo in algo_names:
            means = []
            cis = []

            for g_idx in range(len(cfg.arm_gaps)):
                mean, ci = _mean_ci(results[metric][algo][g_idx])
                means.append(mean)
                cis.append(ci)

            means = np.asarray(means)
            cis = np.asarray(cis)

            plt.plot(cfg.arm_gaps, means, marker="o", label=algo)
            plt.fill_between(cfg.arm_gaps, means - cis, means + cis, alpha=0.15)

        plt.gca().invert_xaxis()
        plt.xlabel("Arm preference gap")
        plt.ylabel(metric.replace("_", " "))
        plt.title(metric.replace("_", " "))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.results_dir, f"gap_sweep_{metric}.png"), dpi=200)
        plt.close()

    print(f"\nSaved gap-sweep results in {cfg.results_dir}")


def run_random_market_gap_sweep(cfg: ExtensionConfig) -> None:
    """
    Robustness experiment with randomized preference markets.

    The preference order is randomized:
        each player receives a random ranking over arms;
        each arm receives a random ranking over players.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    algo_names = ["Known-arm UCB+GS", "Two-sided UCB+GS", "Two-sided ETC"]

    # results[algo] has shape (num_gaps, num_trials)
    results = {
        algo: np.zeros((len(cfg.random_market_gaps), cfg.random_trials), dtype=float)
        for algo in algo_names
    }

    print("\n=== Random-market gap sweep ===")
    print(f"T={cfg.T}, random_trials={cfg.random_trials}")
    print("gaps:", cfg.random_market_gaps)

    for g_idx, gap in enumerate(cfg.random_market_gaps):
        print(f"\nGap {gap:.3f} ({g_idx + 1}/{len(cfg.random_market_gaps)})")

        for trial in range(cfg.random_trials):
            # One RNG generates the market.
            market_rng = np.random.default_rng(100_000 + 10_000 * g_idx + trial)
            instance = _generate_random_preference_instance(market_rng, arm_gap=gap)

            # Use the trial index as the stochastic reward seed.
            seed = trial

            rng = np.random.default_rng(seed)
            out_known = run_ucb_gs(
                instance,
                T=cfg.T,
                rng=rng,
                reward_model=cfg.reward_model,
                reward_sigma=cfg.reward_sigma,
            )

            rng = np.random.default_rng(seed)
            out_two_sided_ucb = run_two_sided_ucb_gs(
                instance,
                T=cfg.T,
                rng=rng,
                reward_model=cfg.reward_model,
                reward_sigma=cfg.reward_sigma,
            )

            rng = np.random.default_rng(seed)
            out_two_sided_etc = run_two_sided_etc(
                instance,
                T=cfg.T,
                rng=rng,
                h=cfg.etc_h,
                reward_model=cfg.reward_model,
                reward_sigma=cfg.reward_sigma,
            )

            outputs = {
                "Known-arm UCB+GS": out_known,
                "Two-sided UCB+GS": out_two_sided_ucb,
                "Two-sided ETC": out_two_sided_etc,
            }

            for algo_name, out in outputs.items():
                stability = stability_curves(instance.mu, instance.arm_prefs, out.match_history)
                results[algo_name][g_idx, trial] = stability["cum_binary_stable_regret"][-1]

            print(f"  Trial {trial + 1}/{cfg.random_trials} done")

    # Save raw values.
    save_dict = {
        "arm_gaps": np.asarray(cfg.random_market_gaps, dtype=float),
        "random_trials": np.asarray([cfg.random_trials], dtype=int),
    }

    for algo_name, arr in results.items():
        key = algo_name.lower().replace(" ", "_").replace("+", "").replace("-", "_")
        save_dict[f"{key}_binary_stable_regret"] = arr

    np.savez(
        os.path.join(cfg.results_dir, "random_market_gap_sweep.npz"),
        **save_dict,
    )

    # Plot final binary stable regret vs arm gap.
    plt.figure(figsize=(7, 4))

    for algo_name, arr in results.items():
        means = []
        cis = []

        for g_idx in range(len(cfg.random_market_gaps)):
            mean, ci = _mean_ci(arr[g_idx])
            means.append(mean)
            cis.append(ci)

        means = np.asarray(means)
        cis = np.asarray(cis)

        plt.plot(cfg.random_market_gaps, means, marker="o", label=algo_name)
        plt.fill_between(cfg.random_market_gaps, means - cis, means + cis, alpha=0.15)

    plt.gca().invert_xaxis()
    plt.xlabel("Arm preference gap")
    plt.ylabel("Average binary stable regret")
    plt.title("Randomized markets: binary stable regret vs arm gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "random_market_gap_sweep_binary_regret.png"), dpi=200)
    plt.close()

    print(f"\nSaved random-market gap sweep results in {cfg.results_dir}")
    print("Main files:")
    print(" - random_market_gap_sweep.npz")
    print(" - random_market_gap_sweep_binary_regret.png")


def run_random_stability_over_time_gap_010(cfg: ExtensionConfig) -> None:
    """
    Randomized robustness plot for cumulative binary stable regret over time.

    This experiment fixes the arm preference gap at 0.10 and generates
    randomized preference markets.

    The output is an average cumulative binary stable regret curve over time.

    Purpose:
        The fixed stability-over-time plot uses one handcrafted utility matrix.
        This version checks whether the same dynamic pattern appears when
        the numerical utilities are randomized while preserving the same
        preference structure.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    algo_names = ["Known-arm UCB+GS", "Two-sided UCB+GS", "Two-sided ETC"]

    # curves[algo] has shape (random_trials, T).
    # Each row is one random market + one stochastic run.
    curves = {
        algo: np.zeros((cfg.random_time_trials, cfg.T), dtype=float)
        for algo in algo_names
    }

    print("\n=== Randomized stability-over-time plot, gap=0.10 ===")
    print(
        f"T={cfg.T}, random_trials={cfg.random_time_trials}, "
        f"gap={cfg.random_time_gap}"
    )

    for trial in range(cfg.random_time_trials):
        # This RNG creates the random market itself.
        # We keep it separate from the reward RNGs for reproducibility.
        market_rng = np.random.default_rng(100_000 + trial)
        instance = _generate_random_preference_instance(
            market_rng,
            arm_gap=cfg.random_time_gap,
        )

        # Same trial seed for stochastic rewards across algorithms.
        # This keeps comparisons reasonably aligned.
        seed = trial

        rng = np.random.default_rng(seed)
        out_known = run_ucb_gs(
            instance,
            T=cfg.T,
            rng=rng,
            reward_model=cfg.reward_model,
            reward_sigma=cfg.reward_sigma,
        )

        rng = np.random.default_rng(seed)
        out_two_sided_ucb = run_two_sided_ucb_gs(
            instance,
            T=cfg.T,
            rng=rng,
            reward_model=cfg.reward_model,
            reward_sigma=cfg.reward_sigma,
        )

        rng = np.random.default_rng(seed)
        out_two_sided_etc = run_two_sided_etc(
            instance,
            T=cfg.T,
            rng=rng,
            h=cfg.etc_h,
            reward_model=cfg.reward_model,
            reward_sigma=cfg.reward_sigma,
        )

        outputs = {
            "Known-arm UCB+GS": out_known,
            "Two-sided UCB+GS": out_two_sided_ucb,
            "Two-sided ETC": out_two_sided_etc,
        }

        for algo_name, out in outputs.items():
            stability = stability_curves(
                instance.mu,
                instance.arm_prefs,
                out.match_history,
            )
            curves[algo_name][trial] = stability["cum_binary_stable_regret"]

        print(f"Trial {trial + 1}/{cfg.random_time_trials} done")

    # Save raw curves for reproducibility.
    save_dict = {
        "random_time_gap": np.asarray([cfg.random_time_gap], dtype=float),
        "random_time_trials": np.asarray([cfg.random_time_trials], dtype=int),
    }

    for algo_name, arr in curves.items():
        key = algo_name.lower().replace(" ", "_").replace("+", "").replace("-", "_")
        save_dict[f"{key}_cum_binary_stable_regret"] = arr

    np.savez(
        os.path.join(cfg.results_dir, "random_stability_over_time_gap_010.npz"),
        **save_dict,
    )

    plot_metric_curves(
        curves=curves,
        save_path=os.path.join(
            cfg.results_dir,
            "random_stability_over_time_gap_010_binary_regret.png",
        ),
        ylabel="Cumulative binary stable regret",
        title="Randomized markets: binary stable regret over time (gap=0.10)",
        xlabel="t",
    )

    print(f"\nSaved randomized stability-over-time results in {cfg.results_dir}")
    print("Main files:")
    print(" - random_stability_over_time_gap_010.npz")
    print(" - random_stability_over_time_gap_010_binary_regret.png")


def run_etc_h_sweep(cfg: ExtensionConfig) -> None:
    """
    Sweep the ETC exploration budget h for the two-sided ETC algorithm.

    This experiment asks:
        How much forced exploration is needed before ETC commits
        to a truly stable matching when arms also need to learn?

    We track:
        - binary stable regret over the full horizon;
        - cumulative blocking pairs;
        - time to permanent stability;
        - whether the committed matching is truly stable;
        - number of blocking pairs in the committed matching.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    metric_names = [
        "binary_stable_regret",
        "cumulative_blocking_pairs",
        "time_to_permanent_stability",
        "stable_commit",
        "commit_blocking_pairs",
    ]

    # results[metric] has shape (num_gaps, num_h_values, num_seeds)
    results = {
        metric: np.zeros(
            (len(cfg.etc_h_sweep_gaps), len(cfg.etc_h_values), len(cfg.seeds)),
            dtype=float,
        )
        for metric in metric_names
    }

    print("\n=== Two-sided ETC h-sweep ===")
    print(f"instance={cfg.instance_name}, T={cfg.T}, seeds={len(cfg.seeds)}")
    print("gaps:", cfg.etc_h_sweep_gaps)
    print("h values:", cfg.etc_h_values)

    for g_idx, gap in enumerate(cfg.etc_h_sweep_gaps):
        instance = generate_handcrafted_instance(cfg.instance_name, delta=gap)

        print(f"\nGap {gap:.3f} ({g_idx + 1}/{len(cfg.etc_h_sweep_gaps)})")

        for h_idx, h in enumerate(cfg.etc_h_values):
            print(f"  h={h} ({h_idx + 1}/{len(cfg.etc_h_values)})")

            for s_idx, seed in enumerate(cfg.seeds):
                rng = np.random.default_rng(seed)

                out = run_two_sided_etc(
                    instance,
                    T=cfg.T,
                    rng=rng,
                    h=h,
                    reward_model=cfg.reward_model,
                    reward_sigma=cfg.reward_sigma,
                )

                stability = stability_curves(instance.mu, instance.arm_prefs, out.match_history)

                final_binary = float(stability["cum_binary_stable_regret"][-1])
                final_blocking = float(stability["cum_blocking_pairs"][-1])

                permanent = stability["time_to_permanent_stability"]
                if permanent is None:
                    permanent_value = cfg.T + 1
                else:
                    permanent_value = float(permanent)

                # ETC commits to one matching after exploration.
                # This checks whether that final committed matching is truly stable.
                # If it is not stable, ETC can remain wrong for the rest of the horizon.
                stable_commit = float(
                    is_stable_matching(instance.mu, instance.arm_prefs, out.committed_match)
                )

                commit_blocking = float(
                    count_blocking_pairs(instance.mu, instance.arm_prefs, out.committed_match)
                )

                results["binary_stable_regret"][g_idx, h_idx, s_idx] = final_binary
                results["cumulative_blocking_pairs"][g_idx, h_idx, s_idx] = final_blocking
                results["time_to_permanent_stability"][g_idx, h_idx, s_idx] = permanent_value
                results["stable_commit"][g_idx, h_idx, s_idx] = stable_commit
                results["commit_blocking_pairs"][g_idx, h_idx, s_idx] = commit_blocking

            mean_binary, ci_binary = _mean_ci(results["binary_stable_regret"][g_idx, h_idx])
            mean_stable_commit, ci_commit = _mean_ci(results["stable_commit"][g_idx, h_idx])

            print(
                f"    binary stable regret: {mean_binary:.2f} ± {ci_binary:.2f} | "
                f"stable commit rate: {mean_stable_commit:.2f} ± {ci_commit:.2f}"
            )

    # Save raw arrays
    save_dict = {
        "gaps": np.asarray(cfg.etc_h_sweep_gaps, dtype=float),
        "h_values": np.asarray(cfg.etc_h_values, dtype=int),
        "seeds": np.asarray(cfg.seeds, dtype=int),
    }

    for metric in metric_names:
        save_dict[metric] = results[metric]

    np.savez(
        os.path.join(cfg.results_dir, "etc_h_sweep.npz"),
        **save_dict,
    )

    # Plot metrics as a function of h.
    # Binary stable regret can have a trade-off:
    #   small h  -> bad estimates, wrong commitment;
    #   large h  -> better estimates, but more unstable exploration rounds.
    for metric in [
        "binary_stable_regret",
        "cumulative_blocking_pairs",
        "time_to_permanent_stability",
        "commit_blocking_pairs",
    ]:
        plt.figure(figsize=(7, 4))

        for g_idx, gap in enumerate(cfg.etc_h_sweep_gaps):
            means = []
            cis = []

            for h_idx in range(len(cfg.etc_h_values)):
                mean, ci = _mean_ci(results[metric][g_idx, h_idx])
                means.append(mean)
                cis.append(ci)

            means = np.asarray(means)
            cis = np.asarray(cis)

            plt.plot(cfg.etc_h_values, means, marker="o", label=f"gap={gap}")
            plt.fill_between(cfg.etc_h_values, means - cis, means + cis, alpha=0.15)

        plt.xscale("log")
        plt.xlabel("ETC exploration budget h")
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"Two-sided ETC: {metric.replace('_', ' ')}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.results_dir, f"etc_h_sweep_{metric}.png"), dpi=200)
        plt.close()

    # Stable commit rate is the cleanest ETC diagnostic:
    # it directly measures whether exploration was sufficient to identify
    # a truly stable matching before committing.
    plt.figure(figsize=(7, 4))

    for g_idx, gap in enumerate(cfg.etc_h_sweep_gaps):
        means = []
        cis = []

        for h_idx in range(len(cfg.etc_h_values)):
            mean, ci = _mean_ci(results["stable_commit"][g_idx, h_idx])
            means.append(mean)
            cis.append(ci)

        means = np.asarray(means)
        cis = np.asarray(cis)

        plt.plot(cfg.etc_h_values, means, marker="o", label=f"gap={gap}")
        plt.fill_between(cfg.etc_h_values, means - cis, means + cis, alpha=0.15)

    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("ETC exploration budget h")
    plt.ylabel("stable commitment rate")
    plt.title("Two-sided ETC: probability of committing to a stable matching")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "etc_h_sweep_stable_commit_rate.png"), dpi=200)
    plt.close()

    print(f"\nSaved ETC h-sweep results in {cfg.results_dir}")
    print("Main files:")
    print(" - etc_h_sweep.npz")
    print(" - etc_h_sweep_binary_stable_regret.png")
    print(" - etc_h_sweep_stable_commit_rate.png")
    print(" - etc_h_sweep_commit_blocking_pairs.png")


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line options.

    This makes the file usable both for full final experiments and quick tests.

    Examples:
        python -m src.extension_experiments --experiment all
        python -m src.extension_experiments --experiment gap --T 100 --num-seeds 5
    """
    parser = argparse.ArgumentParser(
        description="Run two-sided learning extension experiments."
    )

    parser.add_argument(
        "--experiment",
        choices=["fixed", "gap", "etc", "time", "random", "random-time", "all"],
        default="all",
        help=(
            "Which experiment to run: "
            "'fixed' for fixed-gap comparison, "
            "'gap' for arm-gap sweep, "
            "'etc' for ETC h-sweep, "
            "'time' for binary stable regret over time, "
            "'random' for randomized-market robustness sweep, "
            "'random-time' for randomized binary stable regret over time at gap 0.10, "
            "'all' for all experiments."
        ),
    )

    parser.add_argument(
        "--T",
        type=int,
        default=None,
        help="Optional override for the horizon.",
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Optional override for the number of seeds.",
    )

    parser.add_argument(
        "--num-random-trials",
        type=int,
        default=None,
        help="Optional override for randomized-market trials.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional override for the results directory.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for the extension experiment runner.

    The command-line argument --experiment decides which experiment is run:
        fixed -> fixed-gap comparison;
        gap   -> arm-gap sweep;
        etc   -> ETC h-sweep;
        all   -> everything.

    The optional arguments --T, --num-seeds, and --results-dir are useful
    for quick debugging without changing the source code.
    """
    args = _parse_args()
    cfg = ExtensionConfig()

    # Optional command-line overrides.
    # These are useful for quick tests, e.g. T=100 and 5 seeds.
    if args.T is not None:
        cfg.T = args.T

    if args.num_seeds is not None:
        cfg.seeds = list(range(args.num_seeds))

    if args.num_random_trials is not None:
        cfg.random_trials = args.num_random_trials
        cfg.random_time_trials = args.num_random_trials

    if args.results_dir is not None:
        cfg.results_dir = args.results_dir

    print("\n=== Extension experiment runner ===")
    print(f"experiment={args.experiment}")
    print(f"instance={cfg.instance_name}")
    print(f"T={cfg.T}")
    print(f"num_seeds={len(cfg.seeds)}")
    print(f"results_dir={cfg.results_dir}")

    # Run selected experiment blocks.
    if args.experiment in ["fixed", "all"]:
        run_fixed_gap_comparison(cfg)

    if args.experiment in ["gap", "all"]:
        run_arm_gap_sweep(cfg)

    if args.experiment in ["time", "all"]:
        run_stability_over_time(cfg)

    if args.experiment in ["random", "all"]:
        run_random_market_gap_sweep(cfg)

    if args.experiment in ["random-time", "all"]:
        run_random_stability_over_time_gap_010(cfg)
        
    if args.experiment in ["etc", "all"]:
        run_etc_h_sweep(cfg)


if __name__ == "__main__":
    main()