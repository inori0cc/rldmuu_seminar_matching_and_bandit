# Extension 2 — Unknown Arm Preferences

This extension studies a two-sided version of the centralized matching-bandit setting from Liu et al. (2020).

In the original setting, players learn their preferences over arms, but arm-side preferences are assumed to be known by the platform. This extension removes that assumption: both players and arms must learn their preferences from noisy feedback.

The main question is:

> How much harder is it to reach stable matchings when both sides of the market must learn their preferences?

## Main idea

The original model learns only player-side utilities:

```text
players learn mu[i, j]
Gale-Shapley uses learned player rankings + true arm rankings
```

This extension introduces arm-side utilities:

```text
arms learn nu[j, i]
Gale-Shapley uses learned player rankings + learned arm rankings
```

As a result, Gale-Shapley may produce a matching that is stable with respect to the estimated rankings, but unstable with respect to the true preferences.

To evaluate this, we use **binary stable regret**:

```text
BSR(T) = number of rounds where the produced matching is unstable
```

A matching is unstable if at least one blocking pair exists under the true preferences.

## Main files

```text
algorithms.py                  Learning algorithms
market.py                      Market instances and reward sampling
matching.py                    Gale-Shapley implementation
metrics.py                     Stability and regret metrics
plotting.py                    Plotting utilities
extension_experiments.py       Runs the extension experiments
summarize_extension_results.py Summarizes saved results into tables
```


## Algorithms compared

The extension compares three algorithms:

1. **Known-arm UCB+GS**  
   Baseline from the original setting. Players learn their utilities, but Gale-Shapley receives the true arm rankings.

2. **Two-sided UCB+GS**  
   Both players and arms learn with UCB. Gale-Shapley uses learned rankings on both sides.

3. **Two-sided ETC**  
   Both sides explore for a fixed number of rounds, estimate their rankings, then commit to one Gale-Shapley matching.

## How to run

Run all extension experiments:

```bash
python -m src.extension_experiments --experiment all
```

Run only the controlled arm-gap sweep:

```bash
python -m src.extension_experiments --experiment gap
```

Run only the randomized stability-over-time experiment:

```bash
python -m src.extension_experiments --experiment random-time
```

Run only the ETC exploration-budget sweep:

```bash
python -m src.extension_experiments --experiment etc
```

Quick test with fewer seeds and shorter horizon:

```bash
python -m src.extension_experiments --experiment gap --T 100 --num-seeds 5
```

Quick randomized test:

```bash
python -m src.extension_experiments --experiment random-time --T 100 --num-random-trials 5
```

## Outputs

Results are saved in:

```text
results/
```

Typical output files include:

```text
arm_gap_sweep.npz
gap_sweep_binary_stable_regret.png
random_stability_over_time_gap_010.npz
random_stability_over_time_gap_010_binary_regret.png
etc_h_sweep.npz
etc_h_sweep_stable_commit_rate.png
```

To summarize the saved results into Markdown tables:

```bash
python -m src.summarize_extension_results
```

This creates:

```text
results/final_summary_tables.md
```

## Main experiments

### 1. Controlled arm-gap sweep

This experiment uses a controlled 3x3 market where player preferences are easy to learn, while the arm-side preference gap is varied.

```text
large gap = arms can easily distinguish players
small gap = arm-side learning is harder
```

The goal is to isolate the effect of arm-side uncertainty.

### 2. Randomized stability-over-time experiment

This experiment randomly generates player and arm preference orders, converts them into utility values, and measures cumulative binary stable regret over time at fixed arm gap:

```text
delta = 0.10
```

This checks whether the main pattern remains under randomized markets.

## Main takeaway

Stable matching becomes harder when arm preferences are not known and must also be learned. In particular, two-sided UCB+GS accumulates more unstable rounds than the known-arm UCB+GS baseline when arm-side preferences are difficult to distinguish.
