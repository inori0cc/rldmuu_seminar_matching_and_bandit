# Replication Notes — Layer 1

## Environment

- Python 3.10.14 (`/opt/anaconda3/bin/python3`)
- numpy 1.26.4, scipy 1.15.1, matplotlib 3.9.2, pyyaml 6.0.1, pandas 2.2.2, tqdm 4.66.5, pytest 7.4.4
- Platform: macOS (Apple Silicon, Darwin 25.1.0)

## Unit tests

80 tests pass in ~0.3 s:

| Module | Tests |
|---|---|
| `test_gale_shapley.py` | 55 |
| `test_market.py` | 10 |
| `test_agents.py` | 5 |
| `test_platforms.py` | 6 |
| `test_regret.py` | 4 |

## Seeds

Deterministic top-level seeds (from config YAML); per-trial RNGs derived via `np.random.SeedSequence.spawn`. Reproducing any single run requires only the corresponding seed.

- Figure 1a: seed = `20240101` (with `+d_idx` offset per Δ grid point).
- Figure 1b: seed = `20240102`.
- Figure 1c: seed = `20240103`.

## Runtime

| Figure | Wall clock | Trials | Horizon | Note |
|---|---|---|---|---|
| 1a | **23.9 s** | 100 per Δ × 20 Δ values | 400 | |
| 1b | **0.9 s** | 100 | 200 | |
| 1c | **169.1 s** (~2.8 min) | 50 | 8000 | 20-agent GS each round |

Much faster than the ~7-minute estimate in `doc/experiment_design.md`; the 1c bottleneck is the Python-level per-round GS for N=K=20.

## Qualitative comparison with paper Figure 1

### Figure 1a (Example 2)
**Match.** `Agent 2` expected regret falls monotonically from ~170 to ~8 on log-y as Δ sweeps 0.05 → 1.0; `Agent 1` is roughly flat in the 8–15 range. The cross-over toward the right (both near ~8) and the ~20× ratio between the two agents at small Δ both reproduce the paper's Figure 1a. Precision note: our Δ grid starts at 0.05 rather than 0 because `Market` requires strict preferences; the paper starts at 0 which makes its left-most point degenerate.

### Figure 1b (Example 6)
**Match.** Agents 1 and 2 show near-linear cumulative regret, reaching 184 / 178 at horizon 200 (slope ≈ 0.9). Agent 3 stays in the 0–2 band, precisely as the paper predicts: p₃ happens to be matched to a₃ under both the optimal stable matching and the linear-regret matching that p₁ and p₂'s bad learning forces, so p₃'s optimal regret never grows.

### Figure 1c (Example 7)
**Match.** Rank-1 agent has positive sublinear regret (~355 at n=8000), rank-5 agent has smaller positive regret (~85). Rank 10, 15, 20 all end with negative regret (−53, −90, −150 respectively). The lower-ranked the agent, the more negative its expected agent-optimal regret, reproducing the paper's observation that sub-optimal agents often play arms better than their stable-match arm during exploration.

## Deviations from the paper / design doc

1. **Δ grid in Figure 1a** starts at 0.05 instead of 0 (strict-preference requirement). Does not change the qualitative shape.
2. **ETC cyclic index**: paper formula `a_{(t+i-1) mod K + 1}` translated to `(t_py + i_py + 1) % K` (0-indexed). The design doc's comment had a one-step phase shift; both are valid collision-free cyclic assignments with $h$ pulls per (agent, arm). Figure 1 does not use ETC, so this does not affect the output.
3. **UCB index typo** in paper Eq. (6): $T_i(t-1)$ is implemented as $T_{i,j}(t-1)$ (per-pair count) — the only interpretation consistent with per-arm ranking.

## Artifacts

- `data/figure_1a.npz` — `deltas`, `mean_regrets (20, 2)`, `sem_regrets (20, 2)`.
- `data/figure_1b.npz` — `cum_regret_mean (200, 3)`, `cum_regret_sem (200, 3)`.
- `data/figure_1c.npz` — `cum_regret_mean (8000, 20)`, `cum_regret_sem (8000, 20)`, `agents_to_show`.
- `figures/figure_1{a,b,c}.{pdf,png}`.
