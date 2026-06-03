# Competing Bandits in Matching Markets — Layer 1 Replication

Replicates Figure 1 (a, b, c) from Liu, Mania, Jordan, *Competing Bandits in Matching Markets* (2020, arXiv:1906.05363) using centralized UCB on the Gale-Shapley platform.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Unit tests
pytest tests/ -v

# Smoke tests (seconds)
python experiments/run_figure_1a.py --config config/figure_1a.yaml --smoke
python experiments/run_figure_1b.py --config config/figure_1b.yaml --smoke
python experiments/run_figure_1c.py --config config/figure_1c.yaml --smoke

# Full runs
python experiments/run_figure_1a.py --config config/figure_1a.yaml   # ~25 s
python experiments/run_figure_1b.py --config config/figure_1b.yaml   # ~1 s
python experiments/run_figure_1c.py --config config/figure_1c.yaml   # ~5 min

# Plots
python analysis/plot_figure_1a.py
python analysis/plot_figure_1b.py
python analysis/plot_figure_1c.py
```

Outputs: `data/figure_1{a,b,c}.npz`, `figures/figure_1{a,b,c}.{pdf,png}`.

## Repository layout

- `src/`: core library (market, GS, agents, platforms, simulator, regret).
- `tests/`: unit tests — `pytest tests/ -v`.
- `experiments/`: one script per figure, each reads a YAML config.
- `analysis/`: plotting scripts.
- `config/`: YAML configs for the three figures.
- `doc/`: paper PDF and Chinese experiment design document.
- `REPLICATION_NOTES.md`: runtime / qualitative comparison against the paper.

## What is covered

Layer 1 only: three figures exactly as specified in `doc/experiment_design.md`. The ETC Platform (paper Table 1, left) is implemented and unit-tested for completeness; Figure 1 only exercises the GS Platform.
