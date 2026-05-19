"""NPZ I/O for trial results."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .simulator import TrialResult


def run_id(algo: str, adversary: str, C: float, seed: int) -> str:
    return f"algo={algo}__adv={adversary}__C={int(C)}__seed={int(seed)}"


def save_trial(
    result: TrialResult,
    out_dir: Path,
    *,
    algo: str,
    adversary: str,
    C: float,
    seed: int,
    alpha: float,
    eps: float,
    sigma: float = 1.0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id(algo, adversary, C, seed)}.npz"
    np.savez_compressed(
        path,
        matchings=result.matchings,
        cost=result.cost,
        m_star=result.m_star,
        mu=result.mu,
        first_exhaust_round=result.first_exhaust_round,
        total_cost=result.total_cost,
        zero_rounds=result.zero_rounds,
        active_rounds=result.active_rounds,
        saturated_rounds=result.saturated_rounds,
        avg_cost_per_active=result.avg_cost_per_active,
        meta=np.array(
            [
                ("algo", algo),
                ("adversary", adversary),
                ("C", str(C)),
                ("T", str(result.T)),
                ("N", str(result.N)),
                ("K", str(result.K)),
                ("seed", str(seed)),
                ("alpha", str(alpha)),
                ("eps", str(eps)),
                ("sigma", str(sigma)),
            ],
            dtype=object,
        ),
    )
    return path


def load_trial(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        meta = dict(f["meta"].tolist()) if "meta" in f.files else {}
        out = {
            "matchings": f["matchings"],
            "cost": f["cost"],
            "m_star": f["m_star"],
            "mu": f["mu"],
            "first_exhaust_round": f["first_exhaust_round"],
            "total_cost": f["total_cost"],
            "zero_rounds": f["zero_rounds"],
            "active_rounds": f["active_rounds"],
            "meta": meta,
        }
        # New fields (Gaussian/saturation logging) — fall back gracefully for old npz.
        if "saturated_rounds" in f.files:
            out["saturated_rounds"] = f["saturated_rounds"]
        if "avg_cost_per_active" in f.files:
            out["avg_cost_per_active"] = f["avg_cost_per_active"]
        return out
