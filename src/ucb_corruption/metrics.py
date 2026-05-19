"""Clean stable pseudo-regret and aggregation helpers (spec §6, §9.5)."""
from __future__ import annotations

import numpy as np


def clean_pseudo_regret(matchings: np.ndarray, mu: np.ndarray, m_star: np.ndarray) -> np.ndarray:
    """Per-round cumulative clean pseudo-regret of shape (T, N).

    R_i(t) = sum_{s<=t} (mu[i, m_star[i]] - mu[i, m_s(i)]).
    """
    T, N = matchings.shape
    baseline = mu[np.arange(N), m_star]                        # (N,)
    played = mu[np.arange(N)[None, :], matchings.astype(np.int64)]  # (T, N)
    inst = baseline[None, :] - played                          # (T, N)
    return np.cumsum(inst, axis=0)


def max_regret(R: np.ndarray) -> np.ndarray:
    """MaxReg(t) = max_i R_i(t), shape (T,)."""
    return R.max(axis=1)


def downsample(curve: np.ndarray, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) with n evenly spaced points along axis 0."""
    T = curve.shape[0]
    if n >= T:
        idx = np.arange(T)
    else:
        idx = np.linspace(0, T - 1, n, dtype=np.int64)
    return idx, curve[idx]


def stable_match_rate(matchings: np.ndarray, m_star: np.ndarray, tail_frac: float = 0.1) -> float:
    """Fraction of last `tail_frac` rounds where m_t == m_star (full vector)."""
    T = matchings.shape[0]
    tail_start = int(np.floor((1.0 - tail_frac) * T))
    tail = matchings[tail_start:]                              # (Ttail, N)
    full_match = np.all(tail == m_star[None, :].astype(tail.dtype), axis=1)
    return float(full_match.mean())
