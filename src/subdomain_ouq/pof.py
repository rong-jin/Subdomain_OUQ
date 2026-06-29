"""Probability-of-failure evaluation: exact tensor sums, ITS, and CRN."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc


def sample_from_discrete(xs: np.ndarray, ts: np.ndarray, U: np.ndarray) -> np.ndarray:
    w = np.asarray(ts, dtype=float) / np.sum(ts)
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, U, side="right")
    idx = np.clip(idx, 0, len(xs) - 1)
    return np.asarray(xs)[idx]


def _failure_mask(vals: np.ndarray, c_fail: float, side: str) -> np.ndarray:
    if side == "ge":
        return vals >= c_fail
    if side == "le":
        return vals <= c_fail
    raise ValueError("side must be 'ge' or 'le'.")


def pof_from_marginals_exact(marginals: List[Tuple[np.ndarray, np.ndarray]], g_func: Callable[[np.ndarray], np.ndarray], c_fail: float = 0.0, side: str = "ge") -> float:
    d = len(marginals)
    sizes = [len(ts) for _, ts in marginals]
    grids = np.meshgrid(*[xs for xs, _ in marginals], indexing="ij")
    X = np.stack([g.ravel() for g in grids], axis=1)
    W = np.ones(X.shape[0], dtype=float)
    for i, (_, ts) in enumerate(marginals):
        w_i = np.asarray(ts, dtype=float) / np.sum(ts)
        shape = [sizes[j] if j == i else 1 for j in range(d)]
        W *= np.broadcast_to(w_i.reshape(shape), sizes).ravel()
    vals = np.asarray(g_func(X), dtype=float)
    return float(W[_failure_mask(vals, c_fail, side)].sum())


def pof_from_marginals_mc(marginals: List[Tuple[np.ndarray, np.ndarray]], g_func: Callable[[np.ndarray], np.ndarray], c_fail: float, N_mc: int, U_shared: Optional[np.ndarray], side: str = "ge") -> float:
    d = len(marginals)
    if U_shared is None:
        U_shared = np.random.random((N_mc, d))
    X = np.empty((N_mc, d), dtype=float)
    for i, (xs, ts) in enumerate(marginals):
        X[:, i] = sample_from_discrete(xs, ts, U_shared[:, i])
    vals = np.asarray(g_func(X), dtype=float)
    return float(np.mean(_failure_mask(vals, c_fail, side)))


def pof_from_marginals_adaptive(marginals: List[Tuple[np.ndarray, np.ndarray]], g_func: Callable[[np.ndarray], np.ndarray], c_fail: float, N_mc: int, U_shared: Optional[np.ndarray], exact_threshold: int = 250_000, side: str = "ge"):
    total_atoms = int(np.prod([len(ts) for _, ts in marginals]))
    if total_atoms <= exact_threshold:
        return pof_from_marginals_exact(marginals, g_func, c_fail=c_fail, side=side), "exact"
    return pof_from_marginals_mc(marginals, g_func, c_fail=c_fail, N_mc=N_mc, U_shared=U_shared, side=side), "mc"


def sobol_crn(N: int, d: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(N)))
    return sampler.random_base2(m=m)[:N]
