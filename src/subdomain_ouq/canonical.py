"""Canonical-moment and Jacobi-matrix utilities."""
from __future__ import annotations

import math
from typing import List

import numpy as np
from numpy.linalg import eigh

from .moments import raw_to_unit_moments


def canonical_from_unit_moments(mprime: np.ndarray, eps: float = 1e-9) -> List[float]:
    """Return fixed canonical moments p1..pr for r<=3.

    The numerical examples in the paper use moment orders up to r=3 for 1D and
    up to r=2 for the higher-dimensional cases.
    """
    mprime = np.asarray(mprime, dtype=float)
    r = len(mprime) - 1
    if r == 0:
        return []
    if r > 3:
        raise NotImplementedError("Only r in {0,1,2,3} is implemented.")
    m1 = float(mprime[1])
    p1 = float(np.clip(m1, eps, 1.0 - eps))
    if r == 1:
        return [p1]
    m2 = float(mprime[2])
    denom = m1 * (1.0 - m1)
    p2 = 0.5 if denom <= eps else (m2 - m1 * m1) / denom
    p2 = float(np.clip(p2, eps, 1.0 - eps))
    if r == 2:
        return [p1, p2]
    m3 = float(mprime[3])
    m3_a = (m2 * m2) / m1 if m1 > eps else 0.0
    if m1 >= 1.0 - eps:
        m3_b = 1.0
    else:
        t_b = (m1 - m2) / (1.0 - m1)
        t_b = float(np.clip(t_b, 0.0, 1.0))
        m3_b = 1.0 - (1.0 - m1) * (1.0 + t_b + t_b * t_b)
    lo, hi = min(m3_a, m3_b), max(m3_a, m3_b)
    p3 = 0.5 if abs(hi - lo) <= eps else (m3 - lo) / (hi - lo)
    p3 = float(np.clip(p3, eps, 1.0 - eps))
    return [p1, p2, p3]


def jacobi_from_canonical(p: List[float], m: int, eps: float = 0.0) -> np.ndarray:
    need = 2 * m - 1
    if len(p) < need:
        raise ValueError(f"Need canonical moments p1..p{need}, got {len(p)}.")
    vals = [float(np.clip(v, eps, 1.0 - eps)) if eps > 0 else float(v) for v in p[:need]]
    P = [0.0] + vals
    zeta = [0.0] * (2 * m)
    for k in range(1, 2 * m):
        zeta[k] = P[k] * (1.0 - P[k - 1])
    J = np.zeros((m, m), dtype=float)
    for i in range(m):
        J[i, i] = zeta[2 * i] + zeta[2 * i + 1]
        if i >= 1:
            b = max(0.0, zeta[2 * i - 1] * zeta[2 * i])
            J[i, i - 1] = J[i - 1, i] = math.sqrt(b)
    return J


def gauss_nodes_weights_from_J(J: np.ndarray, clip_nodes: bool = True):
    eigvals, eigvecs = eigh(J)
    nodes = np.asarray(eigvals, dtype=float)
    if clip_nodes:
        nodes = np.clip(nodes, 0.0, 1.0)
    weights = np.asarray(eigvecs[0, :] ** 2, dtype=float)
    return nodes, weights


def fixed_canonical_per_bin(M_row: np.ndarray, L: float, U: float, r: int, eps: float = 1e-9) -> List[float]:
    if r == 0:
        return []
    mprime = raw_to_unit_moments(M_row, L, U)
    p_fixed = canonical_from_unit_moments(mprime, eps=eps)
    if len(p_fixed) != r:
        raise RuntimeError("Unexpected fixed canonical-moment length.")
    return p_fixed
