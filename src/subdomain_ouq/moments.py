"""Moment calculations and affine transformations for subdomain OUQ."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.integrate import quad


def make_edges(a: float, b: float, K: int) -> np.ndarray:
    if K <= 0:
        raise ValueError("K must be positive.")
    if not (a < b):
        raise ValueError("Need a < b.")
    return np.linspace(float(a), float(b), int(K) + 1)


def make_partition_edges(dist, K: int, x_star: Optional[float] = None, use_design_split: bool = False) -> np.ndarray:
    """Make equal-probability edges if dist has ppf; otherwise use equal-width edges."""
    if K <= 0:
        raise ValueError("K must be positive.")
    a = float(getattr(dist, "a"))
    b = float(getattr(dist, "b"))
    if K == 1:
        return np.array([a, b], dtype=float)
    if not hasattr(dist, "ppf"):
        return make_edges(a, b, K)
    if x_star is None or not use_design_split or not hasattr(dist, "cdf"):
        qs = np.linspace(0.0, 1.0, K + 1)
        edges = np.asarray(dist.ppf(qs), dtype=float)
        edges[0] = a
        edges[-1] = b
        return edges
    q_star = float(np.clip(dist.cdf(x_star), 1e-10, 1.0 - 1e-10))
    n_left = K // 2
    n_right = K - n_left
    q_left = np.linspace(0.0, q_star, n_left + 1)
    q_right = np.linspace(q_star, 1.0, n_right + 1)
    qs = np.r_[q_left, q_right[1:]]
    edges = np.asarray(dist.ppf(qs), dtype=float)
    edges[0] = a
    edges[-1] = b
    return np.maximum.accumulate(edges)


def moment_over_interval(dist, L: float, U: float, q: int) -> float:
    if q < 0:
        raise ValueError("Moment order q must be nonnegative.")
    if q == 0:
        val, _ = quad(lambda x: dist.pdf(x), L, U, epsabs=1e-12, epsrel=1e-12, limit=300)
    else:
        val, _ = quad(lambda x: (x ** q) * dist.pdf(x), L, U, epsabs=1e-12, epsrel=1e-12, limit=300)
    return float(val)


def subinterval_moments(dist, edges: np.ndarray, r: int) -> np.ndarray:
    K = len(edges) - 1
    M = np.zeros((K, r + 1), dtype=float)
    for j in range(K):
        L, U = float(edges[j]), float(edges[j + 1])
        for q in range(r + 1):
            M[j, q] = moment_over_interval(dist, L, U, q)
    return M


def subinterval_moments_uniform(a_i: float, b_i: float, edges: np.ndarray, r: int) -> np.ndarray:
    K = len(edges) - 1
    denom = float(b_i - a_i)
    if denom <= 0:
        raise ValueError("Need b_i > a_i.")
    M = np.zeros((K, r + 1), dtype=float)
    for j in range(K):
        L, U = float(edges[j]), float(edges[j + 1])
        M[j, 0] = (U - L) / denom
        for q in range(1, r + 1):
            M[j, q] = (U ** (q + 1) - L ** (q + 1)) / ((q + 1) * denom)
    return M


def raw_to_unit_moments(Mj: np.ndarray, L: float, U: float) -> np.ndarray:
    """Transform truncated raw moments on [L, U] to normalized moments on [0, 1]."""
    Mj = np.asarray(Mj, dtype=float)
    r = len(Mj) - 1
    mass = float(Mj[0])
    if mass <= 0:
        raise ValueError("Bin mass M0 must be positive.")
    width = float(U - L)
    if width <= 0:
        raise ValueError("U must be greater than L.")
    mprime = np.zeros(r + 1, dtype=float)
    mprime[0] = 1.0
    for q in range(1, r + 1):
        acc = 0.0
        for k in range(q + 1):
            acc += math.comb(q, k) * ((-L) ** (q - k)) * Mj[k]
        mprime[q] = acc / (mass * (width ** q))
    return mprime


def moment_residuals(x: np.ndarray, t: np.ndarray, M_row: np.ndarray, r: int):
    v = np.array([np.sum(t * (x ** q)) for q in range(r + 1)], dtype=float) - np.asarray(M_row[: r + 1], dtype=float)
    return v, float(np.max(np.abs(v)))
