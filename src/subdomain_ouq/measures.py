"""Dirac-measure recovery and assembly utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .canonical import fixed_canonical_per_bin, gauss_nodes_weights_from_J, jacobi_from_canonical
from .moments import make_partition_edges, moment_residuals, subinterval_moments


def build_bin_measure_from_p(edges: np.ndarray, j: int, p_fixed: List[float], p_free: Sequence[float], M_row: np.ndarray, eps: float = 1e-9):
    L, U = float(edges[j]), float(edges[j + 1])
    r = len(p_fixed)
    m = r + 1
    need = 2 * m - 1
    p_all = [float(np.clip(v, eps, 1.0 - eps)) for v in (list(p_fixed) + list(p_free))]
    if len(p_all) < need:
        raise ValueError(f"Not enough canonical moments: need {need}, got {len(p_all)}.")
    J = jacobi_from_canonical(p_all, m, eps=eps)
    y, w_hat = gauss_nodes_weights_from_J(J)
    x = L + (U - L) * y
    t = float(M_row[0]) * w_hat
    return x, t


def assemble_global(X_by_bin: List[np.ndarray], T_by_bin: List[np.ndarray]):
    return np.concatenate(X_by_bin), np.concatenate(T_by_bin)


def pof_from_global(xs: np.ndarray, ts: np.ndarray, c: float, side: str = "ge") -> float:
    mask = xs >= c if side == "ge" else xs <= c
    return float(np.asarray(ts, dtype=float)[mask].sum())


def decode_theta_isotropic(theta: np.ndarray, d: int, K: int, r: int) -> np.ndarray:
    return np.asarray(theta, dtype=float).reshape(d, K, r + 1)


def build_marginal_with_canonical_moments(edges: np.ndarray, M: np.ndarray, r: int, theta_matrix: np.ndarray, check_residuals: bool = False, eps: float = 1e-9):
    K = len(edges) - 1
    xs_list, ts_list, residuals = [], [], []
    for j in range(K):
        p_fixed = fixed_canonical_per_bin(M[j, :], float(edges[j]), float(edges[j + 1]), r, eps=eps)
        x_j, t_j = build_bin_measure_from_p(edges, j, p_fixed, theta_matrix[j, :].tolist(), M[j, :], eps=eps)
        xs_list.append(x_j)
        ts_list.append(t_j)
        if check_residuals:
            _, res = moment_residuals(x_j, t_j, M[j, :], r)
            residuals.append((j, res))
    xs, ts = assemble_global(xs_list, ts_list)
    return xs, ts, (residuals if check_residuals else None)


@dataclass
class OUQCase:
    name: str
    K_vec: List[int]
    r_vec: List[int]

    @property
    def d(self) -> int:
        return len(self.K_vec)

    @property
    def nvar(self) -> int:
        return int(sum(K * (r + 1) for K, r in zip(self.K_vec, self.r_vec)))

    @property
    def total_atoms(self) -> int:
        total = 1
        for K, r in zip(self.K_vec, self.r_vec):
            total *= K * (r + 1)
        return int(total)


@dataclass
class PreparedCase:
    case: OUQCase
    edges_list: List[np.ndarray]
    moments_list: List[np.ndarray]
    p_fixed_list: List[List[List[float]]]


def prepare_anisotropic_case(dists: Sequence, case: OUQCase, eps: float = 1e-9) -> PreparedCase:
    edges_list, moments_list, p_fixed_list = [], [], []
    for i, dist in enumerate(dists):
        edges = make_partition_edges(dist, case.K_vec[i])
        M = subinterval_moments(dist, edges, case.r_vec[i])
        fixed_i = [fixed_canonical_per_bin(M[j, :], edges[j], edges[j + 1], case.r_vec[i], eps=eps) for j in range(case.K_vec[i])]
        edges_list.append(edges)
        moments_list.append(M)
        p_fixed_list.append(fixed_i)
    return PreparedCase(case=case, edges_list=edges_list, moments_list=moments_list, p_fixed_list=p_fixed_list)


def decode_theta_anisotropic(theta: np.ndarray, case: OUQCase) -> List[np.ndarray]:
    theta = np.asarray(theta, dtype=float).ravel()
    blocks = []
    idx = 0
    for K, r in zip(case.K_vec, case.r_vec):
        n = K * (r + 1)
        block = theta[idx: idx + n]
        if block.size != n:
            raise ValueError("Wrong theta size.")
        blocks.append(block.reshape(K, r + 1))
        idx += n
    return blocks


def build_marginals_from_theta_anisotropic(prep: PreparedCase, theta: np.ndarray, eps: float = 1e-9):
    blocks = decode_theta_anisotropic(theta, prep.case)
    marginals = []
    max_res = 0.0
    for i in range(prep.case.d):
        xs_list, ts_list = [], []
        for j in range(prep.case.K_vec[i]):
            x_j, t_j = build_bin_measure_from_p(
                prep.edges_list[i], j, prep.p_fixed_list[i][j], blocks[i][j, :].tolist(), prep.moments_list[i][j, :], eps=eps
            )
            _, res = moment_residuals(x_j, t_j, prep.moments_list[i][j, :], prep.case.r_vec[i])
            max_res = max(max_res, res)
            xs_list.append(x_j)
            ts_list.append(t_j)
        marginals.append(assemble_global(xs_list, ts_list))
    return marginals, float(max_res)
