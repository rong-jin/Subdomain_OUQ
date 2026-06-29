#!/usr/bin/env python3
"""Case 1: one-dimensional identity-function OUQ examples.

This script consolidates the original Case_1 scripts into a single entry point
for the four distributions used in the paper: truncated normal, uniform,
truncated Weibull, and truncated bimodal normal mixture.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from subdomain_ouq.canonical import fixed_canonical_per_bin
from subdomain_ouq.distributions import truncated_bimodal_normal_mixture, truncated_normal, truncated_weibull, uniform
from subdomain_ouq.measures import build_bin_measure_from_p, pof_from_global
from subdomain_ouq.moments import make_edges, moment_residuals, subinterval_moments
from subdomain_ouq.plotting import plot_pdf_with_failure_region


@dataclass
class Problem1D:
    name: str
    label: str
    a: float
    b: float
    c: float
    dist: object
    reference_pof: Optional[float] = None


def true_pof(dist, c: float, a: float, b: float) -> float:
    lo = max(float(c), float(a))
    hi = float(b)
    if lo >= hi:
        return 0.0
    val, _ = quad(lambda x: dist.pdf(x), lo, hi, epsabs=1e-12, epsrel=1e-12, limit=300)
    return float(val)


def make_case(name: str) -> Problem1D:
    key = name.lower().replace("-", "_")
    if key in {"normal", "truncnorm", "truncated_normal"}:
        a, b, c = -5.0, 5.0, 0.7
        dist = truncated_normal(a, b, 0.0, 1.0)
        return Problem1D("normal", "Truncated normal", a, b, c, dist)
    if key == "uniform":
        a, b, c = -5.0, 5.0, 1.7
        dist = uniform(a, b)
        return Problem1D("uniform", "Uniform", a, b, c, dist)
    if key == "weibull":
        a, b, c = 0.0, 10.0, 3.0
        dist = truncated_weibull(a, b, shape=1.5, scale=2.0)
        return Problem1D("weibull", "Truncated Weibull", a, b, c, dist)
    if key in {"bimodal", "mixture"}:
        a, b, c = -5.0, 5.0, 1.3
        dist = truncated_bimodal_normal_mixture(a, b, w=0.35)
        return Problem1D("bimodal", "Truncated bimodal normal mixture", a, b, c, dist)
    raise ValueError(f"Unknown distribution: {name}")


def decode_theta(theta: np.ndarray, K: int, r: int, delta: float) -> List[List[float]]:
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size != K * (r + 1):
        raise ValueError("Wrong theta size.")
    out = []
    idx = 0
    for _ in range(K):
        out.append([float(np.clip(theta[idx + k], delta, 1.0 - delta)) for k in range(r + 1)])
        idx += r + 1
    return out


def objective_factory(edges: np.ndarray, M: np.ndarray, p_fixed_list: List[List[float]], c: float, r: int, maximize: bool, delta: float):
    K = len(edges) - 1

    def obj(theta):
        try:
            free_per_bin = decode_theta(theta, K, r, delta)
            X_by_bin, T_by_bin = [], []
            for j in range(K):
                x, t = build_bin_measure_from_p(edges, j, p_fixed_list[j], free_per_bin[j], M[j, :], eps=delta)
                X_by_bin.append(x)
                T_by_bin.append(t)
            xs = np.concatenate(X_by_bin)
            ts = np.concatenate(T_by_bin)
            pof = pof_from_global(xs, ts, c, side="ge")
            return -pof if maximize else pof
        except Exception:
            return 1e6

    return obj


def run_ouq_bounds_for_edges(
    prob: Problem1D,
    edges: np.ndarray,
    r: int,
    delta: float = 1e-3,
    seed: int = 0,
    popsize: int = 50,
    maxiter: int = 300,
    tol: float = 1e-6,
):
    M = subinterval_moments(prob.dist, edges, r)
    K = len(edges) - 1
    p_fixed_list = [fixed_canonical_per_bin(M[j, :], edges[j], edges[j + 1], r, eps=delta) for j in range(K)]
    bounds = [(delta, 1.0 - delta)] * (K * (r + 1))

    res_max = differential_evolution(
        objective_factory(edges, M, p_fixed_list, prob.c, r, maximize=True, delta=delta),
        bounds,
        strategy="best1bin",
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
        polish=True,
        disp=False,
    )
    res_min = differential_evolution(
        objective_factory(edges, M, p_fixed_list, prob.c, r, maximize=False, delta=delta),
        bounds,
        strategy="best1bin",
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        maxiter=maxiter,
        tol=tol,
        seed=seed + 1,
        polish=True,
        disp=False,
    )

    def decode_result(res):
        free_per_bin = decode_theta(res.x, K, r, delta)
        X_by_bin, T_by_bin, residuals = [], [], []
        for j in range(K):
            x, t = build_bin_measure_from_p(edges, j, p_fixed_list[j], free_per_bin[j], M[j, :], eps=delta)
            X_by_bin.append(x)
            T_by_bin.append(t)
            _, v_inf = moment_residuals(x, t, M[j, :], r)
            residuals.append((j, float(v_inf)))
        xs = np.concatenate(X_by_bin)
        ts = np.concatenate(T_by_bin)
        return {
            "nit": int(res.nit),
            "success": bool(res.success),
            "pof": pof_from_global(xs, ts, prob.c, side="ge"),
            "xs": xs,
            "ts": ts,
            "X_by_bin": X_by_bin,
            "T_by_bin": T_by_bin,
            "residuals": residuals,
        }

    return {
        "K": K,
        "r": r,
        "edges": edges,
        "M": M,
        "upper": decode_result(res_max),
        "lower": decode_result(res_min),
        "ref_pof": true_pof(prob.dist, prob.c, prob.a, prob.b),
    }


def export_atoms_csv(path: Path, edges: np.ndarray, X_by_bin: List[np.ndarray], T_by_bin: List[np.ndarray], c: float):
    rows = []
    for j, (x, t) in enumerate(zip(X_by_bin, T_by_bin)):
        for xi, ti in zip(x, t):
            rows.append({"bin_id": j, "x": float(xi), "t": float(ti), "ge_c": bool(xi >= c)})
    pd.DataFrame(rows).to_csv(path, index=False)


def sweep_problem(prob: Problem1D, K_list: List[int], r_list: List[int], outdir: Path, split_at_c: bool = False, delta: float = 1e-3, popsize: int = 50, maxiter: int = 300):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    ref = true_pof(prob.dist, prob.c, prob.a, prob.b)
    plot_pdf_with_failure_region(prob.dist, prob.a, prob.b, prob.c, outdir / f"pdf_{prob.name}.png", title=prob.label)
    for K in K_list:
        for r in r_list:
            edges = make_edges(prob.a, prob.b, K)
            if split_at_c and prob.a < prob.c < prob.b and np.min(np.abs(edges - prob.c)) > 1e-12:
                edges = np.sort(np.append(edges, prob.c))
            res = run_ouq_bounds_for_edges(prob, edges, r, delta=delta, seed=123 + 31 * r + K, popsize=popsize, maxiter=maxiter)
            case_dir = outdir / f"K{K}_r{r}"
            case_dir.mkdir(parents=True, exist_ok=True)
            export_atoms_csv(case_dir / f"upper_atoms_K{K}_r{r}.csv", res["edges"], res["upper"]["X_by_bin"], res["upper"]["T_by_bin"], prob.c)
            export_atoms_csv(case_dir / f"lower_atoms_K{K}_r{r}.csv", res["edges"], res["lower"]["X_by_bin"], res["lower"]["T_by_bin"], prob.c)
            rows.append({
                "distribution": prob.name,
                "K": K,
                "r": r,
                "upper": res["upper"]["pof"],
                "lower": res["lower"]["pof"],
                "ref_pof": ref,
                "upper_nit": res["upper"]["nit"],
                "lower_nit": res["lower"]["nit"],
                "max_residual_upper": max(v for _, v in res["upper"]["residuals"]),
                "max_residual_lower": max(v for _, v in res["lower"]["residuals"]),
            })
            print(f"{prob.name:8s} K={K:<2d} r={r:<1d} upper={rows[-1]['upper']:.8f} lower={rows[-1]['lower']:.8f} ref={ref:.8f}")
    df = pd.DataFrame(rows).sort_values(["distribution", "K", "r"]).reset_index(drop=True)
    df.to_csv(outdir / "ouq_bounds_summary.csv", index=False)
    return df


def parse_ints(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def main():
    ap = argparse.ArgumentParser(description="Run Case 1 identity-function OUQ examples.")
    ap.add_argument("--dist", default="all", choices=["all", "normal", "uniform", "weibull", "bimodal"])
    ap.add_argument("--K-list", nargs="+", default=["1", "2", "4", "8"])
    ap.add_argument("--r-list", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--outdir", default="outputs/case1_identity_1d")
    ap.add_argument("--popsize", type=int, default=50)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--split-at-c", action="store_true")
    args = ap.parse_args()

    names = ["normal", "uniform", "weibull", "bimodal"] if args.dist == "all" else [args.dist]
    K_list = parse_ints(args.K_list)
    r_list = parse_ints(args.r_list)
    all_rows = []
    root = Path(args.outdir)
    for name in names:
        prob = make_case(name)
        df = sweep_problem(prob, K_list, r_list, root / prob.name, split_at_c=args.split_at_c, popsize=args.popsize, maxiter=args.maxiter)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(root / "case1_bounds.csv", index=False)
    print(f"Saved combined summary to {root / 'case1_bounds.csv'}")


if __name__ == "__main__":
    main()
