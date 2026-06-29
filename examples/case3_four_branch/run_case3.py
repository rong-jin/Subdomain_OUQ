#!/usr/bin/env python3
"""Case 3: two-dimensional non-smooth four-branch OUQ benchmark.

The script merges the original baseline and thresholded Case_3 workflows.  The
OUQ-facing response is g_eval = -min(g1, g2, g3, g4), so failure is
``g_eval >= yc``.  The paper uses yc=0 and yc=2.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from subdomain_ouq.optimizer import get_adaptive_parameters, run_grid, run_single_configuration
from subdomain_ouq.io import save_ouq_bounds_summary


@dataclass
class TruncatedStandardNormalMarginal:
    name: str
    a: float = -5.0
    b: float = 5.0
    mu: float = 0.0
    sigma: float = 1.0


@dataclass
class FourBranch2D:
    marginals: List[TruncatedStandardNormalMarginal]
    p_const: float = 6.0
    c_fail: float = 0.0

    def __post_init__(self):
        self.d = len(self.marginals)
        if self.d != 2:
            raise ValueError("FourBranch2D expects exactly two inputs.")
        self.dists = []
        for m in self.marginals:
            a_std = (m.a - m.mu) / m.sigma
            b_std = (m.b - m.mu) / m.sigma
            self.dists.append(truncnorm(a=a_std, b=b_std, loc=m.mu, scale=m.sigma))

    def g_sys(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        x1, x2 = X[:, 0], X[:, 1]
        rt2 = np.sqrt(2.0)
        p = float(self.p_const)
        g1 = 3.0 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / rt2
        g2 = 3.0 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / rt2
        g3 = (x1 - x2) + p / rt2
        g4 = (x2 - x1) + p / rt2
        return np.minimum.reduce([g1, g2, g3, g4])

    def g_eval(self, X: np.ndarray) -> np.ndarray:
        return -self.g_sys(X)


def setup_problem(yc: float = 0.0, p_const: float = 6.0, a: float = -5.0, b: float = 5.0) -> FourBranch2D:
    return FourBranch2D(
        marginals=[TruncatedStandardNormalMarginal("x1", a=a, b=b), TruncatedStandardNormalMarginal("x2", a=a, b=b)],
        p_const=p_const,
        c_fail=yc,
    )


def compute_reference_pof(prob: FourBranch2D, n_samples: int = 2_000_000, seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, prob.d), dtype=float)
    for i in range(prob.d):
        try:
            X[:, i] = prob.dists[i].rvs(size=n_samples, random_state=rng)
        except TypeError:
            rs = np.random.RandomState(rng.integers(0, 2**32 - 1, dtype=np.uint32))
            X[:, i] = prob.dists[i].rvs(size=n_samples, random_state=rs)
    return float(np.mean(prob.g_eval(X) >= prob.c_fail))


def run_case_grid(args):
    prob = setup_problem(yc=args.yc, p_const=args.p, a=args.a, b=args.b)
    ref = args.ref_pof if args.ref_pof is not None else compute_reference_pof(prob, n_samples=args.n_ref, seed=args.seed)
    outdir = Path(args.outdir) / f"yc_{args.yc:g}".replace(".", "p")
    df = run_grid(prob, args.K_list, args.r_list, ref, output_dir=outdir, popsize=args.popsize, maxiter=args.maxiter, side="ge")
    df.to_csv(outdir / "results.csv", index=False)
    save_ouq_bounds_summary(df, ref, outdir)
    print(f"Saved Case 3 results to {outdir}")
    return df


def run_case_single(args):
    prob = setup_problem(yc=args.yc, p_const=args.p, a=args.a, b=args.b)
    ref = args.ref_pof if args.ref_pof is not None else compute_reference_pof(prob, n_samples=args.n_ref, seed=args.seed)
    outdir = Path(args.outdir) / f"yc_{args.yc:g}_single".replace(".", "p")
    params = get_adaptive_parameters(prob.d, args.K, args.r, popsize=args.popsize, maxiter=args.maxiter, exact_threshold=args.exact_threshold)
    res = run_single_configuration(prob, args.K, args.r, params, ref, outdir, verbose=True, side="ge")
    pd.DataFrame([res]).to_csv(outdir / "results.csv", index=False)
    save_ouq_bounds_summary(pd.DataFrame([res]), ref, outdir)
    print(res)


def scan_yc(args):
    rows = []
    for yc in args.yc_values:
        prob = setup_problem(yc=yc, p_const=args.p, a=args.a, b=args.b)
        pf = compute_reference_pof(prob, n_samples=args.n_ref, seed=args.seed)
        rows.append({"yc": yc, "ref_pof": pf})
        print(f"yc={yc:g} ref_pof={pf:.8e}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "reference_scan_yc.csv", index=False)


def make_parser():
    ap = argparse.ArgumentParser(description="Run the 2D four-branch OUQ example.")
    ap.add_argument("--mode", choices=["grid", "single", "ref", "scan-yc"], default="grid")
    ap.add_argument("--yc", type=float, default=0.0)
    ap.add_argument("--p", type=float, default=6.0)
    ap.add_argument("--a", type=float, default=-5.0)
    ap.add_argument("--b", type=float, default=5.0)
    ap.add_argument("--K-list", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--r-list", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--r", type=int, default=1)
    ap.add_argument("--n-ref", type=int, default=2_000_000)
    ap.add_argument("--ref-pof", type=float, default=None)
    ap.add_argument("--exact-threshold", type=int, default=1_000_000)
    ap.add_argument("--popsize", type=int, default=50)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", default="outputs/case3_four_branch")
    ap.add_argument("--yc-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    return ap


def main():
    args = make_parser().parse_args()
    if args.mode == "grid":
        run_case_grid(args)
    elif args.mode == "single":
        run_case_single(args)
    elif args.mode == "ref":
        prob = setup_problem(yc=args.yc, p_const=args.p, a=args.a, b=args.b)
        pf = compute_reference_pof(prob, n_samples=args.n_ref, seed=args.seed)
        print(f"Reference PoF ~= {pf:.8e}")
    elif args.mode == "scan-yc":
        scan_yc(args)


if __name__ == "__main__":
    main()
