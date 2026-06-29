#!/usr/bin/env python3
"""Run Case 2: five-dimensional nonlinear smooth OUQ example."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subdomain_ouq import MarginalSpec, truncated_normal, DEControls, OUQRunControls, run_sweep
from subdomain_ouq.io import ensure_dir


@dataclass
class Smooth5DProblem:
    c_fail: float = 0.8
    failure: str = "ge"

    def __post_init__(self) -> None:
        specs = [
            ("X1", -4.0, 3.0, -1.0, 1.2),
            ("X2", -2.0, 4.0, 0.5, 0.7),
            ("X3", -3.5, 5.0, 1.6, 0.9),
            ("X4", -5.0, 2.0, -0.8, 1.5),
            ("X5", -2.5, 2.5, 0.0, 0.6),
        ]
        self.marginals = [MarginalSpec(name, a, b, truncated_normal(a, b, mu, sig)) for name, a, b, mu, sig in specs]
        self.dists = [m.dist for m in self.marginals]
        self.bounds = [(m.a, m.b) for m in self.marginals]
        self.d = len(self.marginals)

    def g_eval(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        return 0.7 * x1 + 0.35 * x2 ** 2 - 0.25 * x3 * x4 + 0.2 * (x3 ** 3) / (1.0 + x3 ** 2) + np.sin(x5)


def compute_reference_pof(prob: Smooth5DProblem, n_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    X = np.zeros((int(n_samples), prob.d), dtype=float)
    for i, dist in enumerate(prob.dists):
        X[:, i] = dist.rvs(size=int(n_samples), random_state=int(rng.integers(0, 2**32 - 1)))
    return float(np.mean(prob.g_eval(X) >= prob.c_fail))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Case 2 five-dimensional smooth OUQ example.")
    parser.add_argument("--K-list", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--r-list", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-its", type=int, default=50_000)
    parser.add_argument("--n-final", type=int, default=500_000)
    parser.add_argument("--reference-samples", type=int, default=200_000)
    parser.add_argument("--reference-pof", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "case2_5d_smooth"))
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--exact-threshold", type=int, default=None)
    parser.add_argument("--canonical-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    prob = Smooth5DProblem()
    ref_pof = args.reference_pof
    if ref_pof is None:
        ref_pof = compute_reference_pof(prob, args.reference_samples, seed=args.seed)
        print(f"Reference PoF estimate = {ref_pof:.8f}")

    threshold = args.n_its if args.exact_threshold is None else args.exact_threshold
    de = DEControls(popsize=args.popsize, maxiter=args.maxiter, polish=False, seed_upper=args.seed, seed_lower=args.seed + 1)
    run = OUQRunControls(
        N_mc_opt=args.n_its,
        N_mc_final=args.n_final,
        exact_threshold_opt=threshold,
        exact_threshold_final=threshold,
        canonical_eps=args.canonical_eps,
        crn_method="random",
        use_crn=True,
    )
    outdir = ensure_dir(args.output_dir)
    sweep = run_sweep(prob, args.K_list, args.r_list, ref_pof, outdir, de=de, run=run, names=[m.name for m in prob.marginals], verbose=not args.quiet)
    print(sweep.dataframe)
    print(f"Saved Case 2 results to {outdir}")


if __name__ == "__main__":
    main()
