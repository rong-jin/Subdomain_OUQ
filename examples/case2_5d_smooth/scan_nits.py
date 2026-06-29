#!/usr/bin/env python3
"""Scan ITS sample sizes for Case 2."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from run_case2 import Smooth5DProblem, compute_reference_pof
from subdomain_ouq import DEControls, OUQRunControls, run_ouq_canonical_de
from subdomain_ouq.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Case 2 bounds for a list of ITS sample sizes.")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--n-list", nargs="+", type=int, default=[5000, 7000, 10000, 15000, 20000, 35000, 50000, 100000])
    parser.add_argument("--reference-pof", type=float, default=0.2948)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "case2_5d_smooth" / "scan_nits"))
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    prob = Smooth5DProblem()
    outdir = ensure_dir(args.output_dir)
    rows = []
    for n in args.n_list:
        de = DEControls(popsize=args.popsize, maxiter=args.maxiter, polish=False, seed_upper=args.seed, seed_lower=args.seed + 1)
        run = OUQRunControls(N_mc_opt=n, N_mc_final=n, exact_threshold_opt=n, exact_threshold_final=n, use_crn=True)
        res = run_ouq_canonical_de(prob, args.K, args.r, de=de, run=run, verbose=not args.quiet)
        rows.append({"N_ITS": n, "K": args.K, "r": args.r, "upper": res["upper"], "lower": res["lower"], "ref_pof": args.reference_pof})
        pd.DataFrame(rows).to_csv(outdir / "scan_nits.csv", index=False)
    print(f"Saved scan to {outdir / 'scan_nits.csv'}")


if __name__ == "__main__":
    main()
