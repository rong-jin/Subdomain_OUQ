#!/usr/bin/env python3
"""Run only the McDiarmid screening step for Case 4."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_case4 import RoofTrussProblem, mcdiarmid_subdiameters
from subdomain_ouq.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run McDiarmid screening for Case 4.")
    parser.add_argument("--eta", type=float, default=1.25e5)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "case4_roof_truss" / "screening"))
    parser.add_argument("--exclude-screen-names", type=str, default="q,l")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prob = RoofTrussProblem(eta=args.eta)
    F = lambda x: float(prob.g_eval(np.asarray(x).reshape(1, -1))[0])
    exclude = [s.strip() for s in args.exclude_screen_names.split(",") if s.strip()]
    rows = mcdiarmid_subdiameters(prob.names, prob.bounds, F, exclude_names=exclude, seed=args.seed, maxiter=args.maxiter, popsize=args.popsize, verbose=True)
    outdir = ensure_dir(args.output_dir)
    pd.DataFrame(rows).to_csv(outdir / "mcdiarmid_screening.csv", index=False)
    print(f"Saved screening to {outdir / 'mcdiarmid_screening.csv'}")


if __name__ == "__main__":
    main()
