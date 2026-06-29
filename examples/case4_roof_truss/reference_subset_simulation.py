#!/usr/bin/env python3
"""Run only the subset-simulation reference calculation for Case 4."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from run_case4 import RoofTrussProblem, subset_simulation_reference
from subdomain_ouq.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset Simulation reference for Case 4.")
    parser.add_argument("--eta", type=float, default=1.25e5)
    parser.add_argument("--N", type=int, default=100000)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--sigma-prop", type=float, default=0.9)
    parser.add_argument("--max-levels", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "case4_roof_truss" / "reference"))
    args = parser.parse_args()
    prob = RoofTrussProblem(eta=args.eta)
    out = subset_simulation_reference(prob, N=args.N, p0=args.p0, sigma_prop=args.sigma_prop, max_levels=args.max_levels, seed=args.seed, verbose=True)
    outdir = ensure_dir(args.output_dir)
    (outdir / "reference_subset_simulation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved reference to {outdir / 'reference_subset_simulation.json'}")


if __name__ == "__main__":
    main()
