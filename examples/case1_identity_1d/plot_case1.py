#!/usr/bin/env python3
"""Plot Case 1 PDF and OUQ-bound summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_case1 import make_case, true_pof
from subdomain_ouq.plotting import plot_pdf_with_failure_region, set_basic_style


def plot_bounds(summary_csv: Path, outdir: Path):
    set_basic_style()
    df = pd.read_csv(summary_csv)
    if "distribution" not in df.columns:
        df["distribution"] = summary_csv.parent.name
    for dist in sorted(df["distribution"].unique()):
        dfd = df[df["distribution"] == dist].copy()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for r in sorted(dfd["r"].unique()):
            dfr = dfd[dfd["r"] == r].sort_values("K")
            ax.plot(dfr["K"], dfr["upper"], marker="o", label=f"r={r} upper")
            ax.plot(dfr["K"], dfr["lower"], marker="o", linestyle="--", label=f"r={r} lower")
        ref = float(dfd["ref_pof"].iloc[0]) if "ref_pof" in dfd else None
        if ref is not None:
            ax.axhline(ref, linestyle=":", label=f"reference={ref:.4f}")
        ax.set_xlabel("Number of subdomains K")
        ax.set_ylabel("PoF")
        ax.set_title(f"Case 1: {dist}")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"case1_bounds_{dist}.png", dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "case1_bounds.csv"))
    ap.add_argument("--outdir", default=str(ROOT / "outputs" / "case1_identity_1d" / "figures"))
    ap.add_argument("--pdfs", action="store_true", help="Also regenerate PDF/failure-region plots.")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    plot_bounds(Path(args.summary), outdir)
    if args.pdfs:
        for name in ["normal", "uniform", "weibull", "bimodal"]:
            prob = make_case(name)
            plot_pdf_with_failure_region(prob.dist, prob.a, prob.b, prob.c, outdir / f"pdf_{prob.name}.png", title=prob.label)
    print(f"Saved Case 1 figures under {outdir}")


if __name__ == "__main__":
    main()
