#!/usr/bin/env python3
"""Plot Case 5 ballistic-impact OUQ outputs."""
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

from subdomain_ouq.plotting import plot_bounds_summary, set_basic_style


def plot_threshold_scan(csv_path: Path, outdir: Path):
    set_basic_style()
    df = pd.read_csv(csv_path)
    ref_col = "ref_pof_final_mc" if "ref_pof_final_mc" in df.columns else "Reference PoF"
    if ref_col in df.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(df[ref_col], df["upper"], marker="o", label="upper")
        ax.plot(df[ref_col], df["lower"], marker="o", linestyle="--", label="lower")
        ax.set_xscale("log")
        ax.set_xlabel("Reference PoF")
        ax.set_ylabel("OUQ bound")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "case5_bounds_vs_pof.png", dpi=300)
        plt.close(fig)
    if "c_fail" in df.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(df["c_fail"], df["upper"], marker="o", label="upper")
        ax.plot(df["c_fail"], df["lower"], marker="o", linestyle="--", label="lower")
        ax.set_xlabel("Yc")
        ax.set_ylabel("OUQ bound")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "case5_bounds_vs_yc.png", dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "case5_bounds.csv"), help="ouq_bounds_summary.csv from run_case5.py")
    ap.add_argument("--scan", default=None, help="ouq_vs_pof_scan.csv from scan_thresholds.py")
    ap.add_argument("--outdir", default=str(ROOT / "outputs" / "case5_ballistic_10d" / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if args.summary:
        plot_bounds_summary(Path(args.summary), outdir / "case5_bounds.png", title="Case 5 ballistic-impact OUQ bounds", logy=True)
        print(f"Saved {outdir / 'case5_bounds.png'}")
    if args.scan:
        plot_threshold_scan(Path(args.scan), outdir)
        print(f"Saved threshold-scan figures under {outdir}")


if __name__ == "__main__":
    main()
