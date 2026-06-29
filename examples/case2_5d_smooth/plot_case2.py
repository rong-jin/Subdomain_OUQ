#!/usr/bin/env python3
"""Plot Case 2 summaries, DE histories, and optional ITS scans."""
from __future__ import annotations

import argparse
import re
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

from subdomain_ouq.plotting import plot_bounds_summary, plot_history, set_basic_style


def plot_all_histories(root: Path, ref_pof=None):
    for hdir in sorted(root.glob("history_K*_r*")):
        plot_history(hdir, hdir / "convergence.png", ref_pof=ref_pof)


def plot_nits_scan(csv_path: Path, out_path: Path):
    set_basic_style()
    df = pd.read_csv(csv_path)
    ncol = "N_its" if "N_its" in df.columns else ("N" if "N" in df.columns else df.columns[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for col in df.columns:
        if col == ncol or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        ax.plot(df[ncol], df[col], marker="o", label=col)
    ax.set_xscale("log")
    ax.set_xlabel("NITS")
    ax.set_ylabel("PoF / bound")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs" / "case2_5d_smooth"), help="Case 2 output directory from run_case2.py")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--histories", action="store_true")
    ap.add_argument("--nits-csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    summary = Path(args.summary) if args.summary else ROOT / "results" / "case2_bounds.csv"
    out = Path(args.out) if args.out else ROOT / "outputs" / "case2_5d_smooth" / "case2_bounds.png"
    if summary.exists():
        plot_bounds_summary(summary, out, title="Case 2 five-dimensional smooth OUQ bounds")
        print(f"Saved {out}")
        ref = None
        try:
            df = pd.read_csv(summary)
            if "ref_pof" in df and df["ref_pof"].notna().any():
                ref = float(df["ref_pof"].dropna().iloc[0])
        except Exception:
            ref = None
        if args.histories:
            plot_all_histories(root, ref_pof=ref)
            print(f"Updated convergence figures under {root}")
    if args.nits_csv:
        csv_path = Path(args.nits_csv)
        plot_nits_scan(csv_path, csv_path.with_suffix(".png"))
        print(f"Saved {csv_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
