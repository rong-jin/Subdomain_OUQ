#!/usr/bin/env python3
"""Plot Case 4 roof-truss OUQ and McDiarmid screening results."""
from __future__ import annotations

import argparse
import ast
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

from subdomain_ouq.plotting import set_basic_style


def _extract_active_K_r(row):
    if "K" in row and "r" in row:
        return int(row["K"]), int(row["r"])
    name = str(row.get("case", ""))
    m = re.search(r"K(\d+)_r(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    if "K_vec" in row and "r_vec" in row:
        K_vec = ast.literal_eval(row["K_vec"]) if isinstance(row["K_vec"], str) else row["K_vec"]
        r_vec = ast.literal_eval(row["r_vec"]) if isinstance(row["r_vec"], str) else row["r_vec"]
        K = max(int(x) for x in K_vec)
        r = max(int(x) for x in r_vec)
        return K, r
    return None, None


def plot_bounds(summary_csv: Path, out_path: Path, ref_pof: float | None = None):
    set_basic_style()
    df = pd.read_csv(summary_csv)
    ks, rs = [], []
    for _, row in df.iterrows():
        K, r = _extract_active_K_r(row)
        ks.append(K)
        rs.append(r)
    df["K_active"] = ks
    df["r_active"] = rs
    df = df.dropna(subset=["K_active", "r_active"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in sorted(df["r_active"].unique()):
        dfr = df[df["r_active"] == r].sort_values("K_active")
        ax.plot(dfr["K_active"], dfr["upper"], marker="o", label=f"r={int(r)} upper")
        ax.plot(dfr["K_active"], dfr["lower"], marker="o", linestyle="--", label=f"r={int(r)} lower")
    if ref_pof is not None:
        ax.axhline(ref_pof, linestyle=":", label=f"reference={ref_pof:.3e}")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("Active-dimension subdomains K")
    ax.set_ylabel("PoF")
    ax.set_title("Case 4 roof-truss OUQ bounds")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_screening(screen_csv: Path, out_path: Path):
    set_basic_style()
    df = pd.read_csv(screen_csv).sort_values("share", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(df["name"], df["share"])
    ax.set_xlabel("Input variable")
    ax.set_ylabel("Normalized McDiarmid subdiameter")
    ax.set_title("Case 4 active-variable screening")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "case4_bounds.csv"))
    ap.add_argument("--screening", default=None)
    ap.add_argument("--ref-pof", type=float, default=5.0249e-7)
    ap.add_argument("--outdir", default=str(ROOT / "outputs" / "case4_roof_truss" / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    summary = Path(args.summary)
    if summary.exists():
        plot_bounds(summary, outdir / "case4_bounds.png", ref_pof=args.ref_pof)
        print(f"Saved {outdir / 'case4_bounds.png'}")
    if args.screening:
        screen_csv = Path(args.screening)
        plot_screening(screen_csv, outdir / "case4_mcdiarmid_screening.png")
        print(f"Saved {outdir / 'case4_mcdiarmid_screening.png'}")


if __name__ == "__main__":
    main()
