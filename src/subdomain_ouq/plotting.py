"""Common plotting utilities for the examples."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_basic_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    })


def plot_bounds_summary(csv_path: str | Path, out_path: str | Path, title: str = "OUQ bounds", logy: bool = False):
    set_basic_style()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in sorted(df["r"].unique()):
        dfr = df[df["r"] == r].sort_values("K")
        ax.plot(dfr["K"], dfr["upper"], marker="o", label=f"r={r} upper")
        ax.plot(dfr["K"], dfr["lower"], marker="o", linestyle="--", label=f"r={r} lower")
    if "ref_pof" in df.columns and df["ref_pof"].notna().any():
        ref = float(df["ref_pof"].dropna().iloc[0])
        ax.axhline(ref, linestyle=":", label=f"reference={ref:.4g}")
    ax.set_xlabel("Number of subdomains K")
    ax.set_ylabel("PoF")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_history(history_dir: str | Path, out_path: Optional[str | Path] = None, ref_pof: Optional[float] = None):
    set_basic_style()
    history_dir = Path(history_dir)
    if out_path is None:
        out_path = history_dir / "convergence.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    up = history_dir / "upper_history.csv"
    lo = history_dir / "lower_history.csv"
    if up.exists():
        df = pd.read_csv(up)
        col = "upper_best_pof" if "upper_best_pof" in df.columns else df.columns[-1]
        ax.plot(df["iter"], df[col], label="upper best")
    if lo.exists():
        df = pd.read_csv(lo)
        col = "lower_best_pof" if "lower_best_pof" in df.columns else df.columns[-1]
        ax.plot(df["iter"], df[col], label="lower best")
    if ref_pof is not None:
        ax.axhline(ref_pof, linestyle=":", label=f"reference={ref_pof:.4g}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("PoF")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_pdf_with_failure_region(dist, a: float, b: float, c: float, out_path: str | Path, title: str = ""):
    set_basic_style()
    x = np.linspace(a, b, 2000)
    y = dist.pdf(x)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, label="PDF")
    ax.axvline(c, linestyle="--", label=f"Yc={c:g}")
    xs = x[x >= c]
    if xs.size:
        ax.fill_between(xs, 0, dist.pdf(xs), alpha=0.25, label="failure region")
    ax.set_xlabel("x")
    ax.set_ylabel("PDF")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
