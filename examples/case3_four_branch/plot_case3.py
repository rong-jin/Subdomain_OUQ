#!/usr/bin/env python3
"""Plot Case 3 four-branch OUQ summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from subdomain_ouq.plotting import plot_bounds_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "case3_bounds_yc0.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--logy", action="store_true", default=True)
    args = ap.parse_args()
    summary = Path(args.summary)
    out = Path(args.out) if args.out else ROOT / "outputs" / "case3_four_branch" / "case3_bounds.png"
    plot_bounds_summary(summary, out, title="Case 3 four-branch OUQ bounds", logy=args.logy)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
