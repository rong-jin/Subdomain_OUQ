"""Input/output helpers for examples."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class Tee:
    """Mirror stdout/stderr to console and a log file."""

    def __init__(self, logfile, mode="w", encoding="utf-8", ascii_only=True, level="INFO"):
        self.log = open(logfile, mode, encoding=encoding, buffering=1)
        self.console_out = sys.stdout
        self.ascii_only = ascii_only
        self.level = level
        self._partial = ""

    def _ts(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def write(self, s):
        if self.ascii_only:
            s = s.replace("✓", "OK").replace("→", "->").replace("≈", "~=").replace("✗", "X").replace("ℹ", "i")
        for line in s.splitlines(True):
            if line.endswith("\n"):
                body = self._partial + line[:-1]
                self._partial = ""
                out = f"{self._ts()} - {self.level} - {body}\n" if body else "\n"
                self.console_out.write(out)
                self.log.write(out)
            else:
                self._partial += line

    def flush(self):
        self.console_out.flush()
        self.log.flush()

    def close(self):
        if self._partial:
            out = f"{self._ts()} - {self.level} - {self._partial}\n"
            self.console_out.write(out)
            self.log.write(out)
            self._partial = ""
        self.log.close()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    return p


def save_marginals_csv(marginals: List[Tuple[np.ndarray, np.ndarray]], outdir: str | Path, K: int, r: int, mode: str, names: Optional[List[str]] = None):
    out = ensure_dir(outdir)
    for i, (xs, ts) in enumerate(marginals, start=1):
        suffix = f"_{names[i-1]}" if names and i - 1 < len(names) else ""
        pd.DataFrame({"x": np.asarray(xs, dtype=float), "t": np.asarray(ts, dtype=float)}).to_csv(out / f"atoms_K{K}_r{r}_{mode}_dim{i}{suffix}.csv", index=False)


def save_ouq_bounds_summary(df: pd.DataFrame, ref_pof: Optional[float], output_dir: str | Path, filename: str = "ouq_bounds_summary.csv") -> Path:
    required = ["K", "r", "upper", "lower"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[required].copy()
    out["ref_pof"] = ref_pof if ref_pof is not None else np.nan
    out = out.sort_values(["K", "r"]).reset_index(drop=True)
    path = ensure_dir(output_dir) / filename
    out.to_csv(path, index=False)
    return path
