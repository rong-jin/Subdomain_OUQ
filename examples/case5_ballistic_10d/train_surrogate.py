#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a 10D -> 1D surrogate from flat files:
  in.1 ... in.N
  out.1 ... out.N

Each input file contains lines like:
  RA1,2.39277e9
  RB1,1.81354e9
  ...
  RS1,1.25439

Key features:
- automatic train/val/test split
- normalizer fit on train only
- early stopping
- exports best_model.pth, forward_model.pth, forward_model.ts
- saves diagnostics CSV/NPZ/JSON compatible with existing plot_metrics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import default_rng
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from scipy.stats import ks_2samp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


DEFAULT_FEATURE_NAMES = [
    "RA1", "RB1", "Rn1", "RC1", "Rm1",
    "Rv0", "RE1", "RC0", "RG0", "RS1",
]


class RegressionNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        layers: List[nn.Module] = [nn.Linear(input_size, hidden_layers[0]), nn.SELU()]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]), nn.SELU()]
        layers += [nn.Linear(hidden_layers[-1], output_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ForwardModel(nn.Module):
    """raw-scale in -> normalize -> net -> denormalize"""
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1):
        super().__init__()
        self.net = RegressionNet(input_size, hidden_layers, output_size)
        self.register_buffer("x_min", torch.zeros(input_size))
        self.register_buffer("x_max", torch.ones(input_size))
        self.register_buffer("y_min", torch.zeros(output_size))
        self.register_buffer("y_max", torch.ones(output_size))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        y_norm = self.net(x_norm)
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min


class Normalizer:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_min = x_train.min(axis=0)
        self.x_max = x_train.max(axis=0)
        self.y_min = y_train.min(axis=0)
        self.y_max = y_train.max(axis=0)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min.tolist(),
            "x_max": self.x_max.tolist(),
            "y_min": self.y_min.tolist(),
            "y_max": self.y_max.tolist(),
        }


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)
    print(f"[saved] {path}", flush=True)


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}", flush=True)


def tensor_dataset_from_np(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())


def parse_hidden_layers(text: str) -> List[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("hidden-layers cannot be empty")
    return vals


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


@torch.no_grad()
def permutation_importance_val(
    model: nn.Module,
    X_val_n: np.ndarray,
    y_val_n: np.ndarray,
    normalizer: Normalizer,
    n_repeats: int = 5,
    device: str = "cpu",
) -> pd.DataFrame:
    model.eval()
    Xv = torch.from_numpy(X_val_n).float().to(device)
    y_pred_n = model(Xv).cpu().numpy()
    base = compute_metrics(
        normalizer.inverse_y(y_val_n),
        normalizer.inverse_y(y_pred_n),
    )["MSE"]

    rng = default_rng(2025)
    F = X_val_n.shape[1]
    rows = []
    for f in range(F):
        perm_mses = []
        for _ in range(n_repeats):
            X_perm = X_val_n.copy()
            rng.shuffle(X_perm[:, f])
            Xp = torch.from_numpy(X_perm).float().to(device)
            yp_n = model(Xp).cpu().numpy()
            perm_mse = compute_metrics(
                normalizer.inverse_y(y_val_n),
                normalizer.inverse_y(yp_n),
            )["MSE"]
            perm_mses.append(perm_mse)
        perm_mses = np.asarray(perm_mses, dtype=float)
        rows.append(
            {
                "feature": f + 1,
                "base_mse": base,
                "perm_mse_mean": float(perm_mses.mean()),
                "perm_mse_std": float(perm_mses.std(ddof=1) if perm_mses.size > 1 else 0.0),
                "delta_mse": float(perm_mses.mean() - base),
            }
        )
    return pd.DataFrame(rows)


def parse_input_file(filepath: str | Path, feature_names: List[str]) -> List[float] | None:
    """Parse one in.i file with lines 'name,value'."""
    data: OrderedDict[str, float] = OrderedDict()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",", 1)]
                if len(parts) != 2:
                    raise ValueError(f"line {line_no}: expected 'name,value', got {raw!r}")
                name, value_text = parts
                if name in data:
                    raise ValueError(f"duplicate feature '{name}' in {filepath}")
                data[name] = float(value_text)
    except Exception as e:
        print(f"[warn] failed to parse {filepath}: {e}", flush=True)
        return None

    missing = [name for name in feature_names if name not in data]
    extra = [name for name in data if name not in feature_names]
    if missing or extra:
        print(
            f"[warn] feature mismatch in {filepath} | missing={missing} extra={extra}",
            flush=True,
        )
        return None

    return [data[name] for name in feature_names]


def parse_output_file(filepath: str | Path) -> float | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception as e:
        print(f"[warn] failed to parse {filepath}: {e}", flush=True)
        return None


def load_flat_inout_data(
    data_dir: str | Path,
    feature_names: List[str],
    in_prefix: str = "in.",
    out_prefix: str = "out.",
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    data_dir = Path(data_dir)
    all_x: List[List[float]] = []
    all_y: List[List[float]] = []
    used_ids: List[int] = []

    in_files = sorted(data_dir.glob(f"{in_prefix}*"), key=lambda p: int(p.name.split(".")[-1]))
    if not in_files:
        raise FileNotFoundError(f"No files matching {in_prefix}* in {data_dir}")

    for in_file in in_files:
        try:
            idx = int(in_file.name.split(".")[-1])
        except ValueError:
            continue
        if start_idx is not None and idx < start_idx:
            continue
        if end_idx is not None and idx > end_idx:
            continue

        out_file = data_dir / f"{out_prefix}{idx}"
        if not out_file.exists():
            print(f"[warn] missing {out_file.name}; skip id={idx}", flush=True)
            continue

        x_vals = parse_input_file(in_file, feature_names)
        y_val = parse_output_file(out_file)
        if x_vals is None or y_val is None:
            continue

        x_arr = np.asarray(x_vals, dtype=np.float64)
        y_scalar = float(y_val)

        if not np.all(np.isfinite(x_arr)):
            print(f"[warn] non-finite input in {in_file.name}; skip id={idx}", flush=True)
            continue
        if not np.isfinite(y_scalar):
            print(f"[warn] non-finite output in {out_file.name}; skip id={idx}", flush=True)
            continue

        all_x.append(x_vals)
        all_y.append([y_scalar])
        used_ids.append(idx)

    if not all_x:
        raise RuntimeError("No valid input/output pairs were loaded.")

    print(f"[info] loaded {len(all_x)} valid samples from {data_dir}", flush=True)
    print(f"[info] id range: min={min(used_ids)} max={max(used_ids)}", flush=True)
    return np.asarray(all_x, dtype=np.float32), np.asarray(all_y, dtype=np.float32), used_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 10D->1D surrogate from flat in/out files.")
    parser.add_argument("--data-dir", type=str, default="Data", help="Directory containing in.i / out.i files.")
    parser.add_argument(
        "--feature-names",
        type=str,
        default=",".join(DEFAULT_FEATURE_NAMES),
        help="Comma-separated feature names in desired order.",
    )
    parser.add_argument("--in-prefix", type=str, default="in.")
    parser.add_argument("--out-prefix", type=str, default="out.")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-layers", type=str, default="256,256,256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[info] device = {device}", flush=True)
    print(f"[info] features = {feature_names}", flush=True)
    print(f"[info] hidden_layers = {hidden_layers}", flush=True)

    features, targets, used_ids = load_flat_inout_data(
        data_dir=args.data_dir,
        feature_names=feature_names,
        in_prefix=args.in_prefix,
        out_prefix=args.out_prefix,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    N, D = features.shape
    print(f"[info] data shape -> X: {features.shape}, y: {targets.shape}", flush=True)

    results_dir = args.results_dir or f"surrogate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(results_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_all = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    total_size = len(dataset_all)
    test_size = int(total_size * args.test_ratio)
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size - test_size
    if train_size <= 0:
        raise ValueError("train_size <= 0. Adjust val-ratio/test-ratio.")

    train_ds, val_ds, test_ds = random_split(
        dataset_all,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    X_train = train_ds[:][0].numpy(); y_train = train_ds[:][1].numpy()
    X_val = val_ds[:][0].numpy(); y_val = val_ds[:][1].numpy()
    X_test = test_ds[:][0].numpy(); y_test = test_ds[:][1].numpy()

    np.savez(
        os.path.join(results_dir, "splits_raw.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print("[saved] splits_raw.npz", flush=True)

    normalizer = Normalizer()
    normalizer.fit(X_train, y_train)

    Xtr_n = normalizer.transform_x(X_train); ytr_n = normalizer.transform_y(y_train)
    Xva_n = normalizer.transform_x(X_val);   yva_n = normalizer.transform_y(y_val)
    Xte_n = normalizer.transform_x(X_test);  yte_n = normalizer.transform_y(y_test)

    np.savez(
        os.path.join(results_dir, "splits_normalized.npz"),
        X_train_n=Xtr_n, y_train_n=ytr_n,
        X_val_n=Xva_n, y_val_n=yva_n,
        X_test_n=Xte_n, y_test_n=yte_n,
    )
    print("[saved] splits_normalized.npz", flush=True)

    save_json(normalizer.to_dict(), os.path.join(results_dir, "normalizer_stats.json"))
    save_json({"feature_names": feature_names, "used_ids": used_ids}, os.path.join(results_dir, "dataset_manifest.json"))

    train_loader = DataLoader(tensor_dataset_from_np(Xtr_n, ytr_n), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tensor_dataset_from_np(Xva_n, yva_n), batch_size=args.batch_size, shuffle=False)

    model = RegressionNet(input_size=D, hidden_layers=hidden_layers, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    no_improve = 0
    train_hist: List[float] = []
    val_hist: List[float] = []

    print("\n[train] start", flush=True)
    t0 = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        run_tr = 0.0
        n_tr = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            run_tr += loss.item() * xb.size(0)
            n_tr += xb.size(0)
        avg_tr = run_tr / max(1, n_tr)

        model.eval()
        run_va = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                run_va += loss.item() * xb.size(0)
                n_va += xb.size(0)
        avg_va = run_va / max(1, n_va)

        train_hist.append(avg_tr)
        val_hist.append(avg_va)
        print(f"epoch {epoch:4d} | train {avg_tr:.6e} | val {avg_va:.6e}", flush=True)

        if avg_va < best_val - 1e-12:
            best_val = avg_va
            no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_size": D,
                    "hidden_layers": hidden_layers,
                    "feature_names": feature_names,
                },
                os.path.join(results_dir, "best_model.pth"),
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early-stopping] no improvement for {args.patience} epochs.", flush=True)
                break
    train_time_sec = time.time() - t0
    print(f"[train] done in {train_time_sec:.2f}s", flush=True)

    save_csv(
        pd.DataFrame({"epoch": np.arange(1, len(train_hist) + 1), "train_mse": train_hist, "val_mse": val_hist}),
        os.path.join(results_dir, "loss_curve.csv"),
    )

    ckpt = torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device)
    best_model = RegressionNet(input_size=D, hidden_layers=hidden_layers, output_size=1).to(device)
    best_model.load_state_dict(ckpt["state_dict"])
    best_model.eval()

    @torch.no_grad()
    def predict_numpy(model_: nn.Module, Xn: np.ndarray) -> np.ndarray:
        xb = torch.from_numpy(Xn).float().to(device)
        yp_n = model_(xb).cpu().numpy()
        return normalizer.inverse_y(yp_n)

    ytr_pred = predict_numpy(best_model, Xtr_n)
    yva_pred = predict_numpy(best_model, Xva_n)
    yte_pred = predict_numpy(best_model, Xte_n)

    def save_split_diagnostics(split_name: str, X_raw: np.ndarray, y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> None:
        df = pd.DataFrame(
            {
                "y_true": y_true_raw.reshape(-1),
                "y_pred": y_pred_raw.reshape(-1),
                "residual": (y_pred_raw - y_true_raw).reshape(-1),
            }
        )
        for j in range(X_raw.shape[1]):
            df[f"X{j+1}"] = X_raw[:, j]
        save_csv(df, os.path.join(results_dir, f"diagnostics_{split_name}.csv"))

    save_split_diagnostics("train", X_train, y_train, ytr_pred)
    save_split_diagnostics("val", X_val, y_val, yva_pred)
    save_split_diagnostics("test", X_test, y_test, yte_pred)

    metrics = {
        "train": compute_metrics(y_train, ytr_pred),
        "val": compute_metrics(y_val, yva_pred),
        "test": compute_metrics(y_test, yte_pred),
        "train_time_sec": train_time_sec,
    }
    save_json(metrics, os.path.join(results_dir, "metrics.json"))

    if _HAS_SCIPY:
        rows = []
        for j in range(D):
            stat_tt, p_tt = ks_2samp(X_train[:, j], X_test[:, j])
            stat_vt, p_vt = ks_2samp(X_val[:, j], X_train[:, j])
            rows.append(
                {
                    "feature": f"X{j+1}",
                    "feature_name": feature_names[j],
                    "ks_train_vs_test": float(stat_tt),
                    "p_train_vs_test": float(p_tt),
                    "ks_val_vs_train": float(stat_vt),
                    "p_val_vs_train": float(p_vt),
                }
            )
        save_csv(pd.DataFrame(rows), os.path.join(results_dir, "ks_feature_distributions.csv"))

    df_pi = permutation_importance_val(best_model, Xva_n, yva_n, normalizer, n_repeats=5, device=device)
    df_pi["feature_name"] = feature_names
    save_csv(df_pi, os.path.join(results_dir, "permutation_importance_val.csv"))

    forward_model = ForwardModel(input_size=D, hidden_layers=hidden_layers, output_size=1).to("cpu")
    forward_model.net.load_state_dict(best_model.state_dict())
    forward_model.x_min.copy_(torch.from_numpy(normalizer.x_min))
    forward_model.x_max.copy_(torch.from_numpy(normalizer.x_max))
    forward_model.y_min.copy_(torch.from_numpy(normalizer.y_min))
    forward_model.y_max.copy_(torch.from_numpy(normalizer.y_max))

    torch.save(
        {
            "state_dict": forward_model.state_dict(),
            "input_size": D,
            "hidden_layers": hidden_layers,
            "feature_names": feature_names,
        },
        os.path.join(results_dir, "forward_model.pth"),
    )
    print("[saved] forward_model.pth", flush=True)

    forward_model.eval()
    example = torch.zeros(1, D)
    ts = torch.jit.trace(forward_model, example)
    ts.save(os.path.join(results_dir, "forward_model.ts"))
    print("[saved] forward_model.ts", flush=True)

    with torch.no_grad():
        y_old = normalizer.inverse_y(best_model(torch.from_numpy(Xte_n).float().to(device)).cpu().numpy())
        y_new = forward_model(torch.from_numpy(X_test).float()).cpu().numpy()
    diff = np.abs(y_old - y_new)
    with open(os.path.join(results_dir, "consistency.txt"), "w", encoding="utf-8") as f:
        f.write(f"max_abs_diff={diff.max():.6e}, mean_abs_diff={diff.mean():.6e}\n")
    print(f"[consistency] max |Δ|={diff.max():.3e}, mean |Δ|={diff.mean():.3e}", flush=True)
    print(f"[done] all artifacts saved under: {results_dir}", flush=True)


if __name__ == "__main__":
    main()
