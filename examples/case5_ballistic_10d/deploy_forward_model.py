#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic deploy script for forward_model.pth exported by train_surrogate_10d.py.
- infers input_size and hidden_layers from checkpoint metadata
- supports either direct vectors or parsing an in.i file
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn


class RegressionNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.SELU()]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]), nn.SELU()]
        layers += [nn.Linear(hidden_layers[-1], output_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ForwardModel(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int = 1):
        super().__init__()
        self.net = RegressionNet(input_size, hidden_layers, output_size)
        self.register_buffer("x_min", torch.zeros(input_size))
        self.register_buffer("x_max", torch.ones(input_size))
        self.register_buffer("y_min", torch.zeros(output_size))
        self.register_buffer("y_max", torch.ones(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        y_norm = self.net(x_norm)
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min


def load_forward_model(path: str | Path = Path(__file__).resolve().parent / "models" / "forward_model.pth", device: str = "cpu"):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        input_size = int(ckpt.get("input_size", len(ckpt["state_dict"]["x_min"])))
        hidden_layers = ckpt.get("hidden_layers", [200, 200, 200, 200])
        feature_names = ckpt.get("feature_names", [f"X{i+1}" for i in range(input_size)])
        sd = ckpt["state_dict"]
    else:
        # fallback for old pure-state_dict files
        sd = ckpt
        input_size = int(sd["x_min"].shape[0])
        hidden_layers = [200, 200, 200, 200]
        feature_names = [f"X{i+1}" for i in range(input_size)]

    model = ForwardModel(input_size=input_size, hidden_layers=hidden_layers)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, feature_names


def warn_out_of_range(X, model):
    xmin = model.x_min.cpu().numpy()
    xmax = model.x_max.cpu().numpy()
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    below = (X < xmin).any(axis=1)
    above = (X > xmax).any(axis=1)
    if below.any() or above.any():
        print("[warn] some inputs are outside training range; model will extrapolate.")
        print("       min:", xmin)
        print("       max:", xmax)


@torch.inference_mode()
def predict_one(x, model, clamp: bool = False, device: str = "cpu") -> float:
    x = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(1, -1)
    if clamp:
        x = torch.max(torch.min(x, model.x_max), model.x_min)
    y = model(x)
    return float(y.item())


@torch.inference_mode()
def predict_batch(X, model, clamp: bool = False, device: str = "cpu") -> np.ndarray:
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 1:
        X = X[None, :]
    if clamp:
        X = torch.max(torch.min(X, model.x_max), model.x_min)
    y = model(X)
    return y.cpu().numpy().reshape(-1, 1)


def parse_input_file(filepath: str | Path, feature_names: List[str]) -> List[float]:
    data: OrderedDict[str, float] = OrderedDict()
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                raise ValueError(f"line {line_no}: expected 'name,value', got {raw!r}")
            name, value_text = parts
            data[name] = float(value_text)

    missing = [name for name in feature_names if name not in data]
    extra = [name for name in data if name not in feature_names]
    if missing or extra:
        raise ValueError(f"feature mismatch | missing={missing} extra={extra}")
    return [data[name] for name in feature_names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with exported forward_model.pth")
    parser.add_argument("--model", type=str, default=str(Path(__file__).resolve().parent / "models" / "forward_model.pth"), help="Path to forward_model.pth")
    parser.add_argument("--input-file", type=str, default=None, help="Path to one in.i file")
    parser.add_argument(
        "--x",
        nargs="*",
        type=float,
        default=None,
        help="Raw input vector. Leave empty when using --input-file.",
    )
    parser.add_argument("--clamp", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model, feature_names = load_forward_model(args.model, device=args.device)
    print("[info] feature order:", feature_names)

    if args.input_file is not None:
        x = parse_input_file(args.input_file, feature_names)
    elif args.x is not None and len(args.x) > 0:
        x = args.x
    else:
        raise ValueError("Provide either --input-file or --x.")

    if len(x) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} inputs, got {len(x)}")

    warn_out_of_range([x], model)
    y = predict_one(x, model=model, clamp=args.clamp, device=args.device)
    print("y =", y)


if __name__ == "__main__":
    main()
