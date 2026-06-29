#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10D Multi-Interval OUQ (uniform ±10% around user-specified means) + trained forward model.

Based on the user's 5D OUQ script, generalized to:
- D = 10 inputs
- bounds generated from mean ± 10% *abs(mean)*
- self-contained forward_model.pth exported from surrogate training
- subdomain OUQ via canonical moments + Jacobi + DE + adaptive exact/MC

Notes
-----
1) You MUST keep MODEL_INPUT_SIZE and MODEL_HIDDEN_LAYERS consistent with the surrogate that
   produced forward_model.pth. If your recent training used 256x4 hidden layers, keep the default.
2) c_fail must be set to the failure threshold on the scalar surrogate output.
3) For 10D, start with small K/r (e.g. K=[1,2], r=[0,1]) before larger sweeps.
"""

import os, sys, json, math, logging, platform, random, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy.linalg import eigh
from scipy.optimize import differential_evolution
from scipy.stats import qmc

# ==================== USER SETTINGS ====================
# Path to your trained self-contained forward model
WEIGHTS_PATH = str(Path(__file__).resolve().parent / "models" / "forward_model.pth")

# IMPORTANT: keep these in sync with the surrogate training run.
# If your current 10D training used 256,256,256,256, keep this as is.
MODEL_INPUT_SIZE = 10
MODEL_HIDDEN_LAYERS = [256, 256, 256, 256]
MODEL_OUTPUT_SIZE = 1

# Failure threshold on the scalar surrogate output
C_FAIL = 0.93

# Means (user supplied)
INPUT_MEANS = {
    "RA1": 2251710e3,
    "RB1": 1683460e3,
    "Rn1": 0.242,
    "RC1": 0.013,
    "Rm1": 1.550,
    "Rv0": -20000.0,
    "RE1": 4.5e11,
    "RC0": 4.52e5,
    "RG0": 1.54e0,
    "RS1": 1.242e0,
}
BOUNDS_RELATIVE_PCT = 0.10

# Legacy OUQ sweep (kept for compatibility; unused in fixed-(K,r) scan mode)
K_LIST = [1, 2, 4]
R_LIST = [0, 1, 2]

# Fixed (K,r) threshold / PoF scan settings
FIXED_K = 4
FIXED_R = 1
TARGET_POFS = np.array([1e-1, 5e-2, 1e-2, 5e-3, 1e-3], dtype=float)
N_POF_SCAN = 1_000_000          # pilot MC size for Y-quantiles / c_fail construction
REF_POF_MC_SEED = 123         # seed used for pilot and reference MC
RECOMPUTE_REF_POF_FINAL = True  # if True, recompute reference PoF at each c_fail using N_MC_FINAL

# Reference MC + optimization MC
N_MC_OPT = 50_000
N_MC_FINAL = 500_000
RNG_SEED = 2025
EXACT_TOTAL_ATOMS_THRESHOLD = 100_000

# Runtime / DE
USE_GPU = False
ANN_BATCH_SIZE = 65536
OUTDIR_PREFIX = "ouq10d_fixedKr_scan_pof"
LOG_LEVEL = "INFO"
LOG_TO_FILE = True

# DE settings
DE_MODE = "manual"  # 'manual' or 'auto'
BUDGET_PER_VAR = 1000
POP_PER_VAR = 20
MANUAL_POPSIZE = 50
MANUAL_MAXITER = 200

# Numerical tolerances
CANONICAL_EPS = 1e-8
MASS_CONSERVATION_TOL = 1e-6
DE_STRATEGY = 'best1bin'
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_TOL = 1e-6
DE_ATOL = 1e-8
DE_POLISH = False

# ======================================================

def make_pm_bounds_from_means(mean_dict: Dict[str, float], rel_pct: float):
    names = list(mean_dict.keys())
    mu = np.array([mean_dict[k] for k in names], dtype=float)
    delta = rel_pct * np.abs(mu)
    lo = mu - delta
    hi = mu + delta
    return names, mu, lo, hi

NAMES, MU, A_LO, B_HI = make_pm_bounds_from_means(INPUT_MEANS, BOUNDS_RELATIVE_PCT)
D = len(NAMES)

# ==================== DE params ====================
def _derive_de_params(nvar: int, budget_per_var: int = BUDGET_PER_VAR, pop_per_var: int = POP_PER_VAR):
    popsize = max(5, int(pop_per_var))
    NP = popsize * max(1, nvar)
    maxiter = max(10, math.ceil(budget_per_var / pop_per_var) - 1)
    nfev_est = NP * (maxiter + 1)
    return popsize, maxiter, nfev_est


def get_de_params(nvar: int):
    if DE_MODE.lower() == "manual":
        popsize = max(1, int(MANUAL_POPSIZE))
        maxiter = max(1, int(MANUAL_MAXITER))
        NP = popsize * max(1, nvar)
        nfev_est = NP * (maxiter + 1)
        return popsize, maxiter, nfev_est
    return _derive_de_params(nvar)


@dataclass
class OUQConfig:
    weights_path: str = WEIGHTS_PATH
    outdir_prefix: str = OUTDIR_PREFIX
    log_level: str = LOG_LEVEL
    log_to_file: bool = LOG_TO_FILE
    c_fail: float = C_FAIL
    K_list: List[int] = None
    R_list: List[int] = None
    nmc_est: int = N_MC_OPT
    nmc_refine: int = N_MC_FINAL
    rng_seed: int = RNG_SEED
    exact_total_atoms_threshold: int = EXACT_TOTAL_ATOMS_THRESHOLD
    canonical_eps: float = CANONICAL_EPS
    mass_conservation_tol: float = MASS_CONSERVATION_TOL
    use_gpu: bool = USE_GPU
    ann_batch_size: int = ANN_BATCH_SIZE
    de_strategy: str = DE_STRATEGY
    de_mutation: Tuple[float, float] = DE_MUTATION
    de_recombination: float = DE_RECOMBINATION
    de_tol: float = DE_TOL
    de_atol: float = DE_ATOL
    de_polish: bool = DE_POLISH

    def __post_init__(self):
        if self.K_list is None:
            self.K_list = K_LIST
        if self.R_list is None:
            self.R_list = R_LIST


CONFIG = OUQConfig()

# -------------------------- logging --------------------------
def _sanitize_msg(msg: str) -> str:
    return (msg.replace("✓", "OK")
               .replace("≈", "~=")
               .replace("→", "->")
               .replace("✗", "X")
               .replace("ℹ", "i"))


class SanitizeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _sanitize_msg(str(record.msg))
        return True


def setup_logging(config: OUQConfig, outdir: str) -> logging.Logger:
    logger = logging.getLogger("OUQ10D")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, config.log_level))
    filt = SanitizeFilter()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
    ch.addFilter(filt)
    logger.addHandler(ch)

    if config.log_to_file:
        os.makedirs(outdir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(outdir, "ouq.log"), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        fh.addFilter(filt)
        logger.addHandler(fh)
    return logger


logger = logging.getLogger("OUQ10D")

# -------------------------- model --------------------------
import torch
import torch.nn as nn


class RegressionNet(nn.Module):
    def __init__(self, input_size=MODEL_INPUT_SIZE, hidden_layers=MODEL_HIDDEN_LAYERS, output_size=MODEL_OUTPUT_SIZE):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.SELU()]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]), nn.SELU()]
        layers += [nn.Linear(hidden_layers[-1], output_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ForwardModel(nn.Module):
    def __init__(self, input_size=MODEL_INPUT_SIZE, hidden_layers=MODEL_HIDDEN_LAYERS, output_size=MODEL_OUTPUT_SIZE):
        super().__init__()
        self.net = RegressionNet(input_size, hidden_layers, output_size)
        self.register_buffer("x_min", torch.zeros(input_size))
        self.register_buffer("x_max", torch.ones(input_size))
        self.register_buffer("y_min", torch.zeros(output_size))
        self.register_buffer("y_max", torch.ones(output_size))

    @torch.no_grad()
    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        y_norm = self.net(x_norm)
        return y_norm * (self.y_max - self.y_min + 1e-8) + self.y_min


class DummyForward(ForwardModel):
    def __init__(self, input_size=MODEL_INPUT_SIZE):
        super().__init__(input_size=input_size)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def setup_bounds(self, a_lo, b_hi):
        with torch.no_grad():
            self.x_min[:] = torch.tensor(a_lo, dtype=torch.float32)
            self.x_max[:] = torch.tensor(b_hi, dtype=torch.float32)
            self.y_min[:] = 0.0
            self.y_max[:] = 2.0

    @torch.no_grad()
    def forward(self, x):
        x = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        coeff = torch.tensor([0.30, -0.18, 0.08, 0.04, -0.10, 0.07, 0.12, -0.05, 0.06, 0.09], device=x.device)
        y = 0.85 + 0.25 * (x * coeff).sum(dim=1, keepdim=True)
        return torch.clamp(y, 0.0, 2.0)


def load_forward_model(weights_path: str, device: str) -> ForwardModel:
    logger.info(f"Loading model: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    model = ForwardModel()
    
    try:
        sd = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    if torch.allclose(model.x_max, model.x_min):
        raise RuntimeError("x_min/x_max not loaded.")
    return model


def try_load_or_dummy(weights_path, device, a_lo, b_hi) -> ForwardModel:
    try:
        m = load_forward_model(weights_path, device)
        logger.info("OK Loaded real ANN")
        return m
    except Exception as e:
        logger.warning(f"Use dummy model: {e}")
        m = DummyForward(input_size=len(a_lo))
        m.setup_bounds(a_lo, b_hi)
        return m.to(device).eval()


@torch.no_grad()
def g_eval_ann(model: ForwardModel, X: np.ndarray, batch: int, device: str) -> np.ndarray:
    N = X.shape[0]
    ys = np.empty(N, dtype=np.float64)
    for i in range(0, N, batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32, copy=False)).to(device)
        ys[i:i + batch] = model(xb).cpu().numpy().reshape(-1)
    return ys


# -------------------------- canonical moments core --------------------------
def make_edges(a: float, b: float, K: int) -> np.ndarray:
    return np.linspace(a, b, K + 1)


def subinterval_moments_uniform(a_i: float, b_i: float, edges: np.ndarray, r: int) -> np.ndarray:
    K = len(edges) - 1
    M = np.zeros((K, r + 1))
    denom = (b_i - a_i)
    for j in range(K):
        L, U = edges[j], edges[j + 1]
        M[j, 0] = (U - L) / denom
        for q in range(1, r + 1):
            M[j, q] = (U ** (q + 1) - L ** (q + 1)) / ((q + 1) * denom)
    return M


def raw_to_unit_moments(Mj: np.ndarray, L: float, U: float) -> np.ndarray:
    r = len(Mj) - 1
    mass = Mj[0]
    W = U - L
    mprime = np.zeros(r + 1)
    mprime[0] = 1.0
    for q in range(1, r + 1):
        acc = 0.0
        for k in range(q + 1):
            acc += math.comb(q, k) * ((-L) ** (q - k)) * Mj[k]
        mprime[q] = acc / (mass * (W ** q))
    return mprime


def canonical_from_unit_moments(mprime: np.ndarray, eps: float) -> List[float]:
    r = len(mprime) - 1
    if r == 0:
        return []
    if r > 3:
        raise NotImplementedError("r>3 not implemented in core")
    m1 = mprime[1]
    p1 = float(np.clip(m1, eps, 1.0 - eps))
    if r == 1:
        return [p1]
    m2 = mprime[2]
    denom = m1 * (1.0 - m1)
    p2 = 0.5 if denom < eps else (m2 - m1 * m1) / denom
    p2 = float(np.clip(p2, eps, 1.0 - eps))
    if r == 2:
        return [p1, p2]
    m3 = mprime[3]
    m3_A = (m2 * m2) / m1 if m1 > eps else 0.0
    if m1 >= 1.0 - eps:
        m3_B = 1.0
    else:
        tB = (m1 - m2) / (1.0 - m1)
        tB = float(np.clip(tB, 0.0, 1.0))
        m3_B = 1.0 - (1.0 - m1) * (1.0 + tB + tB * tB)
    lo, hi = min(m3_A, m3_B), max(m3_A, m3_B)
    p3 = 0.5 if abs(hi - lo) < eps else (m3 - lo) / (hi - lo)
    p3 = float(np.clip(p3, eps, 1.0 - eps))
    return [p1, p2, p3]


def jacobi_from_canonical(p: List[float], m: int, eps: float) -> np.ndarray:
    need = 2 * m - 1
    P = [0.0] + list(p[:need])
    zeta = [0.0] * (2 * m)
    for k in range(1, 2 * m):
        zeta[k] = P[k] * (1.0 - P[k - 1])
    J = np.zeros((m, m), dtype=float)
    for i in range(m):
        J[i, i] = zeta[2 * i] + zeta[2 * i + 1]
        if i >= 1:
            b = max(0.0, zeta[2 * i - 1] * zeta[2 * i])
            J[i, i - 1] = J[i - 1, i] = math.sqrt(b)
    return J


def gauss_nodes_weights_from_J(J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = eigh(J)
    nodes = np.clip(eigvals, 0.0, 1.0)
    weights = eigvecs[0, :] ** 2
    return nodes, weights


def fixed_canonical_per_bin(M_row: np.ndarray, L: float, U: float, r: int, eps: float) -> List[float]:
    if r == 0:
        return []
    mprime = raw_to_unit_moments(M_row, L, U)
    return canonical_from_unit_moments(mprime, eps)


def build_bin_measure_from_p(edges: np.ndarray, j: int, p_fixed: List[float], p_free: List[float],
                             M_row: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    L, U = edges[j], edges[j + 1]
    r = len(p_fixed)
    m = r + 1
    need = 2 * m - 1
    p_all = [float(np.clip(v, eps, 1.0 - eps)) for v in (list(p_fixed) + list(p_free))]
    assert len(p_all) == need
    J = jacobi_from_canonical(p_all, m, eps)
    y, w_hat = gauss_nodes_weights_from_J(J)
    x = L + (U - L) * y
    t = M_row[0] * w_hat
    return x, t


def verify_mass_conservation(marginals: List[Tuple[np.ndarray, np.ndarray]], tol: float):
    for i, (xs, ts) in enumerate(marginals):
        if abs(np.sum(ts) - 1.0) > tol:
            raise ValueError(f"Dim {i + 1} mass != 1")


# -------------------------- sampling / PoF --------------------------
def _sample_from_discrete(xs: np.ndarray, ts: np.ndarray, U: np.ndarray) -> np.ndarray:
    w = ts / np.sum(ts)
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, U, side='right')
    idx = np.clip(idx, 0, len(xs) - 1)
    return xs[idx]


def sobol_crn(N: int, d: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(N)))
    return sampler.random_base2(m=m)[:N]


def _pof_from_marginals_ann_exact(model, marginals, c_fail, device, batch_size) -> float:
    d = len(marginals)
    sizes = [len(ts) for _, ts in marginals]
    grids = np.meshgrid(*[xs for xs, _ in marginals], indexing='ij')
    X = np.stack([g.ravel() for g in grids], axis=1)
    W = np.ones(X.shape[0], dtype=np.float64)
    for i, (_, ts) in enumerate(marginals):
        w_i = ts / np.sum(ts)
        shape = [sizes[j] if j == i else 1 for j in range(d)]
        W *= np.broadcast_to(w_i.reshape(shape), sizes).ravel()
    y = g_eval_ann(model, X, batch=batch_size, device=device)
    return float(W[y >= c_fail].sum())


def _pof_from_marginals_ann_mc(model, marginals, c_fail, device, batch_size, Ushared: np.ndarray) -> float:
    N, d = Ushared.shape
    X = np.empty((N, d), dtype=np.float64)
    for i, (xs, ts) in enumerate(marginals):
        X[:, i] = _sample_from_discrete(xs, ts, Ushared[:, i])
    y = g_eval_ann(model, X, batch=batch_size, device=device)
    return float((y >= c_fail).mean())


def pof_from_marginals_ann_adaptive(model, marginals, c_fail, device, batch_size,
                                    exact_threshold: int, Ushared: Optional[np.ndarray]):
    sizes = [len(ts) for _, ts in marginals]
    total_atoms = int(np.prod(sizes))
    if total_atoms <= exact_threshold:
        return _pof_from_marginals_ann_exact(model, marginals, c_fail, device, batch_size), 'exact'
    if Ushared is None:
        raise ValueError("Ushared required for MC evaluation.")
    return _pof_from_marginals_ann_mc(model, marginals, c_fail, device, batch_size, Ushared), 'mc'


def _iter_best_from_evals(evals: List[float], NP: int, maximize: bool) -> List[float]:
    if NP <= 0:
        return []
    best = -np.inf if maximize else np.inf
    out = []
    for k, pof in enumerate(evals, start=1):
        best = max(best, pof) if maximize else min(best, pof)
        if k % NP == 0:
            out.append(best)
    if len(evals) % NP != 0:
        out.append(best)
    return out


class DEOptimizer:
    def __init__(self, model: ForwardModel, K: int, r: int, D: int,
                 bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                 Ushared: np.ndarray, config: OUQConfig, device: str):
        self.model = model
        self.K = K
        self.r = r
        self.D = D
        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi
        self.Ushared = Ushared
        self.cfg = config
        self.device = device

        self.edges_list = []
        self.M_list = []
        self.p_fixed_list = []
        for i in range(D):
            edges = make_edges(bounds_lo[i], bounds_hi[i], K)
            M = subinterval_moments_uniform(bounds_lo[i], bounds_hi[i], edges, r)
            p_fix_dim = []
            for j in range(K):
                p_fix_dim.append(fixed_canonical_per_bin(M[j, :], edges[j], edges[j + 1], r, self.cfg.canonical_eps))
            self.edges_list.append(edges)
            self.M_list.append(M)
            self.p_fixed_list.append(p_fix_dim)
        self.nvar = self._compute_nvar()
        self.eval_count = 0
        self.eval_vals: List[float] = []
        self.NP = None
        self._last_logged_gen = 0

    def _compute_nvar(self) -> int:
        m = self.r + 1
        free_per_bin = (2 * m - 1) - self.r
        return self.D * self.K * free_per_bin

    def _decode(self, x: np.ndarray):
        m = self.r + 1
        need_total = 2 * m - 1
        fixed_len = self.r
        free_len = need_total - fixed_len
        idx = 0
        out = []
        try:
            for i in range(self.D):
                atoms_dim = []
                for j in range(self.K):
                    p_fixed = self.p_fixed_list[i][j]
                    p_free = x[idx:idx + free_len]
                    idx += free_len
                    xs, ts = build_bin_measure_from_p(self.edges_list[i], j, p_fixed, p_free,
                                                     self.M_list[i][j, :], self.cfg.canonical_eps)
                    atoms_dim.append((xs, ts))
                out.append(atoms_dim)
            return out
        except Exception as e:
            logger.debug(f"decode fail: {e}")
            return None

    def _flatten(self, marginals_by_dim):
        marginals = []
        for atoms_dim in marginals_by_dim:
            xs = np.concatenate([a[0] for a in atoms_dim])
            ts = np.concatenate([a[1] for a in atoms_dim])
            marginals.append((xs, ts))
        return marginals

    def _objective(self, x: np.ndarray, mode: str) -> float:
        self.eval_count += 1
        m_by_dim = self._decode(x)
        if m_by_dim is None:
            self.eval_vals.append(0.0 if mode == 'upper' else 1.0)
            return 1e10
        marginals = self._flatten(m_by_dim)
        try:
            verify_mass_conservation(marginals, self.cfg.mass_conservation_tol)
        except Exception:
            self.eval_vals.append(0.0 if mode == 'upper' else 1.0)
            return 1e10

        pof, method = pof_from_marginals_ann_adaptive(
            self.model, marginals, self.cfg.c_fail, self.device, self.cfg.ann_batch_size,
            self.cfg.exact_total_atoms_threshold, self.Ushared
        )
        self.eval_vals.append(pof)
        return -pof if mode == 'upper' else pof

    def optimize(self, mode: str, seed: int):
        popsize, maxiter, nfev_est = get_de_params(self.nvar)
        self.NP = popsize * self.nvar
        self._last_logged_gen = 0

        logger.info(f"DE (mode={mode}) nvar={self.nvar}, pop(mult)={popsize}, NP={self.NP}, iter={maxiter}, nfev_est~{nfev_est}")
        bounds = [(self.cfg.canonical_eps, 1.0 - self.cfg.canonical_eps)] * self.nvar
        self.eval_vals = []
        self.eval_count = 0
        maximize = (mode == 'upper')

        def _cb(xk, convergence):
            hist = _iter_best_from_evals(self.eval_vals, self.NP, maximize=maximize)
            gen = len(hist)
            if gen > self._last_logged_gen and gen >= 1:
                best = hist[-1]
                logger.info(f"    [{mode}] gen {gen:03d}: best={best:.6f}")
                self._last_logged_gen = gen
            return False

        res = differential_evolution(
            lambda x: self._objective(x, mode),
            bounds=bounds,
            strategy=self.cfg.de_strategy,
            maxiter=maxiter,
            popsize=popsize,
            tol=self.cfg.de_tol,
            atol=self.cfg.de_atol,
            mutation=self.cfg.de_mutation,
            recombination=self.cfg.de_recombination,
            seed=seed,
            workers=1,
            updating='deferred',
            disp=False,
            polish=self.cfg.de_polish,
            callback=_cb,
        )
        m_by_dim = self._decode(res.x)
        marginals = self._flatten(m_by_dim)
        value = -res.fun if mode == 'upper' else res.fun
        info = {
            "nit": int(res.nit), "nfev": int(res.nfev), "success": bool(res.success),
            "message": str(res.message), "best_x": res.x, "eval_count": self.eval_count,
            "popsize_mult": popsize, "maxiter": maxiter, "NP": self.NP,
        }
        logger.info(f"DE done: value={value:.6f}, nit={info['nit']}, nfev={info['nfev']}")
        history = _iter_best_from_evals(self.eval_vals, self.NP, maximize=maximize)
        return value, marginals, info, history


# -------------------------- save/plot --------------------------
def save_marginals_csv(marginals: List[Tuple[np.ndarray, np.ndarray]], outdir: str, K: int, r: int, mode: str):
    os.makedirs(outdir, exist_ok=True)
    for i, (xs, ts) in enumerate(marginals, start=1):
        df = pd.DataFrame({"x": xs.astype(float), "t": ts.astype(float)})
        path = os.path.join(outdir, f"atoms_K{K}_r{r}_{mode}_dim{i}_{NAMES[i-1]}.csv")
        df.to_csv(path, index=False)


def save_convergence_history(history_u: List[float], history_l: List[float], outdir: str, K: int, r: int, true_pof: Optional[float]):
    hist_dir = os.path.join(outdir, f"history_K{K}_r{r}")
    os.makedirs(hist_dir, exist_ok=True)
    if history_u:
        pd.DataFrame({"iter": np.arange(1, len(history_u) + 1), "upper_best_pof": history_u}).to_csv(
            os.path.join(hist_dir, "upper_history.csv"), index=False)
    if history_l:
        pd.DataFrame({"iter": np.arange(1, len(history_l) + 1), "lower_best_pof": history_l}).to_csv(
            os.path.join(hist_dir, "lower_history.csv"), index=False)
    plt.figure(figsize=(8, 5))
    if history_u:
        plt.plot(history_u, label='Upper best', linewidth=2)
    if history_l:
        plt.plot(history_l, label='Lower best', linewidth=2)
    if true_pof is not None:
        plt.axhline(true_pof, color='k', ls='--', alpha=0.7, label='MC ref')
    plt.xlabel("Generation")
    plt.ylabel("PoF")
    plt.title(f"DE Convergence (K={K}, r={r})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, "convergence.png"), dpi=150)
    plt.close()


def log_settings_table(K_list: List[int], R_list: List[int], d: int):
    lines = []
    lines.append("=" * 140)
    lines.append("DE + MC (two-stage) Settings")
    lines.append("=" * 140)
    hdr = f"{'K':>3} {'r':>3} | {'nvar':>6} {'atoms':>18} | {'pop':>5} {'maxit':>6} {'nfev_est':>10} | {'N_mc_opt':>9} {'N_mc_final':>11} {'method':>7}"
    lines.append(hdr)
    lines.append("-" * 140)
    for K in K_list:
        for r in R_list:
            nvar = d * K * (r + 1)
            pop, maxit, nfev_est = get_de_params(nvar)
            total_atoms = (K * (r + 1)) ** d
            method = "exact" if total_atoms <= CONFIG.exact_total_atoms_threshold else "mc"
            line = (f"{K:3d} {r:3d} | {nvar:6d} {total_atoms:18,d} | {pop:5d} {maxit:6d} {nfev_est:10d} | "
                    f"{CONFIG.nmc_est:9,d} {CONFIG.nmc_refine:11,d} {method:>7}")
            lines.append(line)
    lines.append("=" * 140)
    for ln in lines:
        logger.info(ln)


# -------------------------- top-level --------------------------
def true_pof_uniform_mc_ann(model: ForwardModel, N: int, seed: int, device: str) -> float:
    U = sobol_crn(N, D, seed)
    X = A_LO + (B_HI - A_LO) * U
    y = g_eval_ann(model, X, batch=CONFIG.ann_batch_size, device=device)
    return float((y >= CONFIG.c_fail).sum()) / N


def pilot_reference_outputs(model: ForwardModel, N: int, seed: int, device: str) -> np.ndarray:
    U = sobol_crn(N, D, seed)
    X = A_LO + (B_HI - A_LO) * U
    y = g_eval_ann(model, X, batch=CONFIG.ann_batch_size, device=device)
    return y


def build_cfail_scan_from_target_pofs(y_ref: np.ndarray, target_pofs: np.ndarray) -> pd.DataFrame:
    rows = []
    for p in target_pofs:
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"Invalid target PoF: {p}")
        c_fail = float(np.quantile(y_ref, 1.0 - float(p)))
        p_check = float(np.mean(y_ref >= c_fail))
        rows.append({
            "target_pof": float(p),
            "c_fail": c_fail,
            "ref_pof_from_pilot": p_check,
        })
    return pd.DataFrame(rows)


def log_single_case_settings(K: int, r: int, d: int):
    nvar = d * K * (r + 1)
    pop, maxit, nfev_est = get_de_params(nvar)
    total_atoms = (K * (r + 1)) ** d
    method = "exact" if total_atoms <= CONFIG.exact_total_atoms_threshold else "mc"
    logger.info("=" * 140)
    logger.info("Fixed-(K,r) DE + MC (two-stage) Settings")
    logger.info("=" * 140)
    logger.info(f"K={K}, r={r}, D={d}, nvar={nvar}, atoms={total_atoms:,}, pop={pop}, maxit={maxit}, "
                f"nfev_est~{nfev_est}, N_mc_opt={CONFIG.nmc_est:,}, N_mc_final={CONFIG.nmc_refine:,}, method={method}")
    logger.info("=" * 140)


def run_single_ouq_case(model: ForwardModel, device: str, outdir: str, p_true: Optional[float], K: int, r: int):
    logger.info("=" * 80)
    logger.info(f"Fixed scan case: K={K}, r={r}")
    log_single_case_settings(K, r, D)

    Ushared_opt = sobol_crn(CONFIG.nmc_est, D, CONFIG.rng_seed)
    opt = DEOptimizer(model, K, r, D, A_LO, B_HI, Ushared_opt, CONFIG, device)

    ub_val_opt, ub_m, ub_info, ub_hist = opt.optimize(mode='upper', seed=CONFIG.rng_seed)
    lb_val_opt, lb_m, lb_info, lb_hist = opt.optimize(mode='lower', seed=CONFIG.rng_seed + 1)

    Ushared_final = sobol_crn(CONFIG.nmc_refine, D, CONFIG.rng_seed + 12345)
    ub_val_final, method_u = pof_from_marginals_ann_adaptive(
        model, ub_m, CONFIG.c_fail, device, CONFIG.ann_batch_size,
        CONFIG.exact_total_atoms_threshold, Ushared_final)
    lb_val_final, method_l = pof_from_marginals_ann_adaptive(
        model, lb_m, CONFIG.c_fail, device, CONFIG.ann_batch_size,
        CONFIG.exact_total_atoms_threshold, Ushared_final)

    logger.info(f"  -> upper(opt)={ub_val_opt:.6f}, lower(opt)={lb_val_opt:.6f}")
    logger.info(f"  -> upper(final)={ub_val_final:.6f} [{method_u}], lower(final)={lb_val_final:.6f} [{method_l}], width={ub_val_final - lb_val_final:.6f}")

    save_marginals_csv(ub_m, outdir, K, r, 'upper')
    save_marginals_csv(lb_m, outdir, K, r, 'lower')
    save_convergence_history(ub_hist, lb_hist, outdir, K, r, p_true)

    result = {
        "K": K,
        "r": r,
        "upper_opt": ub_val_opt,
        "lower_opt": lb_val_opt,
        "upper": ub_val_final,
        "lower": lb_val_final,
        "width": ub_val_final - lb_val_final,
        "nvar": opt.nvar,
        "nit_upper": ub_info["nit"],
        "nfev_upper": ub_info["nfev"],
        "nit_lower": lb_info["nit"],
        "nfev_lower": lb_info["nfev"],
        "popsize": ub_info["popsize_mult"],
        "maxiter": ub_info["maxiter"],
        "NP": ub_info["NP"],
        "method_upper": method_u,
        "method_lower": method_l,
    }
    return result


def run_threshold_scan(model: ForwardModel, device: str, outdir: str, K: int, r: int) -> pd.DataFrame:
    logger.info("=== Pilot reference output scan ===")
    y_ref = pilot_reference_outputs(model, N_POF_SCAN, seed=REF_POF_MC_SEED, device=device)
    logger.info(f"Pilot Y range: min={y_ref.min():.6f}, max={y_ref.max():.6f}")

    scan_df = build_cfail_scan_from_target_pofs(y_ref, TARGET_POFS)
    target_path = os.path.join(outdir, "target_pof_to_cfail.csv")
    scan_df.to_csv(target_path, index=False)
    logger.info(f"Saved {target_path}")

    all_rows = []
    for i, row in scan_df.iterrows():
        target_pof = float(row["target_pof"])
        c_fail = float(row["c_fail"])
        p_ref_pilot = float(row["ref_pof_from_pilot"])

        logger.info("-" * 100)
        logger.info(f"Scan point {i + 1}/{len(scan_df)}: target_pof={target_pof:.6e}, c_fail={c_fail:.6f}")
        CONFIG.c_fail = c_fail

        if RECOMPUTE_REF_POF_FINAL:
            p_true = true_pof_uniform_mc_ann(model, CONFIG.nmc_refine, seed=REF_POF_MC_SEED, device=device)
            logger.info(f"Reference MC PoF ~= {p_true:.6e}")
        else:
            p_true = p_ref_pilot
            logger.info(f"Reference PoF from pilot ~= {p_true:.6e}")

        case_dir = os.path.join(outdir, f"scan_{i + 1:02d}_yc_{c_fail:.6f}")
        os.makedirs(case_dir, exist_ok=True)

        result = run_single_ouq_case(model, device, case_dir, p_true, K, r)
        result.update({
            "target_pof": target_pof,
            "c_fail": c_fail,
            "ref_pof_pilot": p_ref_pilot,
            "ref_pof_final_mc": p_true,
        })
        all_rows.append(result)

    df = pd.DataFrame(all_rows)
    scan_out = os.path.join(outdir, "ouq_vs_pof_scan.csv")
    df.to_csv(scan_out, index=False)
    logger.info(f"Saved {scan_out}")
    return df


def plot_bounds_vs_pof(scan_df: pd.DataFrame, outdir: str):
    df = scan_df.sort_values("ref_pof_final_mc", ascending=True).copy()

    x = df["ref_pof_final_mc"].to_numpy(dtype=float)
    ub = df["upper"].to_numpy(dtype=float)
    lb = df["lower"].to_numpy(dtype=float)

    plt.figure(figsize=(8.8, 5.8))
    plt.plot(x, ub, "-o", linewidth=2, label="Upper bound")
    plt.plot(x, lb, "--o", linewidth=2, label="Lower bound")
    plt.fill_between(x, ub, lb, alpha=0.15)

    plt.xscale("log")
    plt.xlabel("Reference PoF")
    plt.ylabel("OUQ bound")
    plt.title(f"OUQ bounds vs PoF (K={FIXED_K}, r={FIXED_R})")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "bounds_vs_pof.png")
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved {path}")


def plot_bounds_vs_cfail(scan_df: pd.DataFrame, outdir: str):
    df = scan_df.sort_values("c_fail", ascending=True).copy()

    x = df["c_fail"].to_numpy(dtype=float)
    ub = df["upper"].to_numpy(dtype=float)
    lb = df["lower"].to_numpy(dtype=float)

    plt.figure(figsize=(8.8, 5.8))
    plt.plot(x, ub, "-o", linewidth=2, label="Upper bound")
    plt.plot(x, lb, "--o", linewidth=2, label="Lower bound")
    plt.fill_between(x, ub, lb, alpha=0.15)
    plt.xlabel(r"$Y_c$")
    plt.ylabel("OUQ bound")
    plt.title(f"OUQ bounds vs threshold (K={FIXED_K}, r={FIXED_R})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "bounds_vs_cfail.png")
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved {path}")


def save_scan_summary_txt(scan_df: pd.DataFrame, outdir: str):
    path = os.path.join(outdir, "scan_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Fixed-(K,r) OUQ threshold scan summary\n")
        f.write(f"K = {FIXED_K}, r = {FIXED_R}\n")
        f.write(f"Target PoFs = {TARGET_POFS.tolist()}\n\n")
        cols = ["target_pof", "c_fail", "ref_pof_final_mc", "upper", "lower", "width",
                "method_upper", "method_lower"]
        f.write(scan_df[cols].to_string(index=False))
        f.write("\n")
    logger.info(f"Saved {path}")



def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run Case 5 fixed-(K,r) threshold / target-PoF scan.")
    parser.add_argument("--fixed-K", type=int, default=FIXED_K)
    parser.add_argument("--fixed-r", type=int, default=FIXED_R)
    parser.add_argument("--target-pofs", nargs="+", type=float, default=TARGET_POFS.tolist())
    parser.add_argument("--weights-path", default=CONFIG.weights_path)
    parser.add_argument("--outdir-prefix", default=CONFIG.outdir_prefix)
    parser.add_argument("--n-pof-scan", type=int, default=N_POF_SCAN)
    parser.add_argument("--n-mc-opt", type=int, default=CONFIG.nmc_est)
    parser.add_argument("--n-mc-final", type=int, default=CONFIG.nmc_refine)
    parser.add_argument("--popsize", type=int, default=MANUAL_POPSIZE)
    parser.add_argument("--maxiter", type=int, default=MANUAL_MAXITER)
    parser.add_argument("--exact-threshold", type=int, default=CONFIG.exact_total_atoms_threshold)
    parser.add_argument("--use-gpu", action="store_true", default=CONFIG.use_gpu)
    return parser.parse_args()

def main():
    args = parse_cli_args()
    global FIXED_K, FIXED_R, TARGET_POFS, N_POF_SCAN, MANUAL_POPSIZE, MANUAL_MAXITER
    FIXED_K = int(args.fixed_K)
    FIXED_R = int(args.fixed_r)
    TARGET_POFS = np.asarray(args.target_pofs, dtype=float)
    N_POF_SCAN = int(args.n_pof_scan)
    MANUAL_POPSIZE = int(args.popsize)
    MANUAL_MAXITER = int(args.maxiter)
    CONFIG.weights_path = str(args.weights_path)
    CONFIG.outdir_prefix = str(args.outdir_prefix)
    CONFIG.nmc_est = int(args.n_mc_opt)
    CONFIG.nmc_refine = int(args.n_mc_final)
    CONFIG.exact_total_atoms_threshold = int(args.exact_threshold)
    CONFIG.use_gpu = bool(args.use_gpu)

    np.random.seed(CONFIG.rng_seed)
    random.seed(CONFIG.rng_seed)
    torch.manual_seed(CONFIG.rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.rng_seed)

    stamp = np.datetime_as_string(np.datetime64('now'), unit='s').replace(':', '-')
    outdir = f"{CONFIG.outdir_prefix}_K{FIXED_K}_r{FIXED_R}_{stamp}"
    os.makedirs(outdir, exist_ok=True)

    global logger
    logger = setup_logging(CONFIG, outdir)
    logger.info("=== OUQ 10D fixed-(K,r) threshold/PoF scan ===")
    logger.info(f"Platform: {platform.system()} {platform.release()}  Python: {sys.version.split()[0]}  Torch: {torch.__version__}")
    logger.info(f"DE_MODE={DE_MODE} (manual popsize={MANUAL_POPSIZE}, maxiter={MANUAL_MAXITER})" if DE_MODE == 'manual'
                else f"DE_MODE={DE_MODE} (budget_per_var={BUDGET_PER_VAR}, pop_per_var={POP_PER_VAR})")
    logger.info(f"exact_total_atoms_threshold = {CONFIG.exact_total_atoms_threshold:,}")
    logger.info(f"Names = {NAMES}")
    logger.info(f"Means = {MU}")
    logger.info(f"Lower bounds = {A_LO}")
    logger.info(f"Upper bounds = {B_HI}")
    logger.info(f"Fixed scan: K={FIXED_K}, r={FIXED_R}")
    logger.info(f"Target PoFs = {TARGET_POFS}")

    device = 'cuda' if CONFIG.use_gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info("workers fixed to 1 (single process)")
        CONFIG.ann_batch_size = min(CONFIG.ann_batch_size, 131072)
    logger.info(f"Device: {device}")

    model = try_load_or_dummy(CONFIG.weights_path, device, A_LO, B_HI)
    df_scan = run_threshold_scan(model, device, outdir, FIXED_K, FIXED_R)

    plot_bounds_vs_pof(df_scan, outdir)
    plot_bounds_vs_cfail(df_scan, outdir)
    save_scan_summary_txt(df_scan, outdir)

    meta = {
        "version": "10d-fixedKr-threshold-scan",
        "names": NAMES,
        "means": MU.tolist(),
        "lower_bounds": A_LO.tolist(),
        "upper_bounds": B_HI.tolist(),
        "fixed_K": FIXED_K,
        "fixed_r": FIXED_R,
        "target_pofs": TARGET_POFS.tolist(),
        "n_pof_scan": N_POF_SCAN,
        "c_fail_last": CONFIG.c_fail,
        "nmc_est": CONFIG.nmc_est,
        "nmc_refine": CONFIG.nmc_refine,
        "rng_seed": CONFIG.rng_seed,
        "ref_pof_mc_seed": REF_POF_MC_SEED,
        "de_mode": DE_MODE,
        "model_input_size": MODEL_INPUT_SIZE,
        "model_hidden_layers": MODEL_HIDDEN_LAYERS,
        "model_output_size": MODEL_OUTPUT_SIZE,
        "exact_total_atoms_threshold": CONFIG.exact_total_atoms_threshold,
    }
    with open(os.path.join(outdir, "config_scan.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Artifacts:")
    logger.info("  - target_pof_to_cfail.csv")
    logger.info("  - ouq_vs_pof_scan.csv")
    logger.info("  - bounds_vs_pof.png")
    logger.info("  - bounds_vs_cfail.png")
    logger.info("  - scan_summary.txt")
    logger.info("  - config_scan.json")
    logger.info("  - scan_*/ (history + atom CSVs)")
    logger.info("Done.")



if __name__ == "__main__":
    main()
