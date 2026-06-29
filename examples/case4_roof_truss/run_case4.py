#!/usr/bin/env python3
"""Run Case 4: eight-dimensional rare-event roof-truss OUQ example."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm as std_norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subdomain_ouq import MarginalSpec, bounded_lognormal, bounded_normal, DEControls
from subdomain_ouq.optimizer import AnisotropicCase, prepare_anisotropic_case, solve_anisotropic_exact
from subdomain_ouq.io import ensure_dir


@dataclass
class RoofTrussProblem:
    eta: float = 1.25e5
    positive_model: str = "lognormal"
    trunc_q: float = 1e-6
    c_fail: float = 0.0
    failure: str = "le"

    def __post_init__(self) -> None:
        maker = bounded_lognormal if self.positive_model == "lognormal" else bounded_normal
        inputs = [
            ("q", 2.0e4, 0.07),
            ("l", 12.0, 0.01),
            ("As", 9.82e-4, 0.06),
            ("Ac", 4.0e-2, 0.12),
            ("Es", 2.0e11, 0.06),
            ("Ec", 3.0e11, 0.06),
            ("fs", 3.35e8, 0.12),
            ("fc", 1.34e7, 0.18),
        ]
        self.marginals = []
        for name, mean, cov in inputs:
            dist = maker(mean, cov, self.trunc_q, name=name)
            self.marginals.append(MarginalSpec(name, dist.a, dist.b, dist))
        self.names = [m.name for m in self.marginals]
        self.dists = [m.dist for m in self.marginals]
        self.bounds = [(m.a, m.b) for m in self.marginals]
        self.d = len(self.marginals)

    @staticmethod
    def g_components(X: np.ndarray):
        q, l, As, Ac, Es, Ec, fs, fc = np.asarray(X, dtype=float).T
        g1 = 0.03 - (q * l * l / 2.0) * (3.81 / (Ac * Ec) + 1.13 / (As * Es))
        g2 = fc * Ac - 1.185 * q * l
        g3 = fs * As - 0.75 * q * l
        return g1, g2, g3

    def g_eval(self, X: np.ndarray) -> np.ndarray:
        g1, g2, g3 = self.g_components(X)
        return np.minimum.reduce([g1, g2, g3]) + float(self.eta)


def u_to_x(problem: RoofTrussProblem, U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(1, -1)
    P = np.clip(std_norm.cdf(U), 1e-300, 1.0 - 1e-16)
    return np.column_stack([dist.ppf(P[:, i]) for i, dist in enumerate(problem.dists)])


def g_system_u(problem: RoofTrussProblem, U: np.ndarray) -> np.ndarray:
    return problem.g_eval(u_to_x(problem, U))


def subset_simulation_reference(
    problem: RoofTrussProblem,
    N: int = 100000,
    p0: float = 0.1,
    sigma_prop: float = 0.9,
    max_levels: int = 20,
    seed: int = 7,
    verbose: bool = True,
) -> dict:
    n_seeds = int(round(N * p0))
    if n_seeds <= 0 or N % n_seeds != 0:
        raise ValueError("Choose N so N*p0 is a positive divisor of N.")
    chain_len = N // n_seeds
    rng = np.random.default_rng(seed)
    d = problem.d
    U = rng.normal(size=(N, d))
    gvals = g_system_u(problem, U)
    thresholds = []
    accepts = []

    def g_scalar(u: np.ndarray) -> float:
        return float(g_system_u(problem, np.asarray(u).reshape(1, -1))[0])

    for level in range(max_levels):
        order = np.argsort(gvals)
        U = U[order]
        gvals = gvals[order]
        b = float(gvals[n_seeds - 1])
        thresholds.append(b)
        if verbose:
            print(f"[SuS level {level:02d}] threshold={b:.8e}, frac_fail={np.mean(gvals <= 0.0):.6f}", flush=True)
        if b <= 0.0:
            p_last = float(np.mean(gvals <= 0.0))
            return {"p_hat": (p0 ** level) * p_last, "levels": level + 1, "N_per_level": N, "p0": p0, "sigma_prop": sigma_prop, "thresholds": thresholds, "mean_acceptance": float(np.mean(accepts)) if accepts else None}
        seeds = U[:n_seeds].copy()
        g_seeds = gvals[:n_seeds].copy()
        U_next = np.zeros_like(U)
        g_next = np.zeros_like(gvals)
        flags = []
        idx = 0
        for j in range(n_seeds):
            u_cur = seeds[j].copy()
            g_cur = float(g_seeds[j])
            U_next[idx] = u_cur
            g_next[idx] = g_cur
            idx += 1
            for _ in range(chain_len - 1):
                u_prop = u_cur + sigma_prop * rng.normal(size=d)
                g_prop = g_scalar(u_prop)
                accepted = False
                if g_prop <= b:
                    log_alpha = -0.5 * (np.dot(u_prop, u_prop) - np.dot(u_cur, u_cur))
                    if np.log(rng.random()) < min(0.0, log_alpha):
                        u_cur = u_prop
                        g_cur = g_prop
                        accepted = True
                flags.append(accepted)
                U_next[idx] = u_cur
                g_next[idx] = g_cur
                idx += 1
        U, gvals = U_next, g_next
        accepts.append(float(np.mean(flags)) if flags else 1.0)
        if verbose:
            print(f"[SuS level {level:02d}] acceptance={accepts[-1]:.6f}", flush=True)
    return {"p_hat": (p0 ** max_levels) * float(np.mean(gvals <= 0.0)), "levels": max_levels, "N_per_level": N, "p0": p0, "sigma_prop": sigma_prop, "thresholds": thresholds, "mean_acceptance": float(np.mean(accepts)) if accepts else None, "warning": "max_levels reached"}


def mcdiarmid_subdiameters(
    names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    F: Callable[[np.ndarray], float],
    exclude_names: Sequence[str] = ("q", "l"),
    seed: int = 7,
    maxiter: int = 200,
    popsize: int = 50,
    verbose: bool = True,
) -> list[dict]:
    rows = []
    exclude = set(exclude_names)
    d = len(names)
    for i, name in enumerate(names):
        if name in exclude:
            continue
        other_idx = [j for j in range(d) if j != i]
        other_bounds = [bounds[j] for j in other_idx]

        def diameter(z):
            z = np.asarray(z, dtype=float)
            x_lo = np.zeros(d)
            x_hi = np.zeros(d)
            k = 0
            for j in range(d):
                if j == i:
                    x_lo[j] = bounds[j][0]
                    x_hi[j] = bounds[j][1]
                else:
                    x_lo[j] = z[k]
                    x_hi[j] = z[k]
                    k += 1
            return abs(F(x_hi) - F(x_lo))

        if verbose:
            print(f"[screen] {name}", flush=True)
        res = differential_evolution(lambda z: -diameter(z), other_bounds, seed=seed, maxiter=maxiter, popsize=popsize, polish=True, workers=1, updating="deferred")
        rows.append({"index": i, "name": name, "D_i": float(-res.fun), "optimizer_success": bool(res.success), "optimizer_nit": int(res.nit), "optimizer_nfev": int(res.nfev)})
    total = sum(r["D_i"] for r in rows)
    for r in rows:
        r["share"] = float(r["D_i"] / total) if total > 0 else 0.0
    rows.sort(key=lambda r: r["D_i"], reverse=True)
    return rows


def build_case_catalog(names: Sequence[str], active_names: Sequence[str], K_list: Sequence[int], r_list: Sequence[int]) -> list[AnisotropicCase]:
    name_to_idx = {name: i for i, name in enumerate(names)}
    cases = []
    for K in K_list:
        for r in r_list:
            K_vec = [1] * len(names)
            r_vec = [0] * len(names)
            for name in active_names:
                idx = name_to_idx[name]
                K_vec[idx] = int(K)
                r_vec[idx] = int(r)
            cases.append(AnisotropicCase(name=f"active_K{K}_r{r}", K_vec=K_vec, r_vec=r_vec))
    return cases


def run_case4(args) -> None:
    outdir = ensure_dir(args.output_dir)
    prob = RoofTrussProblem(eta=args.eta, positive_model=args.positive_model)
    pd.DataFrame([{"name": m.name, "lower": m.a, "upper": m.b} for m in prob.marginals]).to_csv(outdir / "input_bounds.csv", index=False)

    sus = None
    if not args.skip_reference:
        sus = subset_simulation_reference(prob, N=args.sus_N, p0=args.sus_p0, sigma_prop=args.sus_sigma_prop, max_levels=args.sus_max_levels, seed=args.seed, verbose=not args.quiet)
        (outdir / "reference_subset_simulation.json").write_text(json.dumps(sus, indent=2), encoding="utf-8")
        print(f"Subset Simulation PoF = {sus['p_hat']:.8e}")

    screening = None
    if not args.skip_screen:
        F = lambda x: float(prob.g_eval(np.asarray(x, dtype=float).reshape(1, -1))[0])
        exclude = [s.strip() for s in args.exclude_screen_names.split(",") if s.strip()]
        screening = mcdiarmid_subdiameters(prob.names, prob.bounds, F, exclude_names=exclude, seed=args.seed, maxiter=args.screen_maxiter, popsize=args.screen_popsize, verbose=not args.quiet)
        pd.DataFrame(screening).to_csv(outdir / "mcdiarmid_screening.csv", index=False)

    active_names = [s.strip() for s in args.active_names.split(",") if s.strip()]
    cases = build_case_catalog(prob.names, active_names, args.K_list, args.r_list)
    de = DEControls(popsize=args.popsize, maxiter=args.maxiter, polish=False, seed_upper=args.seed, seed_lower=args.seed + 17)
    rows = []
    for case in cases:
        if case.total_atoms > args.max_exact_atoms:
            print(f"[skip] {case.name}: total_atoms={case.total_atoms} > max_exact_atoms={args.max_exact_atoms}")
            continue
        print(f"\n=== {case.name}: atoms={case.total_atoms}, nvar={case.nvar} ===")
        prep = prepare_anisotropic_case(prob, case, canonical_eps=args.canonical_eps)
        res = solve_anisotropic_exact(prob, prep, de=de, canonical_eps=args.canonical_eps, verbose=not args.quiet)
        K_active = max(case.K_vec)
        r_active = max(case.r_vec)
        rows.append({k: v for k, v in res.items() if k not in {"marginals_upper", "marginals_lower", "theta_upper", "theta_lower", "upper_history", "lower_history"}} | {"K": K_active, "r": r_active})
        pd.DataFrame(rows).to_csv(outdir / "ouq_bounds_summary.csv", index=False)
    meta = {"eta": args.eta, "positive_model": args.positive_model, "active_names": active_names, "reference_pof": None if sus is None else sus["p_hat"], "screening": screening}
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Saved Case 4 output to {outdir}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run roof-truss rare-event OUQ.")
    parser.add_argument("--eta", type=float, default=1.25e5)
    parser.add_argument("--positive-model", choices=["lognormal", "normal"], default="lognormal")
    parser.add_argument("--active-names", type=str, default="q,l,Ac,fc")
    parser.add_argument("--exclude-screen-names", type=str, default="q,l")
    parser.add_argument("--K-list", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--r-list", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "case4_roof_truss"))
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--max-exact-atoms", type=int, default=500_000)
    parser.add_argument("--canonical-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--screen-maxiter", type=int, default=200)
    parser.add_argument("--screen-popsize", type=int, default=50)
    parser.add_argument("--sus-N", type=int, default=100000)
    parser.add_argument("--sus-p0", type=float, default=0.1)
    parser.add_argument("--sus-sigma-prop", type=float, default=0.9)
    parser.add_argument("--sus-max-levels", type=int, default=20)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-screen", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    run_case4(args)


if __name__ == "__main__":
    main()
