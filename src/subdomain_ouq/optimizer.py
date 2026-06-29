"""Differential-Evolution wrappers for canonical-moment OUQ.

The functions in this module are intentionally thin wrappers around SciPy's
``differential_evolution``.  They keep the examples readable while preserving the
core OUQ workflow used in the original scripts: decode free canonical moments,
recover Dirac atoms with the Jacobi matrix, evaluate PoF exactly or by ITS/CRN,
and optimize upper/lower bounds.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .io import save_marginals_csv, save_ouq_bounds_summary
from .measures import (
    OUQCase,
    build_marginal_with_canonical_moments,
    build_marginals_from_theta_anisotropic,
    prepare_anisotropic_case as _prepare_anisotropic_case,
)
from .moments import make_edges, subinterval_moments
from .pof import pof_from_marginals_adaptive, pof_from_marginals_exact


@dataclass
class DEControls:
    popsize: int = 50
    maxiter: int = 200
    mutation: tuple = (0.5, 1.0)
    recombination: float = 0.7
    tol: float = 1e-6
    atol: float = 0.0
    polish: bool = True
    seed_upper: int = 123
    seed_lower: int = 124
    workers: int = 1
    strategy: str = "best1bin"


# Backward-compatible alias for earlier drafts of this refactor.
DEConfig = DEControls


@dataclass
class OUQRunControls:
    N_mc_opt: int = 50_000
    N_mc_final: int = 500_000
    exact_threshold_opt: int = 250_000
    exact_threshold_final: int = 250_000
    canonical_eps: float = 1e-9
    use_crn: bool = True
    crn_method: str = "random"
    record_history: bool = True


@dataclass
class SweepResult:
    dataframe: pd.DataFrame
    output_dir: Path
    rows: list[dict]


# Case 4 uses anisotropic refinement: only active variables receive larger K,r.
AnisotropicCase = OUQCase


def derive_de_params(nvar: int, budget_per_var: int = 1000, pop_per_var: int = 20):
    popsize = max(5, int(pop_per_var))
    NP = popsize * max(1, nvar)
    maxiter = max(10, math.ceil(budget_per_var / pop_per_var) - 1)
    return popsize, maxiter, NP * (maxiter + 1)


def iter_best_from_evals(evals: List[float], NP: int, maximize: bool) -> List[float]:
    if NP <= 0:
        return []
    best = -np.inf if maximize else np.inf
    out = []
    for k, v in enumerate(evals, start=1):
        pof = -v if maximize else v
        best = max(best, pof) if maximize else min(best, pof)
        if k % NP == 0:
            out.append(float(best))
    if len(evals) % NP != 0:
        out.append(float(best))
    return out


class DEObjective:
    def __init__(self, ctx, maximize: bool = True):
        self.ctx = ctx
        self.maximize = maximize
        self.eval_count = 0
        self.method_used = None
        self.eval_vals = []

    def __call__(self, theta):
        val = objective_function_core(theta, self.ctx, self.maximize, self)
        self.eval_count += 1
        self.eval_vals.append(float(val))
        return val


def _side_from_problem(prob, explicit: Optional[str] = None) -> str:
    if explicit in {"ge", "le"}:
        return explicit
    failure = getattr(prob, "failure", "ge")
    return "le" if failure == "le" else "ge"


def objective_function_core(theta, ctx, maximize: bool, obj_instance: DEObjective):
    prob = ctx["prob"]
    edges_list = ctx["edges_list"]
    moments = ctx["moments"]
    r = ctx["r"]
    d = ctx["d"]
    K = ctx["K"]
    eps = ctx.get("eps", 1e-9)
    theta = np.asarray(theta, dtype=float).reshape(d, K, r + 1)
    marginals = []
    for i in range(d):
        try:
            xs, ts, _ = build_marginal_with_canonical_moments(
                edges_list[i], moments[i], r, theta[i], check_residuals=False, eps=eps
            )
        except Exception:
            return 1e6
        marginals.append((xs, ts))
    try:
        pof, method = pof_from_marginals_adaptive(
            marginals,
            prob.g_eval,
            getattr(prob, "c_fail", 0.0),
            N_mc=ctx["N_mc"],
            U_shared=ctx["U_shared"],
            exact_threshold=ctx["exact_threshold"],
            side=ctx.get("side", "ge"),
        )
        if obj_instance.method_used is None:
            obj_instance.method_used = method
    except Exception:
        return 1e6
    return -pof if maximize else pof


def precompute_edges_moments(prob, K: int, r: int):
    edges_list, moments = [], []
    for i in range(prob.d):
        m = prob.marginals[i]
        edges = make_edges(m.a, m.b, K)
        M = subinterval_moments(prob.dists[i], edges, r)
        edges_list.append(edges)
        moments.append(M)
    return edges_list, moments


def _make_crn(N: int, d: int, seed: int = 42, method: str = "random"):
    # The original scripts use fixed random uniforms as Common Random Numbers.
    # Keeping this default avoids changing the optimization objective between
    # candidate measures while remaining deterministic across runs.
    rng = np.random.RandomState(seed)
    return rng.random((int(N), int(d)))


def run_isotropic_ouq_de(
    prob,
    K: int,
    r: int,
    de: Optional[DEControls] = None,
    run: Optional[OUQRunControls] = None,
    N_mc_opt: Optional[int] = None,
    N_mc_final: Optional[int] = None,
    exact_threshold_opt: Optional[int] = None,
    exact_threshold_final: Optional[int] = None,
    use_crn: Optional[bool] = None,
    side: str = "ge",
    eps: Optional[float] = None,
    record_history: Optional[bool] = None,
    verbose: bool = True,
):
    if de is None:
        de = DEControls()
    if run is None:
        run = OUQRunControls()
    if N_mc_opt is not None:
        run.N_mc_opt = int(N_mc_opt)
    if N_mc_final is not None:
        run.N_mc_final = int(N_mc_final)
    if exact_threshold_opt is not None:
        run.exact_threshold_opt = int(exact_threshold_opt)
    if exact_threshold_final is not None:
        run.exact_threshold_final = int(exact_threshold_final)
    if use_crn is not None:
        run.use_crn = bool(use_crn)
    if eps is not None:
        run.canonical_eps = float(eps)
    if record_history is not None:
        run.record_history = bool(record_history)

    d = prob.d
    nvar = d * K * (r + 1)
    NP = de.popsize * max(1, nvar)
    edges_list, moments = precompute_edges_moments(prob, K, r)
    if run.use_crn:
        U_all = _make_crn(max(run.N_mc_opt, run.N_mc_final), d, seed=42, method=run.crn_method)
        U_opt = U_all[: run.N_mc_opt, :]
        U_final = U_all[: run.N_mc_final, :]
    else:
        U_opt = None
        U_final = None
    ctx = {
        "prob": prob,
        "edges_list": edges_list,
        "moments": moments,
        "r": r,
        "d": d,
        "K": K,
        "N_mc": run.N_mc_opt,
        "U_shared": U_opt,
        "exact_threshold": run.exact_threshold_opt,
        "side": side,
        "eps": run.canonical_eps,
    }
    bounds = [(run.canonical_eps, 1.0 - run.canonical_eps)] * nvar
    obj_u = DEObjective(ctx, maximize=True)
    res_u = differential_evolution(
        obj_u,
        bounds,
        strategy=de.strategy,
        popsize=de.popsize,
        mutation=de.mutation,
        recombination=de.recombination,
        maxiter=de.maxiter,
        tol=de.tol,
        atol=de.atol,
        seed=de.seed_upper,
        polish=de.polish,
        disp=verbose,
        workers=de.workers,
        updating="deferred" if de.workers != 1 else "immediate",
    )
    obj_l = DEObjective(ctx, maximize=False)
    res_l = differential_evolution(
        obj_l,
        bounds,
        strategy=de.strategy,
        popsize=de.popsize,
        mutation=de.mutation,
        recombination=de.recombination,
        maxiter=de.maxiter,
        tol=de.tol,
        atol=de.atol,
        seed=de.seed_lower,
        polish=de.polish,
        disp=verbose,
        workers=de.workers,
        updating="deferred" if de.workers != 1 else "immediate",
    )

    def final_eval(theta_flat):
        theta = np.asarray(theta_flat, dtype=float).reshape(d, K, r + 1)
        marginals = []
        for i in range(d):
            xs, ts, _ = build_marginal_with_canonical_moments(
                edges_list[i], moments[i], r, theta[i], check_residuals=False, eps=run.canonical_eps
            )
            marginals.append((xs, ts))
        return pof_from_marginals_adaptive(
            marginals,
            prob.g_eval,
            getattr(prob, "c_fail", 0.0),
            run.N_mc_final,
            U_final,
            run.exact_threshold_final,
            side=side,
        )

    upper_final, method_u_final = final_eval(res_u.x)
    lower_final, method_l_final = final_eval(res_l.x)
    theta_u = res_u.x.reshape(d, K, r + 1)
    theta_l = res_l.x.reshape(d, K, r + 1)
    marg_u, marg_l = [], []
    resids_u, resids_l = [], []
    for i in range(d):
        xs, ts, ru = build_marginal_with_canonical_moments(
            edges_list[i], moments[i], r, theta_u[i], check_residuals=True, eps=run.canonical_eps
        )
        marg_u.append((xs, ts))
        resids_u.extend([(i, j, rr) for j, rr in ru])
        xs, ts, rl = build_marginal_with_canonical_moments(
            edges_list[i], moments[i], r, theta_l[i], check_residuals=True, eps=run.canonical_eps
        )
        marg_l.append((xs, ts))
        resids_l.extend([(i, j, rr) for j, rr in rl])
    return {
        "K": K,
        "r": r,
        "upper": -float(res_u.fun),
        "lower": float(res_l.fun),
        "upper_final": float(upper_final),
        "lower_final": float(lower_final),
        "method_upper": obj_u.method_used,
        "method_lower": obj_l.method_used,
        "method_upper_final": method_u_final,
        "method_lower_final": method_l_final,
        "upper_iter_best": iter_best_from_evals(obj_u.eval_vals, NP, True) if run.record_history else [],
        "lower_iter_best": iter_best_from_evals(obj_l.eval_vals, NP, False) if run.record_history else [],
        "NP": NP,
        "nfev_upper": obj_u.eval_count,
        "nfev_lower": obj_l.eval_count,
        "upper_theta": theta_u,
        "lower_theta": theta_l,
        "upper_marginals": marg_u,
        "lower_marginals": marg_l,
        "upper_residuals": resids_u,
        "lower_residuals": resids_l,
        "res_upper": res_u,
        "res_lower": res_l,
    }


def get_adaptive_parameters(
    d: int,
    K: int,
    r: int,
    popsize: int = 50,
    maxiter: int = 200,
    N_mc_opt: int = 50_000,
    N_mc_final_min: int = 500_000,
    exact_threshold: int = 100_000,
):
    nvar = d * K * (r + 1)
    total_atoms = (K * (r + 1)) ** d
    return {
        "nvar": nvar,
        "total_atoms": int(total_atoms),
        "popsize": int(popsize),
        "maxiter": int(maxiter),
        "nfev_est": int(popsize * max(1, nvar) * (maxiter + 1)),
        "N_mc_opt": int(N_mc_opt),
        "N_mc_final": int(max(N_mc_opt, N_mc_final_min)),
        "exact_threshold_opt": int(exact_threshold),
        "exact_threshold_final": int(exact_threshold),
        "method_expected": "mc" if total_atoms > exact_threshold else "exact",
        "workers": 1,
    }


def _write_run_artifacts(res: dict, cfg_dir: Path, K: int, r: int, names: Optional[Sequence[str]] = None):
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if res["upper_iter_best"]:
        pd.DataFrame({"iter": np.arange(1, len(res["upper_iter_best"]) + 1), "upper_best_pof": res["upper_iter_best"]}).to_csv(cfg_dir / "upper_history.csv", index=False)
    if res["lower_iter_best"]:
        pd.DataFrame({"iter": np.arange(1, len(res["lower_iter_best"]) + 1), "lower_best_pof": res["lower_iter_best"]}).to_csv(cfg_dir / "lower_history.csv", index=False)
    save_marginals_csv(res["upper_marginals"], cfg_dir, K, r, "upper", names=list(names) if names else None)
    save_marginals_csv(res["lower_marginals"], cfg_dir, K, r, "lower", names=list(names) if names else None)


def run_single_configuration(
    prob,
    K: int,
    r: int,
    params: dict,
    true_pof: Optional[float],
    output_dir: str | Path,
    verbose: bool = True,
    side: Optional[str] = None,
    de: Optional[DEControls] = None,
    run: Optional[OUQRunControls] = None,
    names: Optional[Sequence[str]] = None,
):
    if de is None:
        de = DEControls(popsize=params["popsize"], maxiter=params["maxiter"], workers=params.get("workers", 1))
    if run is None:
        run = OUQRunControls(
            N_mc_opt=params["N_mc_opt"],
            N_mc_final=params["N_mc_final"],
            exact_threshold_opt=params["exact_threshold_opt"],
            exact_threshold_final=params["exact_threshold_final"],
        )
    side_use = _side_from_problem(prob, side)
    t0 = time.time()
    res = run_isotropic_ouq_de(prob, K=K, r=r, de=de, run=run, side=side_use, verbose=verbose)
    elapsed = time.time() - t0
    cfg_dir = Path(output_dir) / f"history_K{K}_r{r}"
    _write_run_artifacts(res, cfg_dir, K, r, names=names)
    return {
        "success": True,
        "K": K,
        "r": r,
        "nvar": params["nvar"],
        "total_atoms": params["total_atoms"],
        "popsize": de.popsize,
        "maxiter": de.maxiter,
        "N_mc_opt": run.N_mc_opt,
        "N_mc_final": run.N_mc_final,
        "upper_opt": res["upper"],
        "lower_opt": res["lower"],
        "upper": res["upper_final"],
        "lower": res["lower_final"],
        "width": res["upper_final"] - res["lower_final"],
        "method_upper_final": res["method_upper_final"],
        "method_lower_final": res["method_lower_final"],
        "contains_true": (true_pof is not None) and (res["lower_final"] <= true_pof <= res["upper_final"]),
        "time_seconds": elapsed,
    }


def run_grid(
    prob,
    K_list,
    r_list,
    ref_pof: Optional[float],
    output_dir: str | Path,
    popsize: int = 50,
    maxiter: int = 200,
    side: Optional[str] = None,
):
    rows = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for K in K_list:
        for r in r_list:
            params = get_adaptive_parameters(prob.d, K, r, popsize=popsize, maxiter=maxiter)
            rows.append(run_single_configuration(prob, K, r, params, ref_pof, output_dir=output_dir, side=side))
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / "results.csv", index=False)
    save_ouq_bounds_summary(df, ref_pof, output_dir)
    return df


def run_sweep(
    prob,
    K_list,
    r_list,
    ref_pof: Optional[float],
    output_dir: str | Path,
    de: Optional[DEControls] = None,
    run: Optional[OUQRunControls] = None,
    names: Optional[Sequence[str]] = None,
    verbose: bool = True,
    side: Optional[str] = None,
):
    if de is None:
        de = DEControls()
    if run is None:
        run = OUQRunControls()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    side_use = _side_from_problem(prob, side)
    rows = []
    for K in K_list:
        for r in r_list:
            params = get_adaptive_parameters(
                prob.d,
                K,
                r,
                popsize=de.popsize,
                maxiter=de.maxiter,
                N_mc_opt=run.N_mc_opt,
                N_mc_final_min=run.N_mc_final,
                exact_threshold=run.exact_threshold_opt,
            )
            row = run_single_configuration(
                prob,
                K,
                r,
                params,
                ref_pof,
                output_dir,
                verbose=verbose,
                side=side_use,
                de=de,
                run=run,
                names=names,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
            save_ouq_bounds_summary(pd.DataFrame(rows), ref_pof, output_dir)
    df = pd.DataFrame(rows)
    return SweepResult(dataframe=df, output_dir=output_dir, rows=rows)


# -------------------------- anisotropic exact OUQ --------------------------

def prepare_anisotropic_case(prob_or_dists, case: AnisotropicCase, canonical_eps: float = 1e-9):
    dists = getattr(prob_or_dists, "dists", prob_or_dists)
    return _prepare_anisotropic_case(dists, case, eps=canonical_eps)


class _AnisotropicExactObjective:
    def __init__(self, prob, prep, maximize: bool, side: str, eps: float):
        self.prob = prob
        self.prep = prep
        self.maximize = maximize
        self.side = side
        self.eps = eps
        self.eval_vals: list[float] = []
        self.eval_count = 0

    def __call__(self, theta):
        try:
            marginals, _ = build_marginals_from_theta_anisotropic(self.prep, theta, eps=self.eps)
            pof = pof_from_marginals_exact(marginals, self.prob.g_eval, c_fail=getattr(self.prob, "c_fail", 0.0), side=self.side)
        except Exception:
            pof = 0.0 if self.maximize else 1.0
        val = -pof if self.maximize else pof
        self.eval_vals.append(float(val))
        self.eval_count += 1
        return val


def solve_anisotropic_exact(
    prob,
    prep,
    de: Optional[DEControls] = None,
    canonical_eps: float = 1e-9,
    verbose: bool = True,
):
    if de is None:
        de = DEControls(polish=False)
    case = prep.case
    bounds = [(canonical_eps, 1.0 - canonical_eps)] * case.nvar
    side = _side_from_problem(prob, None)
    NP = de.popsize * max(1, case.nvar)

    obj_u = _AnisotropicExactObjective(prob, prep, maximize=True, side=side, eps=canonical_eps)
    res_u = differential_evolution(
        obj_u,
        bounds,
        strategy=de.strategy,
        popsize=de.popsize,
        mutation=de.mutation,
        recombination=de.recombination,
        maxiter=de.maxiter,
        tol=de.tol,
        atol=de.atol,
        seed=de.seed_upper,
        polish=de.polish,
        disp=verbose,
        workers=de.workers,
        updating="deferred" if de.workers != 1 else "immediate",
    )
    obj_l = _AnisotropicExactObjective(prob, prep, maximize=False, side=side, eps=canonical_eps)
    res_l = differential_evolution(
        obj_l,
        bounds,
        strategy=de.strategy,
        popsize=de.popsize,
        mutation=de.mutation,
        recombination=de.recombination,
        maxiter=de.maxiter,
        tol=de.tol,
        atol=de.atol,
        seed=de.seed_lower,
        polish=de.polish,
        disp=verbose,
        workers=de.workers,
        updating="deferred" if de.workers != 1 else "immediate",
    )
    marg_u, res_u_max = build_marginals_from_theta_anisotropic(prep, res_u.x, eps=canonical_eps)
    marg_l, res_l_max = build_marginals_from_theta_anisotropic(prep, res_l.x, eps=canonical_eps)
    upper = pof_from_marginals_exact(marg_u, prob.g_eval, c_fail=getattr(prob, "c_fail", 0.0), side=side)
    lower = pof_from_marginals_exact(marg_l, prob.g_eval, c_fail=getattr(prob, "c_fail", 0.0), side=side)
    return {
        "case_name": case.name,
        "K_vec": case.K_vec,
        "r_vec": case.r_vec,
        "nvar": case.nvar,
        "total_atoms": case.total_atoms,
        "upper": float(upper),
        "lower": float(lower),
        "width": float(upper - lower),
        "upper_opt_fun": float(res_u.fun),
        "lower_opt_fun": float(res_l.fun),
        "upper_success": bool(res_u.success),
        "lower_success": bool(res_l.success),
        "upper_nit": int(res_u.nit),
        "lower_nit": int(res_l.nit),
        "upper_nfev": int(obj_u.eval_count),
        "lower_nfev": int(obj_l.eval_count),
        "upper_history": iter_best_from_evals(obj_u.eval_vals, NP, True),
        "lower_history": iter_best_from_evals(obj_l.eval_vals, NP, False),
        "max_residual_upper": float(res_u_max),
        "max_residual_lower": float(res_l_max),
        "method_upper_final": "exact",
        "method_lower_final": "exact",
        "theta_upper": res_u.x,
        "theta_lower": res_l.x,
        "marginals_upper": marg_u,
        "marginals_lower": marg_l,
    }


def run_ouq_canonical_de(
    prob,
    K: int,
    r: int,
    de: Optional[DEControls] = None,
    run: Optional[OUQRunControls] = None,
    verbose: bool = True,
    side: Optional[str] = None,
):
    """Compatibility wrapper used by the example scripts."""
    return run_isotropic_ouq_de(prob, K=K, r=r, de=de, run=run, side=_side_from_problem(prob, side), verbose=verbose)
