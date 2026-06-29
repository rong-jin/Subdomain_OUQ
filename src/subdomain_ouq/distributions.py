"""Distribution helpers for subdomain OUQ examples.

The repository examples use bounded or truncated marginals because the
canonical-moment construction is defined on bounded subdomains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm, norm, truncnorm, uniform as rv_uniform, weibull_min


class BoundedDist:
    """Renormalize any scipy-like distribution on a bounded interval [a, b]."""

    def __init__(self, base, a: float, b: float, name: str = ""):
        if not (a < b):
            raise ValueError("Need a < b for a bounded distribution.")
        self.base = base
        self.a = float(a)
        self.b = float(b)
        self.name = str(name) if name else "dist"
        if hasattr(base, "cdf"):
            z = float(base.cdf(self.b) - base.cdf(self.a))
        else:
            z, _ = quad(lambda x: base.pdf(x), self.a, self.b, epsabs=1e-12, epsrel=1e-12, limit=300)
            z = float(z)
        if z <= 0 or not math.isfinite(z):
            raise ValueError("Degenerate truncation interval.")
        self._Fa = float(base.cdf(self.a)) if hasattr(base, "cdf") else 0.0
        self._Z = z

    def pdf(self, x):
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=float)
        mask = (x >= self.a) & (x <= self.b)
        if np.any(mask):
            out[mask] = self.base.pdf(x[mask]) / self._Z
        return float(out) if out.ndim == 0 else out

    def cdf(self, x):
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=float)
        left = x <= self.a
        right = x >= self.b
        mid = (~left) & (~right)
        out[left] = 0.0
        out[right] = 1.0
        if np.any(mid):
            if hasattr(self.base, "cdf"):
                out[mid] = (self.base.cdf(x[mid]) - self._Fa) / self._Z
            else:
                vals = []
                for xi in x[mid]:
                    v, _ = quad(lambda y: self.pdf(y), self.a, float(xi), epsabs=1e-12, epsrel=1e-12, limit=300)
                    vals.append(v)
                out[mid] = vals
        return float(out) if out.ndim == 0 else out

    def ppf(self, u):
        if not hasattr(self.base, "ppf"):
            raise NotImplementedError("The wrapped base distribution has no ppf().")
        u = np.asarray(u, dtype=float)
        uu = np.clip(u, 0.0, 1.0)
        return self.base.ppf(self._Fa + uu * self._Z)

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.random(size=size))


def truncate_to_interval(base, a: float, b: float, name: str = "") -> BoundedDist:
    return BoundedDist(base, a, b, name=name)


class GaussianMixture:
    """Two-component Gaussian mixture with scipy-like pdf/cdf."""

    def __init__(self, m1: float, s1: float, m2: float, s2: float, w: float):
        self.w = float(w)
        self.n1 = norm(loc=m1, scale=s1)
        self.n2 = norm(loc=m2, scale=s2)

    def pdf(self, x):
        return self.w * self.n1.pdf(x) + (1.0 - self.w) * self.n2.pdf(x)

    def cdf(self, x):
        return self.w * self.n1.cdf(x) + (1.0 - self.w) * self.n2.cdf(x)


def scipy_lognormal_from_mean_cov(mean: float, cov: float):
    sigma_ln = math.sqrt(math.log(1.0 + cov * cov))
    mu_ln = math.log(mean) - 0.5 * sigma_ln * sigma_ln
    return lognorm(s=sigma_ln, scale=math.exp(mu_ln))


def bounded_lognormal(mean: float, cov: float, q: float = 1e-6, name: str = "") -> BoundedDist:
    base = scipy_lognormal_from_mean_cov(mean, cov)
    return BoundedDist(base, float(base.ppf(q)), float(base.ppf(1.0 - q)), name=name)


def bounded_normal(mean: float, cov: float, q: float = 1e-6, name: str = "") -> BoundedDist:
    sigma = abs(mean) * cov if mean != 0 else cov
    base = norm(loc=mean, scale=sigma)
    return BoundedDist(base, float(base.ppf(q)), float(base.ppf(1.0 - q)), name=name)


def truncated_normal(a: float, b: float, mu: float, sigma: float):
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    return truncnorm(a=a_std, b=b_std, loc=mu, scale=sigma)


def uniform(a: float, b: float):
    return rv_uniform(loc=a, scale=b - a)


def truncated_weibull(a: float, b: float, shape: float, scale: float):
    return BoundedDist(weibull_min(c=shape, scale=scale, loc=0.0), a, b, name="truncated_weibull")


def truncated_bimodal_normal_mixture(a: float, b: float, w: float = 0.35) -> BoundedDist:
    return BoundedDist(GaussianMixture(-2.0, 1.0, 2.0, 1.0, w), a, b, name="truncated_bimodal")


def make_pm_bounds_from_means(mean_dict: Dict[str, float], rel_pct: float):
    names = list(mean_dict.keys())
    mu = np.array([mean_dict[k] for k in names], dtype=float)
    delta = rel_pct * np.abs(mu)
    lo = mu - delta
    hi = mu + delta
    return names, mu, lo, hi


@dataclass
class MarginalSpec:
    """Metadata for one bounded marginal distribution.

    The fourth positional argument is intentionally ``dist`` so the examples can
    write ``MarginalSpec(name, a, b, distribution)``.  If ``dist`` is omitted, a
    truncated normal with the supplied ``mu`` and ``sigma`` is created lazily.
    """

    name: str
    a: float
    b: float
    dist: object | None = None
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self):
        self.a = float(self.a)
        self.b = float(self.b)
        if self.dist is None:
            self.dist = truncated_normal(self.a, self.b, self.mu, self.sigma)

    def truncnorm(self):
        return truncated_normal(self.a, self.b, self.mu, self.sigma)
