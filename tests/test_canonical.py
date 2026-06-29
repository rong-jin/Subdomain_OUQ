import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from subdomain_ouq.canonical import canonical_from_unit_moments
from subdomain_ouq.measures import build_bin_measure_from_p


def test_canonical_uniform_first_two_moments():
    p = canonical_from_unit_moments(np.array([1.0, 0.5, 1.0 / 3.0]), eps=1e-10)
    assert np.allclose(p, [0.5, 1.0 / 3.0], atol=1e-10)


def test_bin_measure_matches_moments():
    edges = np.array([0.0, 1.0])
    M = np.array([1.0, 0.5, 1.0 / 3.0])
    p_fixed = canonical_from_unit_moments(M, eps=1e-10)
    x, t = build_bin_measure_from_p(edges, 0, p_fixed, [0.5, 0.5, 0.5], M, eps=1e-10)
    assert np.isclose(np.sum(t), M[0])
    assert np.isclose(np.sum(t * x), M[1])
    assert np.isclose(np.sum(t * x**2), M[2])
