import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from subdomain_ouq.moments import make_edges, raw_to_unit_moments, subinterval_moments_uniform


def test_uniform_subinterval_moments_on_zero_one():
    edges = make_edges(0.0, 1.0, 2)
    M = subinterval_moments_uniform(0.0, 1.0, edges, r=2)
    assert np.allclose(M[:, 0], [0.5, 0.5])
    assert np.allclose(M[:, 1], [0.125, 0.375])
    assert np.allclose(M[:, 2], [1 / 24, 7 / 24])


def test_raw_to_unit_moments_first_half_uniform():
    M_row = np.array([0.5, 0.125, 1 / 24])
    mprime = raw_to_unit_moments(M_row, 0.0, 0.5)
    assert np.allclose(mprime, [1.0, 0.5, 1.0 / 3.0])
