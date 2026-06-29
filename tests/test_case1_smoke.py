import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CASE1 = ROOT / "examples" / "case1_identity_1d"
for p in (SRC, CASE1):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_case1 import make_case, run_ouq_bounds_for_edges, true_pof
from subdomain_ouq.moments import make_edges


def test_case1_uniform_k1_r0_smoke():
    prob = make_case("uniform")
    edges = make_edges(prob.a, prob.b, 1)
    res = run_ouq_bounds_for_edges(prob, edges, r=0, popsize=3, maxiter=1, seed=1)
    ref = true_pof(prob.dist, prob.c, prob.a, prob.b)
    assert 0.0 <= res["lower"]["pof"] <= ref <= res["upper"]["pof"] <= 1.0 + 1e-9
