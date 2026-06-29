# Subdomain OUQ

Reference code and reproduction scripts for the paper:

> Rong Jin and Xingsheng Sun, **Optimal uncertainty quantification under general moment constraints on input subdomains**, *Computer Methods in Applied Mechanics and Engineering*, 461, 119177, 2026. DOI: `10.1016/j.cma.2026.119177`

## Overview

This repository implements an optimal uncertainty quantification (OUQ) framework for independent uncertain inputs with truncated moment constraints defined on input subdomains. The infinite-dimensional optimization over admissible probability measures is reduced to a finite-dimensional search over Dirac measures. Canonical moments are used to enforce moment constraints, the Jacobi matrix is used to recover Dirac supports and weights, and exact tensor-product summation or inverse transform sampling (ITS) is used to evaluate probabilities of failure.

## Repository structure

```text
Subdomain_OUQ/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/subdomain_ouq/              Core reusable OUQ utilities
├── examples/case1_identity_1d/     One-dimensional identity-function examples
├── examples/case2_5d_smooth/       Five-dimensional nonlinear smooth example
├── examples/case3_four_branch/     Two-dimensional four-branch non-smooth example
├── examples/case4_roof_truss/      Eight-dimensional roof-truss rare-event example
├── examples/case5_ballistic_10d/   Ten-dimensional ballistic-impact surrogate example
├── results/                        Curated paper-result CSV files
└── tests/                          Basic smoke and numerical consistency tests
```

## Installation

```bash
git clone https://github.com/rong-jin/Subdomain_OUQ.git
cd Subdomain_OUQ
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The scripts work when run from the repository root because each example adds `src/` to `sys.path`.

## Quick checks

```bash
pytest -q
```

Plot the bundled curated result CSV files without rerunning the expensive optimizations:

```bash
python examples/case1_identity_1d/plot_case1.py
python examples/case2_5d_smooth/plot_case2.py
python examples/case3_four_branch/plot_case3.py
python examples/case4_roof_truss/plot_case4.py
python examples/case5_ballistic_10d/plot_case5.py
```

## Run examples

### Case 1: one-dimensional identity functions

```bash
python examples/case1_identity_1d/run_case1.py --dist all --K-list 1 2 4 8 --r-list 0 1 2 3
python examples/case1_identity_1d/plot_case1.py --summary outputs/case1_identity_1d/case1_bounds.csv --pdfs
```

### Case 2: five-dimensional nonlinear smooth problem

```bash
python examples/case2_5d_smooth/run_case2.py --K-list 1 2 4 8 --r-list 0 1 2 --n-its 50000
python examples/case2_5d_smooth/scan_nits.py --K 8 --r 2
python examples/case2_5d_smooth/plot_case2.py --summary outputs/case2_5d_smooth/ouq_bounds_summary.csv
```

### Case 3: two-dimensional four-branch problem

```bash
python examples/case3_four_branch/run_case3.py --mode grid --yc 0.0 --K-list 1 2 4 8 --r-list 0 1 2
python examples/case3_four_branch/run_case3.py --mode grid --yc 2.0 --K-list 1 2 4 8 --r-list 0 1 2
python examples/case3_four_branch/plot_case3.py --summary outputs/case3_four_branch/yc_0/ouq_bounds_summary.csv
```

### Case 4: eight-dimensional roof-truss rare event

```bash
python examples/case4_roof_truss/screen_mcdiarmid.py --eta 1.25e5 --exclude-screen-names q,l
python examples/case4_roof_truss/reference_subset_simulation.py --eta 1.25e5
python examples/case4_roof_truss/run_case4.py --eta 1.25e5 --active-names q,l,Ac,fc
python examples/case4_roof_truss/plot_case4.py --summary outputs/case4_roof_truss/ouq_bounds_summary.csv
```

Case 4 is computationally expensive. Use smaller `--maxiter`, smaller `--max-exact-atoms`, or the bundled `results/case4_bounds.csv` for quick plotting.

### Case 5: ten-dimensional ballistic impact

A pretrained `forward_model.pth` is included under `examples/case5_ballistic_10d/models/`, so the OUQ runs do not require LS-DYNA.

```bash
python examples/case5_ballistic_10d/deploy_forward_model.py --help
python examples/case5_ballistic_10d/run_case5.py --yc 0.93 --K-list 1 2 4 8 --r-list 0 1 2
python examples/case5_ballistic_10d/scan_thresholds.py
python examples/case5_ballistic_10d/plot_case5.py --summary <run-output-dir>/ouq_bounds_summary.csv
```

The training script is included for completeness:

```bash
python examples/case5_ballistic_10d/train_surrogate.py --help
```

It expects the LS-DYNA-generated input/output dataset used for training; the dataset itself is not included.

## Curated results

The `results/` folder contains lightweight CSV summaries corresponding to the paper examples and appendix tables:

```text
results/case1_bounds.csv
results/case2_bounds.csv
results/case3_bounds_yc0.csv
results/case3_bounds_yc2.csv
results/case4_bounds.csv
results/case5_bounds.csv
```

Generated histories, atoms, logs, and figures are written to output folders such as `outputs/`, `ouq_5d_full_grid/`, or timestamped case directories. These runtime outputs are ignored by Git.

## Citation

```bibtex
@article{jin2026subdomainouq,
  title   = {Optimal uncertainty quantification under general moment constraints on input subdomains},
  author  = {Jin, Rong and Sun, Xingsheng},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {461},
  pages   = {119177},
  year    = {2026},
  doi     = {10.1016/j.cma.2026.119177}
}
```

## License

This project is released under the MIT License.
