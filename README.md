# Subdomain OUQ

This repository contains the reference implementation and numerical examples for:

> Rong Jin and Xingsheng Sun,  
> **Optimal uncertainty quantification under general moment constraints on input subdomains**,  
> *Computer Methods in Applied Mechanics and Engineering*, 461, 119177, 2026.  
> DOI: 10.1016/j.cma.2026.119177

## Overview

This code implements an optimal uncertainty quantification (OUQ) framework for systems with statistically independent uncertain inputs characterized by truncated moment constraints on input subdomains.

The implementation converts the original infinite-dimensional optimization problem over admissible probability measures into a finite-dimensional optimization problem over discrete Dirac measures. The constrained moment problem is parameterized by free canonical moments, and the Dirac supports and weights are recovered through the eigendecomposition of a Jacobi matrix. For high-dimensional examples, inverse transform sampling (ITS) is used to reduce the cost of probability-of-failure evaluation.

## Features

- Subdomain-based truncated moment constraints
- Canonical-moment parameterization
- Jacobi-matrix eigendecomposition for Dirac supports and weights
- Exact PoF evaluation for low-dimensional examples
- ITS-based PoF estimation for high-dimensional examples
- Differential Evolution optimization for upper and lower OUQ bounds
- Reproduction scripts for the five numerical examples in the paper

## Repository structure

```text
src/subdomain_ouq/          Core OUQ implementation
examples/case1_identity_1d/ 1D identity-function examples
examples/case2_5d_smooth/   5D nonlinear smooth example
examples/case3_four_branch/ 2D non-smooth four-branch example
examples/case4_roof_truss/  8D rare-event roof-truss example
examples/case5_ballistic_10d/ 10D ballistic-impact example
results/                    Curated reference results
tests/                      Basic tests
```
## Installation

Clone the repository:

git clone https://github.com/rong-jin/Subdomain_OUQ.git
cd Subdomain_OUQ

Create a Python environment:

conda create -n subdomain-ouq python=3.11
conda activate subdomain-ouq
pip install -r requirements.txt

For the ballistic-impact surrogate model in Case 5, PyTorch is required:

pip install torch
Quick start

Run the one-dimensional examples:

python examples/case1_identity_1d/run_case1.py --dist all --K-list 1 2 4 8 --r-list 0 1 2 3

Run the five-dimensional nonlinear smooth example:

python examples/case2_5d_smooth/run_case2.py --K-list 1 2 4 8 --r-list 0 1 2 --n-its 50000

Run the two-dimensional four-branch example:

python examples/case3_four_branch/run_case3.py --yc 0.0
python examples/case3_four_branch/run_case3.py --yc 2.0

Run the roof-truss rare-event example:

python examples/case4_roof_truss/screen_mcdiarmid.py
python examples/case4_roof_truss/run_case4.py

Run the ten-dimensional ballistic-impact example:

python examples/case5_ballistic_10d/run_case5.py --yc 0.93
python examples/case5_ballistic_10d/scan_thresholds.py

## Numerical examples

The repository reproduces the five examples in the paper:

Case	Description	Dimension	Main feature
Case 1	Identity functions with normal, uniform, Weibull, and bimodal distributions	1	Baseline verification
Case 2	Nonlinear smooth function	5	ITS and DE convergence
Case 3	Four-branch non-smooth benchmark	2	Non-smooth and low-probability cases
Case 4	Roof-truss rare-event problem	8	Active-dimension refinement
Case 5	Ballistic impact of AZ31B plate	10	Neural-network surrogate model
Outputs

By default, scripts write generated files to outputs/. These files are not tracked by Git.

Curated reference results used in the paper are stored in results/.

Reproducibility notes

The OUQ optimization uses Differential Evolution and, for high-dimensional problems, inverse transform sampling with common random numbers. Small numerical differences may occur across platforms and random seeds.

The ballistic-impact example uses a pretrained neural-network surrogate model. The original LS-DYNA simulations are not required to run the OUQ example.

---
## Citation

If you use this code, please cite:

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

---
## License

This project is released under the MIT License.
