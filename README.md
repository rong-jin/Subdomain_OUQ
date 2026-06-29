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
