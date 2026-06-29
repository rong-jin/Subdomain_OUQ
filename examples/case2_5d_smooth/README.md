# Case 2: 5D nonlinear smooth problem

This folder contains the five-dimensional nonlinear smooth OUQ workflow and the ITS sample-size scan.

Common commands:

```bash
python examples\case2_5d_smooth\run_case2.py --K-list 1 2 4 8 --r-list 0 1 2 --n-its 50000
python examples\case2_5d_smooth\scan_nits.py --K 8 --r 2
python examples\case2_5d_smooth\plot_case2.py --summary outputs/case2_5d_smooth/ouq_bounds_summary.csv
```

The script uses canonical moments, Jacobi eigendecomposition, Differential Evolution, and exact/ITS-adaptive PoF evaluation.
