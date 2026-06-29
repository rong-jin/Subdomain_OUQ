# Case 1: 1D identity-function examples

This folder contains the consolidated code for the four one-dimensional identity examples:

- truncated normal on `[-5, 5]`, `Yc = 0.7`
- uniform on `[-5, 5]`, `Yc = 1.7`
- truncated Weibull on `[0, 10]`, `Yc = 3.0`
- truncated bimodal normal mixture on `[-5, 5]`, `Yc = 1.3`

Run all paper settings:

```bash
python examples/case1_identity_1d/run_case1.py --dist all --K-list 1 2 4 8 --r-list 0 1 2 3
python examples/case1_identity_1d/plot_case1.py --summary outputs/case1_identity_1d/case1_bounds.csv --pdfs
```

`run_case1.py` writes per-distribution atoms and a combined `case1_bounds.csv` summary under `outputs/case1_identity_1d/`.
