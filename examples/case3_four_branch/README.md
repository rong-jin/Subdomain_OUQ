# Case 3: 2D four-branch benchmark

The baseline and thresholded four-branch workflows are merged into `run_case3.py`.  The paper settings are `Yc = 0` and `Yc = 2`.

```bash
python run_case3.py --mode grid --yc 0.0 --K-list 1 2 4 8 --r-list 0 1 2
python run_case3.py --mode grid --yc 2.0 --K-list 1 2 4 8 --r-list 0 1 2
python plot_case3.py --summary outputs/case3_four_branch/yc_0/ouq_bounds_summary.csv
```

For a reference Monte Carlo estimate only:

```bash
python run_case3.py --mode ref --yc 2.0 --n-ref 20000000
```
