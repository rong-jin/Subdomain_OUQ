# Case 4: 8D rare-event roof-truss problem

This folder contains the exact anisotropic OUQ workflow, McDiarmid screening, subset-simulation reference estimator, and plotting utility.

```bash
python screen_mcdiarmid.py --eta 1.25e5 --exclude-screen-names q,l
python reference_subset_simulation.py --eta 1.25e5
python run_case4.py --eta 1.25e5 --active-names q,l,Ac,fc
python plot_case4.py --summary outputs/case4_roof_truss/ouq_bounds_summary.csv
```

The default active set is `q,l,Ac,fc`, while the remaining variables are kept at the coarsest resolution.
