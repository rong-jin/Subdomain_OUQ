# Case 5: 10D ballistic-impact problem

This folder contains the ten-dimensional ballistic-impact OUQ workflow using a pretrained neural-network surrogate.

Files:

- `models/forward_model.pth`: pretrained surrogate model checkpoint
- `run_case5.py`: OUQ bounds for fixed `Yc`
- `scan_thresholds.py`: fixed-`K,r` scan over failure thresholds / target PoFs
- `train_surrogate.py`: training workflow for the surrogate if LS-DYNA data are available
- `deploy_forward_model.py`: standalone inference helper

Run OUQ for `Yc = 0.93`:

```bash
python run_case5.py --yc 0.93 --K-list 1 2 4 8 --r-list 0 1 2
```

Run the threshold scan:

```bash
python scan_thresholds.py
```

The pretrained surrogate is included so LS-DYNA is not required for OUQ evaluation.
