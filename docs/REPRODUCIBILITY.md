# Reproducibility notes

## Frozen study inference

- Architecture: dual-view, high-resolution ROI and reaction-evidence fusion.
- Ensemble: 15 members (five outer folds × three random seeds).
- Calibration: member-specific frozen temperature scaling.
- Aggregation: arithmetic mean of the 15 three-class probability vectors.
- Primary decision: direct argmax; no additional weak-positive threshold.
- Binary mapping: class 0 to non-reactive; classes 1 and 2 to reactive.

The original Windows inference record reported Python 3.12, a CUDA 12.8 PyTorch build, torchvision 0.26.0+cu128, NumPy 2.4.4 and Pillow 12.2.0. The archived local environment record contained a machine-local PyTorch wheel path and is therefore not copied verbatim into the public repository. Install the matching official PyTorch/CUDA build for the target machine.

Exact reproducibility requires the 15 checkpoint hashes in `config/frozen_ensemble_v1.0.0.json`. A successful one-member CPU smoke test demonstrates code-path execution only; it is not the study inference acceptance test.
