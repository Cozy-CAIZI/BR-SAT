# BR-SAT

Research code for BR-SAT, a protocol-constrained AI second reader for one patient-level RBPT image and four ordered SAT dilution images.

This repository release candidate contains the five-image preprocessing and model definition, frozen 15-member ensemble inference, examination-level probability aggregation, the prespecified three-class direct-argmax rule and its post-prediction mapping to reactive versus non-reactive, statistical evaluation, and generation of quantitative figures and tables.

## Research-use boundary

BR-SAT is not an autonomous diagnostic system and is not a replacement for expert laboratory interpretation or clinical assessment. A reactive output is a serological interpretation, not a patient-level diagnosis of brucellosis. The software has not been validated for automatic release of results, production HIS/LIS integration, or use outside the reported protocol.

## Input contract

Each examination is represented by exactly five ordered images:

1. `rbpt_image`
2. `sat25_image`
3. `sat50_image`
4. `sat100_image`
5. `sat200_image`

The public inference manifest accepts only `sample_id` and those five paths, with optional SHA-256 columns. It rejects truth fields and clinical metadata. Paths are resolved relative to the manifest file.

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For a CUDA build, install the PyTorch build appropriate for the local CUDA runtime before installing this project. The exact frozen Windows inference environment is recorded in `docs/REPRODUCIBILITY.md`.

## Synthetic example

The images in `examples/synthetic/` are generated placeholders. They are not participant images and do not represent authentic assay reactions.

```bash
python scripts/generate_synthetic_example.py
python scripts/run_frozen_inference.py \
  --manifest examples/synthetic/manifest.csv \
  --weights-root /path/to/models \
  --output-dir results/synthetic \
  --device cpu \
  --member-limit 1
```

`--member-limit 1` is a smoke test only. Study-compatible inference requires all 15 members and their exact hashes.

For the complete synthetic demonstration, omit `--member-limit`. The expected 15-member output and cross-device probability tolerance are recorded in `outputs/expected_demo_output.json`.

```bash
python scripts/verify_demo_output.py --predictions results/synthetic/predictions.csv
```

## Frozen decision rule

For member `m`, logits are divided by that member's prespecified temperature and converted to three-class probabilities. The 15 probability vectors are averaged arithmetically. The primary three-class prediction is the direct argmax of the mean vector. Only after that prediction is fixed is the binary mapping applied:

- `negative` → `non-reactive`
- `weak_positive` or `positive` → `reactive`

The continuous score `q_R = P(weak_positive) + P(positive) = 1 - P(negative)` is used for discrimination analyses only. It is not a replacement threshold for the primary decision.

## Statistical evaluation and quantitative outputs

`scripts/evaluate_predictions.py` accepts a privacy-reviewed case-level CSV with the schema documented in `docs/INPUT_SCHEMAS.md`. It produces cohort metrics, Wilson intervals, patient-level stratified-bootstrap intervals, prospective ROC/PR coordinates, and optional site analyses. `scripts/generate_quantitative_outputs.py` renders the quantitative figures and tables from those outputs.

```bash
python scripts/evaluate_predictions.py \
  --input examples/synthetic/predictions.csv \
  --output-dir results/evaluation \
  --bootstrap-replicates 100

python scripts/generate_quantitative_outputs.py \
  --evaluation-dir results/evaluation \
  --output-dir results/figures
```

No real participant-level input or prediction table is included in this code release candidate.

## Model weights

The exact 15-member manifest and hashes are in `config/frozen_ensemble_v1.0.0.json`. Under the author-approved scope recorded in `config/release_policy_v1.0.0.json`, the frozen model weights are not included in the public `v1.0.0` repository or archive. The repository must not state or imply that the weights are publicly available. The rationale and any access conditions must be documented in the final Code availability statement before the release is cited in the manuscript.

## Data and privacy

See `DATA_ACCESS.md`. No participant-level images, hospital-identifying metadata, truth tables, case/member predictions, error-audit rows, or confidential clinical data belong in this repository.

## Release status

This directory is a release candidate, not yet the public `v1.0.0`. The source code is licensed under BSD-3-Clause. Final release is blocked on documentation of the non-public weight rationale and access conditions, and DOI archiving.

Every push and pull request runs compilation, decision-rule tests, and a public-boundary audit over the working tree and reachable Git history. A `v*` tag additionally requires the approved `LICENSE`, verified `CITATION.cff`, and compliance with the frozen release policy. For `v1.0.0`, the gate fails if any model weight is present or if the non-public weight rationale and access conditions are not documented.
