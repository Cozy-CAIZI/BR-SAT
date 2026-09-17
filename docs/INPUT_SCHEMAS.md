# Input schemas

## Inference manifest

Required UTF-8 CSV columns:

```text
sample_id,rbpt_image,sat25_image,sat50_image,sat100_image,sat200_image
```

Optional integrity columns use the same role plus `_sha256`, for example `rbpt_sha256`. No truth, hospital, participant, sample-number or date fields are accepted by the public inference entry point.

## Evaluation input

Required UTF-8 CSV columns:

```text
case_id,cohort,reference_binary_id,frozen_three_class_prediction_id,p_negative,p_weak_positive,p_positive
```

`reference_binary_id` is `0` for non-reactive and `1` for reactive. `frozen_three_class_prediction_id` is `0` for negative, `1` for weak positive and `2` for positive and is retained for traceability. The script derives `q_R` from the three probabilities and applies the frozen development-selected threshold of 0.572; it does not refit a threshold. An optional non-identifying `site_code` enables descriptive site and leave-one-site-out analyses.

Only a privacy-reviewed, authorised analysis table should be supplied. The repository intentionally contains no real case-level evaluation input.
