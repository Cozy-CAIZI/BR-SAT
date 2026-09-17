#!/usr/bin/env python3
"""Evaluate frozen BR-SAT predictions at the development-selected threshold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_SEED = 20260804
REACTIVE_THRESHOLD = 0.572
PREFERRED_COHORT_ORDER = (
    "development_618",
    "internal_160",
    "multicentre_200",
    "prospective_102",
)
REQUIRED = {
    "case_id",
    "cohort",
    "reference_binary_id",
    "frozen_three_class_prediction_id",
    "p_negative",
    "p_weak_positive",
    "p_positive",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"missing evaluation columns: {missing}")
        raw = list(reader)
    rows = []
    seen = set()
    for item in raw:
        key = (item["cohort"], item["case_id"])
        if key in seen:
            raise ValueError(f"duplicate cohort/case_id: {key}")
        seen.add(key)
        y = int(item["reference_binary_id"])
        three_class = int(item["frozen_three_class_prediction_id"])
        probabilities = np.array(
            [float(item["p_negative"]), float(item["p_weak_positive"]), float(item["p_positive"])],
            dtype=float,
        )
        if y not in (0, 1) or three_class not in (0, 1, 2):
            raise ValueError(f"invalid class ID for {key}")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError(f"invalid probabilities for {key}")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError(f"probabilities do not sum to one for {key}")
        rows.append(
            {
                "case_id": item["case_id"],
                "cohort": item["cohort"],
                "site_code": item.get("site_code", "").strip(),
                "y": y,
                "three_class": three_class,
                "q_r": float(probabilities[1] + probabilities[2]),
                "pred": int(float(probabilities[1] + probabilities[2]) >= REACTIVE_THRESHOLD),
            }
        )
    if not rows:
        raise ValueError("evaluation input is empty")
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - margin, centre + margin


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(score.size, dtype=float)
    start = 0
    while start < score.size:
        end = start + 1
        while end < score.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2
        start = end
    statistic = ranks[positive].sum() - n_pos * (n_pos + 1) / 2
    return float(statistic / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted, score_sorted = y[order], score[order]
    tp = fp = 0
    previous_recall = total = 0.0
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and score_sorted[end] == score_sorted[start]:
            end += 1
        group = y_sorted[start:end]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())
        recall = tp / n_pos
        total += (recall - previous_recall) * (tp / (tp + fp))
        previous_recall = recall
        start = end
    return float(total)


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "n": int(y.size),
        "reactive_n": int((y == 1).sum()),
        "non_reactive_n": int((y == 0).sum()),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": safe_div(tp + tn, y.size),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": safe_div(sensitivity + specificity, 2),
        "auroc": auc_score(y, score),
        "auprc": average_precision(y, score),
    }


def bootstrap(y: np.ndarray, pred: np.ndarray, score: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if positive.size == 0 or negative.size == 0:
        return np.empty((0, 4), dtype=float)
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, 4), dtype=float)
    for index in range(replicates):
        sample = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        value = metrics(y[sample], pred[sample], score[sample])
        output[index] = [value["f1"], value["balanced_accuracy"], value["auroc"], value["auprc"]]
    return output


def add_intervals(value: dict, samples: np.ndarray) -> dict:
    output = dict(value)
    for name, successes, total in (
        ("sensitivity", value["tp"], value["tp"] + value["fn"]),
        ("specificity", value["tn"], value["tn"] + value["fp"]),
        ("accuracy", value["tp"] + value["tn"], value["n"]),
        ("ppv", value["tp"], value["tp"] + value["fp"]),
        ("npv", value["tn"], value["tn"] + value["fn"]),
    ):
        output[f"{name}_ci_low"], output[f"{name}_ci_high"] = wilson(successes, total)
        output[f"{name}_ci_method"] = "Wilson"
    for index, name in enumerate(("f1", "balanced_accuracy", "auroc", "auprc")):
        finite = samples[:, index][np.isfinite(samples[:, index])] if samples.size else np.array([])
        low, high = (np.quantile(finite, [0.025, 0.975]) if finite.size else [float("nan")] * 2)
        output[f"{name}_ci_low"] = float(low)
        output[f"{name}_ci_high"] = float(high)
        output[f"{name}_ci_method"] = "stratified patient-level bootstrap percentile"
    return output


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row["y"] for row in rows], dtype=int),
        np.asarray([row["pred"] for row in rows], dtype=int),
        np.asarray([row["q_r"] for row in rows], dtype=float),
    )


def curves(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    y, _, score = arrays(rows)
    if len(np.unique(y)) != 2:
        return [], []
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    tp = fp = 0
    roc = [{"threshold": "inf", "tpr": 0.0, "fpr": 0.0}]
    pr = [{"threshold": "inf", "recall": 0.0, "precision": 1.0}]
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and score[end] == score[start]:
            end += 1
        group = y[start:end]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())
        roc.append({"threshold": score[start], "tpr": tp / positives, "fpr": fp / negatives})
        pr.append({"threshold": score[start], "recall": tp / positives, "precision": tp / (tp + fp)})
        start = end
    return roc, pr


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_path)
    cohorts = list(dict.fromkeys(row["cohort"] for row in rows))
    cohorts.sort(key=lambda item: (PREFERRED_COHORT_ORDER.index(item) if item in PREFERRED_COHORT_ORDER else 99, item))
    cohort_rows = []
    for offset, cohort in enumerate(cohorts):
        subset = [row for row in rows if row["cohort"] == cohort]
        y, pred, score = arrays(subset)
        samples = bootstrap(y, pred, score, args.bootstrap_replicates, args.seed + offset)
        cohort_rows.append({"cohort": cohort, **add_intervals(metrics(y, pred, score), samples)})
    write_csv(output_dir / "cohort_metrics.csv", cohort_rows)

    prospective_key = "prospective_102" if "prospective_102" in cohorts else cohorts[-1]
    roc, pr = curves([row for row in rows if row["cohort"] == prospective_key])
    write_csv(output_dir / "prospective_roc.csv", roc)
    write_csv(output_dir / "prospective_pr.csv", pr)

    site_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["site_code"]:
            site_groups[row["site_code"]].append(row)
    site_rows = []
    for offset, site in enumerate(sorted(site_groups)):
        y, pred, score = arrays(site_groups[site])
        samples = bootstrap(y, pred, score, args.bootstrap_replicates, args.seed + 100 + offset)
        site_rows.append({"site_code": site, **add_intervals(metrics(y, pred, score), samples)})
    write_csv(output_dir / "site_metrics.csv", site_rows)

    manifest = {
        "status": "PASS",
        "primary_rule": "reactive when q_R >= 0.572; threshold selected from development out-of-fold predictions",
        "continuous_score": "q_R = p_weak_positive + p_positive = 1 - p_negative",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "cohorts": cohorts,
        "bootstrap_replicates": args.bootstrap_replicates,
        "base_seed": args.seed,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
