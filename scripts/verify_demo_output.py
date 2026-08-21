#!/usr/bin/env python3
"""Verify the complete synthetic demonstration against its expected output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "outputs" / "expected_demo_output.json",
    )
    args = parser.parse_args()
    expected_payload = json.loads(args.expected.read_text(encoding="utf-8"))
    expected = expected_payload["expected"]
    tolerance = float(expected_payload["probability_tolerance"])
    with args.predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError(f"expected one synthetic prediction row, found {len(rows)}")
    actual = rows[0]
    exact_fields = (
        "sample_id",
        "ensemble_member_count",
        "frozen_three_class_prediction_id",
        "frozen_three_class_prediction",
        "binary_prediction_id",
        "binary_prediction",
    )
    expected_values = {**expected, "ensemble_member_count": expected_payload["ensemble_member_count"]}
    for field in exact_fields:
        if str(actual[field]) != str(expected_values[field]):
            raise AssertionError(f"{field}: expected {expected_values[field]!r}, found {actual[field]!r}")
    for field in ("p_negative", "p_weak_positive", "p_positive", "q_reactive"):
        delta = abs(float(actual[field]) - float(expected[field]))
        if delta > tolerance:
            raise AssertionError(f"{field}: difference {delta} exceeds tolerance {tolerance}")
    probability_sum = sum(float(actual[field]) for field in ("p_negative", "p_weak_positive", "p_positive"))
    if abs(probability_sum - 1.0) > tolerance:
        raise AssertionError("three-class probabilities do not sum to one")
    print(json.dumps({"status": "PASS", "prediction": str(args.predictions), "tolerance": tolerance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
