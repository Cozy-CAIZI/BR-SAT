"""Frozen BR-SAT aggregation and post-prediction binary mapping."""

from __future__ import annotations

import numpy as np


THREE_CLASS_NAMES = ("negative", "weak_positive", "positive")
BINARY_NAMES = ("non-reactive", "reactive")


def aggregate_member_probabilities(member_probabilities: np.ndarray) -> np.ndarray:
    """Return the arithmetic mean across frozen ensemble members."""
    values = np.asarray(member_probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("expected shape (members, examinations, 3)")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("at least one member and one examination are required")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("each member probability vector must sum to one")
    return values.mean(axis=0)


def direct_argmax(mean_probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(mean_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("expected shape (examinations, 3)")
    return np.argmax(values, axis=1).astype(np.int64)


def map_three_class_to_binary(class_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(class_ids, dtype=np.int64)
    if not np.isin(values, [0, 1, 2]).all():
        raise ValueError("three-class IDs must be 0, 1 or 2")
    return (values > 0).astype(np.int64)


def reactive_score(mean_probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(mean_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("expected shape (examinations, 3)")
    return values[:, 1] + values[:, 2]
