from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


ROLE_NAMES = ("rbpt", "sat25", "sat50", "sat100", "sat200")
REACTION_NAMES = ("no_agglutination", "weak", "strong")
REASON_TEXT_CN = {
    "WEAK_PROBABILITY_NEAR_THRESHOLD": "弱阳性概率接近决策阈值",
    "RBPT_STRONG_VS_ALL_SAT_NONE": "RBPT高置信度强凝集，但四个SAT均高置信度未见凝集",
    "RBPT_NONE_VS_ANY_SAT_STRONG": "RBPT高置信度未见凝集，但至少一个SAT滴度高置信度强凝集",
    "MISSING_IMAGE": "图像缺失",
    "UNREADABLE_IMAGE": "图像无法读取",
    "LOW_RESOLUTION": "图像分辨率低于开发集支持范围",
    "SEVERE_UNDEREXPOSURE": "图像严重欠曝",
    "SEVERE_OVEREXPOSURE": "图像严重过曝",
    "LOW_CONTRAST": "图像对比度过低",
    "EXCESSIVE_CLIPPING": "图像大面积纯黑或纯白截断",
    "POSSIBLE_BLUR": "图像清晰度异常，疑似模糊",
}


def _finite_probability_vector(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (expected,) or not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must contain {expected} finite probabilities")
    if bool((result < 0).any()):
        raise ValueError(f"{name} contains a negative probability")
    total = float(result.sum())
    if total <= 0:
        raise ValueError(f"{name} probabilities sum to zero")
    return result / total


def image_quality_metrics(path: str | Path | None) -> dict:
    """Return deterministic, label-free image quality descriptors.

    These descriptors are intended for a conservative referral rule, not for
    diagnosing the assay. Thresholds must be fitted on the development images
    and frozen before a confirmation cohort is opened.
    """
    if path is None:
        return {"readable": False, "error": "missing"}
    path = Path(path)
    if not path.is_file():
        return {"readable": False, "error": "missing"}
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            width, height = image.size
            thumbnail = image.copy()
            thumbnail.thumbnail((384, 384), Image.Resampling.BILINEAR)
    except (OSError, ValueError) as error:
        return {"readable": False, "error": f"unreadable:{type(error).__name__}"}

    array = np.asarray(thumbnail, dtype=np.float32) / 255.0
    margin_y = max(1, array.shape[0] // 20)
    margin_x = max(1, array.shape[1] // 20)
    if array.shape[0] > 2 * margin_y + 4 and array.shape[1] > 2 * margin_x + 4:
        core = array[margin_y:-margin_y, margin_x:-margin_x]
    else:
        core = array
    if core.shape[0] >= 3 and core.shape[1] >= 3:
        laplacian = (
            -4.0 * core[1:-1, 1:-1]
            + core[:-2, 1:-1]
            + core[2:, 1:-1]
            + core[1:-1, :-2]
            + core[1:-1, 2:]
        )
        laplacian_variance = float(laplacian.var())
    else:
        laplacian_variance = 0.0
    return {
        "readable": True,
        "width": int(width),
        "height": int(height),
        "short_side": int(min(width, height)),
        "mean_luminance": float(core.mean()),
        "contrast_std": float(core.std()),
        "clipped_fraction": float(((core <= 5.0 / 255.0) | (core >= 250.0 / 255.0)).mean()),
        "laplacian_variance": laplacian_variance,
    }


def fit_review_policy(
    records: Iterable,
    *,
    weak_threshold_margin: float = 0.05,
    reaction_confidence_floor: float = 0.80,
    quality_lower_quantile: float = 0.005,
    quality_upper_quantile: float = 0.995,
) -> dict:
    """Fit label-free image support limits and return a prespecified policy.

    Diagnosis labels are deliberately not used. The resulting policy is saved
    before model fitting and embedded unchanged in the frozen ensemble.
    """
    if not 0 < weak_threshold_margin < 0.25:
        raise ValueError("weak_threshold_margin must be between 0 and 0.25")
    if not 0.5 <= reaction_confidence_floor <= 1.0:
        raise ValueError("reaction_confidence_floor must be between 0.5 and 1.0")
    if not 0 <= quality_lower_quantile < quality_upper_quantile <= 1:
        raise ValueError("invalid quality quantiles")

    by_role: dict[str, list[dict]] = collections.defaultdict(list)
    unreadable = []
    for record in records:
        for role, path in zip(ROLE_NAMES, record.images):
            metrics = image_quality_metrics(path)
            if metrics.get("readable"):
                by_role[role].append(metrics)
            else:
                unreadable.append(f"{getattr(record, 'sample_id', 'unknown')}:{role}")
    if unreadable:
        preview = ", ".join(unreadable[:5])
        raise ValueError(f"development quality reference contains missing/unreadable images: {preview}")

    role_limits = {}
    for role in ROLE_NAMES:
        values = by_role.get(role, [])
        if len(values) < 20:
            raise ValueError(f"at least 20 readable development images are required for {role}")

        def quantile(key: str, q: float) -> float:
            return float(np.quantile([float(item[key]) for item in values], q))

        # Quantile limits identify images outside the image domain represented
        # during development. Absolute bounds prevent nonsensical thresholds.
        role_limits[role] = {
            "reference_count": len(values),
            "minimum_short_side": max(64, int(np.floor(quantile("short_side", 0.001)))),
            "minimum_mean_luminance": max(0.005, quantile("mean_luminance", quality_lower_quantile)),
            "maximum_mean_luminance": min(0.995, quantile("mean_luminance", quality_upper_quantile)),
            "minimum_contrast_std": max(0.005, quantile("contrast_std", quality_lower_quantile)),
            "maximum_clipped_fraction": min(0.98, quantile("clipped_fraction", quality_upper_quantile)),
            "minimum_laplacian_variance": max(
                1e-6, quantile("laplacian_variance", quality_lower_quantile)
            ),
        }

    return {
        "version": "airbptsat_clinical_review_v1",
        "status": "prespecified_from_development_images_before_confirmation",
        "locked_validation_used": False,
        "weak_threshold_proximity_margin": float(weak_threshold_margin),
        "reaction_confidence_floor": float(reaction_confidence_floor),
        "discordance_rules": [
            "high-confidence RBPT strong versus four high-confidence SAT no-agglutination results",
            "high-confidence RBPT no-agglutination versus at least one high-confidence SAT strong result",
        ],
        "quality_reference": {
            "method": "role-specific label-free development-image support limits",
            "lower_quantile": float(quality_lower_quantile),
            "upper_quantile": float(quality_upper_quantile),
            "thumbnail_max_side": 384,
            "role_limits": role_limits,
        },
        "clinical_boundary": (
            "Review flags do not overwrite the three-class prediction. RBPT and SAT remain "
            "independent assays, and discordance can be clinically real."
        ),
    }


def assess_case_quality(
    image_paths: Sequence[str | Path | None],
    policy: dict,
) -> dict:
    if len(image_paths) != len(ROLE_NAMES):
        raise ValueError(f"expected {len(ROLE_NAMES)} image paths")
    limits_by_role = policy["quality_reference"]["role_limits"]
    role_results = {}
    all_reason_codes = []
    for role, path in zip(ROLE_NAMES, image_paths):
        metrics = image_quality_metrics(path)
        reasons = []
        if not metrics.get("readable"):
            reasons.append("MISSING_IMAGE" if metrics.get("error") == "missing" else "UNREADABLE_IMAGE")
        else:
            limits = limits_by_role[role]
            if metrics["short_side"] < limits["minimum_short_side"]:
                reasons.append("LOW_RESOLUTION")
            if metrics["mean_luminance"] < limits["minimum_mean_luminance"]:
                reasons.append("SEVERE_UNDEREXPOSURE")
            if metrics["mean_luminance"] > limits["maximum_mean_luminance"]:
                reasons.append("SEVERE_OVEREXPOSURE")
            if metrics["contrast_std"] < limits["minimum_contrast_std"]:
                reasons.append("LOW_CONTRAST")
            if metrics["clipped_fraction"] > limits["maximum_clipped_fraction"]:
                reasons.append("EXCESSIVE_CLIPPING")
            if metrics["laplacian_variance"] < limits["minimum_laplacian_variance"]:
                reasons.append("POSSIBLE_BLUR")
        role_results[role] = {"metrics": metrics, "reason_codes": reasons}
        all_reason_codes.extend(f"{role}:{reason}" for reason in reasons)
    return {
        "poor_quality": bool(all_reason_codes),
        "poor_quality_roles": [role for role, value in role_results.items() if value["reason_codes"]],
        "reason_codes": all_reason_codes,
        "roles": role_results,
    }


def make_review_decision(
    diagnosis_probabilities: Sequence[float],
    weak_threshold: float,
    reaction_probabilities: Sequence[Sequence[float]],
    modality_present: Sequence[bool],
    policy: dict,
    *,
    quality_assessment: dict | None = None,
) -> dict:
    diagnosis = _finite_probability_vector(diagnosis_probabilities, 3, "diagnosis_probabilities")
    reactions = np.asarray(
        [
            _finite_probability_vector(values, 3, f"reaction_probabilities[{index}]")
            for index, values in enumerate(reaction_probabilities)
        ],
        dtype=np.float64,
    )
    if reactions.shape != (len(ROLE_NAMES), len(REACTION_NAMES)):
        raise ValueError("reaction_probabilities must have shape (5, 3)")
    present = np.asarray(modality_present, dtype=bool)
    if present.shape != (len(ROLE_NAMES),):
        raise ValueError("modality_present must contain five values")

    reason_codes = []
    weak_margin = float(policy["weak_threshold_proximity_margin"])
    near_threshold = abs(float(diagnosis[1]) - float(weak_threshold)) <= weak_margin + 1e-12
    if near_threshold:
        reason_codes.append("WEAK_PROBABILITY_NEAR_THRESHOLD")

    reaction_prediction = reactions.argmax(axis=1)
    reaction_confidence = reactions.max(axis=1)
    confidence_floor = float(policy["reaction_confidence_floor"])
    confident_none = (reaction_prediction == 0) & (reaction_confidence >= confidence_floor) & present
    confident_strong = (reaction_prediction == 2) & (reaction_confidence >= confidence_floor) & present

    discordance_pattern = ""
    if present[0] and bool(present[1:].all()) and confident_strong[0] and bool(confident_none[1:].all()):
        discordance_pattern = "RBPT_STRONG_VS_ALL_SAT_NONE"
        reason_codes.append(discordance_pattern)
    elif present[0] and confident_none[0] and bool(confident_strong[1:].any()):
        discordance_pattern = "RBPT_NONE_VS_ANY_SAT_STRONG"
        reason_codes.append(discordance_pattern)

    quality_codes = []
    if quality_assessment is not None:
        quality_codes = list(quality_assessment.get("reason_codes") or [])
        reason_codes.extend(quality_codes)
    # A missing modality is always a workflow-quality problem, even if an
    # upstream manifest did not provide image paths for quality assessment.
    for index, is_present in enumerate(present):
        code = f"{ROLE_NAMES[index]}:MISSING_IMAGE"
        if not is_present and code not in reason_codes:
            reason_codes.append(code)
            quality_codes.append(code)

    reason_codes = list(dict.fromkeys(reason_codes))
    reason_text = []
    for code in reason_codes:
        role, separator, base_code = code.partition(":")
        if separator:
            role_text = "RBPT" if role == "rbpt" else role.upper()
            reason_text.append(f"{role_text}：{REASON_TEXT_CN.get(base_code, base_code)}")
        else:
            reason_text.append(REASON_TEXT_CN.get(code, code))
    return {
        "requires_manual_review": bool(reason_codes),
        "workflow_status": "需复核" if reason_codes else "可自动判读",
        "review_reason_codes": reason_codes,
        "review_reason_cn": reason_text,
        "trigger_weak_threshold_proximity": near_threshold,
        "trigger_assay_discordance": bool(discordance_pattern),
        "trigger_image_quality": bool(quality_codes),
        "assay_discordance_pattern": discordance_pattern,
        "reaction_predicted_ids": reaction_prediction.astype(int).tolist(),
        "reaction_predicted_classes": [REACTION_NAMES[index] for index in reaction_prediction],
        "reaction_confidences": reaction_confidence.astype(float).tolist(),
        "review_policy_version": policy["version"],
    }


def compact_quality_json(assessment: dict) -> str:
    return json.dumps(assessment, ensure_ascii=False, separators=(",", ":"))
