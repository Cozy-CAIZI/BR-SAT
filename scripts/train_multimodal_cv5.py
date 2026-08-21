from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

# This must be set before importing torch.  It is required by CUDA when
# torch.use_deterministic_algorithms(True) is enabled for deterministic GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from clinical_review import REACTION_NAMES as CLINICAL_REACTION_NAMES
from clinical_review import assess_case_quality
from clinical_review import compact_quality_json
from clinical_review import fit_review_policy
from clinical_review import make_review_decision
from multimodal_cv_data import AIRBPTSATDualViewDataset
from multimodal_cv_data import estimate_fold_roi_boxes
from multimodal_cv_model import AIRBPTSATDualViewModel
from multimodal_cv_model import DualViewModelConfig
from multimodal_cv_model import backbone_parameters, head_parameters
from multimodal_data_nvidia import ROLE_NAMES
from multimodal_data_nvidia import load_development_records, reaction_target_from_json, seed_worker
from multimodal_data_nvidia import stratified_patient_split
from multimodal_model_nvidia import DIAGNOSIS_NAMES, REACTION_NAMES


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BUNDLE_ROOT / "manifests" / "development_train_618_v2_frozen_20260717.csv"
DEFAULT_FOLD_MANIFEST = BUNDLE_ROOT / "manifests" / "development_cv_folds_fixed_r4_20260715.csv"
DEFAULT_RUN_DIR = BUNDLE_ROOT / "runs" / "v2_uniformweight_nested_cv_repeated3"
CHECKPOINT_OBJECTIVES = ("robust_clinical_utility",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIRBPTSAT V2 candidate: R5-B architecture with fixed uniform case weights"
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--fold-manifest", default=str(DEFAULT_FOLD_MANIFEST))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-fold", type=int, default=-1, help="0-based fold; -1 trains all folds")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--global-image-size", type=int, default=192)
    parser.add_argument("--roi-image-size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=min(6, max(2, (os.cpu_count() or 8) // 2)))
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--seeds",
        default="20260715,20260716,20260717",
        help="comma-separated repeated-CV seeds; --only-fold uses the first seed",
    )
    parser.add_argument("--inner-val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--imbalance-mode",
        choices=("weighted_ce", "weighted_sampler", "none", "both"),
        default="weighted_ce",
        help="R4 default avoids double weighting; 'both' is sensitivity analysis only",
    )
    parser.add_argument(
        "--roi-source",
        choices=("zone", "reaction"),
        default="zone",
        help="zone uses rbt_reaction_zone/sat_bottom_zone from inner-training cases",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--head-warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--reaction-weight", type=float, default=0.50)
    parser.add_argument("--reaction-ordinal-weight", type=float, default=0.25)
    parser.add_argument("--domain-weight", type=float, default=0.02)
    parser.add_argument("--modality-dropout", type=float, default=0.08)
    parser.add_argument("--roi-padding", type=float, default=0.12)
    parser.add_argument("--weak-specificity-floor", type=float, default=0.85)
    parser.add_argument("--macro-retention", type=float, default=0.95)
    parser.add_argument("--review-weak-margin", type=float, default=0.05)
    parser.add_argument("--review-reaction-confidence", type=float, default=0.80)
    parser.add_argument("--quality-lower-quantile", type=float, default=0.005)
    parser.add_argument("--quality-upper-quantile", type=float, default=0.995)
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "off"), default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def deterministic_runtime_settings(deterministic: bool) -> dict[str, object]:
    """Configure strict deterministic CUDA execution or fail before training.

    V2 intentionally trades some speed for a reproducible development result.
    Math SDP is used because the R5 log showed that memory-efficient attention
    remained non-deterministic despite the former --deterministic flag.
    """
    if not deterministic:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return {"enabled": False}
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "V2 deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch import"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    sdp = torch.backends.cuda
    required = {
        "enable_flash_sdp": False,
        "enable_mem_efficient_sdp": False,
        "enable_math_sdp": True,
    }
    for name, value in required.items():
        setter = getattr(sdp, name, None)
        if setter is None:
            raise RuntimeError(f"V2 requires torch.backends.cuda.{name} for deterministic attention")
        setter(value)
    cudnn_sdp = getattr(sdp, "enable_cudnn_sdp", None)
    if cudnn_sdp is not None:
        cudnn_sdp(False)
    mha_backend = getattr(torch.backends, "mha", None)
    if mha_backend is not None:
        mha_backend.set_fastpath_enabled(False)
    return {
        "enabled": True,
        "torch_deterministic_algorithms": True,
        "warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "tf32": False,
        "flash_sdp": False,
        "memory_efficient_sdp": False,
        "math_sdp": True,
        "cudnn_sdp": False if cudnn_sdp is not None else "not_available",
        "mha_fastpath": False if mha_backend is not None else "not_available",
    }


def seed_everything(seed: int, deterministic: bool) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return deterministic_runtime_settings(deterministic)


def select_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        print(
            f"CUDA device: {props.name}; compute capability {props.major}.{props.minor}; "
            f"VRAM {props.total_memory / 2**30:.1f} GiB"
        )
        return device
    if allow_cpu:
        print("WARNING: CPU is enabled for diagnostics only.")
        return torch.device("cpu")
    raise SystemExit("CUDA unavailable; run code/check_nvidia_gpu.py first")


def resolve_amp(device: torch.device, requested: str) -> tuple[bool, torch.dtype]:
    if device.type != "cuda" or requested == "off":
        return False, torch.float32
    if requested == "bf16" or (requested == "auto" and torch.cuda.is_bf16_supported()):
        return True, torch.bfloat16
    return True, torch.float16


def grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def class_distribution(records) -> dict[int, int]:
    return dict(collections.Counter(int(record.label) for record in records))


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain one or more unique integers")
    return seeds


def load_fixed_folds(path: str | Path, records, expected_folds: int) -> dict[str, int]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assignments = {row["sample_id"]: int(row["fold"]) for row in rows}
    record_ids = {record.sample_id for record in records}
    if set(assignments) != record_ids:
        missing = sorted(record_ids - set(assignments))[:10]
        extra = sorted(set(assignments) - record_ids)[:10]
        raise ValueError(f"fixed-fold manifest mismatch; missing={missing}, extra={extra}")
    if set(assignments.values()) != set(range(expected_folds)):
        raise ValueError("fixed-fold manifest does not contain exactly the configured outer folds")
    by_id = {record.sample_id: record for record in records}
    for row in rows:
        record = by_id[row["sample_id"]]
        if int(row["class_id"]) != int(record.label):
            raise ValueError(f"fold-label mismatch for {record.sample_id}")
    return assignments


def create_sampler(records, seed: int) -> WeightedRandomSampler:
    counts = class_distribution(records)
    weights = [1.0 / math.sqrt(counts[int(record.label)]) for record in records]
    return WeightedRandomSampler(
        weights,
        num_samples=len(records),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def diagnosis_weights(records, device: torch.device) -> torch.Tensor:
    counts = class_distribution(records)
    result = torch.tensor(
        [1.0 / math.sqrt(max(1, counts.get(index, 0))) for index in range(3)],
        dtype=torch.float32,
        device=device,
    )
    return result / result.mean()


def diagnosis_loss_for_mode(records, device: torch.device, mode: str) -> nn.Module:
    weight = diagnosis_weights(records, device) if mode in {"weighted_ce", "both"} else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05, reduction="none")


def reaction_weights(records, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rbpt, sat = collections.Counter(), collections.Counter()
    for record in records:
        for role_index, json_path in enumerate(record.jsons):
            target = reaction_target_from_json(json_path, role_index)
            if target >= 0:
                (rbpt if role_index == 0 else sat)[target] += 1

    def make(counts):
        values = torch.tensor(
            [1.0 / math.sqrt(max(1, counts.get(index, 0))) for index in range(3)],
            device=device,
        )
        return values / values.mean()

    return make(rbpt), make(sat)


def metrics_from_predictions(targets: list[int], predictions: list[int]) -> dict:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    precisions, recalls, f1s = [], [], []
    for class_id in range(3):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    weak_tp = int(matrix[1, 1])
    weak_fp = int(matrix[:, 1].sum() - weak_tp)
    weak_tn = int(matrix.sum() - matrix[1, :].sum() - weak_fp)
    accuracy = float(np.trace(matrix) / max(1, matrix.sum()))
    macro_f1 = float(np.mean(f1s))
    balanced = float(np.mean(recalls))
    weak_specificity = weak_tn / max(1, weak_tn + weak_fp)
    minimum_recall = float(min(recalls))
    clinical_utility = 0.50 * recalls[1] + 0.30 * macro_f1 + 0.20 * minimum_recall
    robust_clinical_utility = (
        0.35 * f1s[1]
        + 0.30 * macro_f1
        + 0.20 * balanced
        + 0.15 * minimum_recall
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced,
        "weak_precision": precisions[1],
        "weak_sensitivity": recalls[1],
        "weak_specificity": weak_specificity,
        "weak_f1": f1s[1],
        "minimum_class_recall": minimum_recall,
        "clinical_utility": clinical_utility,
        "robust_clinical_utility": robust_clinical_utility,
        "class_precision": precisions,
        "class_recall": recalls,
        "class_f1": f1s,
        "confusion_matrix": matrix.tolist(),
    }


def fp32_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a finite probability simplex without AMP-rounding drift."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = logits.float().softmax(dim=dim)
        probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("non-finite probability encountered")
    return probabilities


def simplex_from_logits(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """FP32-normalized probabilities used only for exported predictions."""
    return fp32_softmax(logits.detach(), dim=dim)


def normalize_probability_vector(values, *, context: str) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all() or (vector < 0).any():
        raise ValueError(f"invalid three-class probability vector: {context}")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"zero-sum three-class probability vector: {context}")
    vector = vector / total
    if abs(float(vector.sum()) - 1.0) > 1e-6:
        raise ValueError(f"probability simplex normalization failed: {context}")
    return [float(value) for value in vector]


def normalize_probability_groups(row: dict, *, context: str) -> None:
    groups = [
        ("probability_negative", "probability_weak_positive", "probability_positive"),
        *[
            tuple(f"{role}_probability_{reaction}" for reaction in CLINICAL_REACTION_NAMES)
            for role in ROLE_NAMES
        ],
    ]
    for columns in groups:
        values = normalize_probability_vector([row[column] for column in columns], context=context)
        row.update(dict(zip(columns, values)))


def masked_reaction_loss(logits, targets, valid, weights, ordinal_weight: float):
    valid = valid.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    logits = logits[valid]
    targets = targets[valid]
    categorical = nn.functional.cross_entropy(
        logits, targets, weight=weights, label_smoothing=0.02
    )
    # R5 Windows hotfix, now part of the frozen V2 source: CUDA BF16 autocast
    # rejects BCE on probabilities and can introduce unstable ordinal gradients.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = fp32_softmax(logits, dim=1)
        probability_at_least_weak = (probabilities[:, 1] + probabilities[:, 2]).clamp(1e-6, 1 - 1e-6)
        probability_strong = probabilities[:, 2].clamp(1e-6, 1 - 1e-6)
        target_at_least_weak = (targets >= 1).float()
        target_strong = (targets == 2).float()
        ordinal = 0.5 * (
            nn.functional.binary_cross_entropy(probability_at_least_weak, target_at_least_weak)
            + nn.functional.binary_cross_entropy(probability_strong, target_strong)
        )
    return categorical + float(ordinal_weight) * ordinal


def move_batch(batch: dict, device: torch.device) -> dict:
    non_blocking = device.type == "cuda"
    keys = (
        "rbpt_global",
        "rbpt_roi",
        "sat_global",
        "sat_roi",
        "modality_present",
        "reaction_targets",
        "reaction_valid",
        "label",
        "center",
        "sample_weight",
    )
    return {key: batch[key].to(device, non_blocking=non_blocking) for key in keys}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    diagnosis_loss: nn.Module,
    *,
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    scaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    accumulation: int,
    reaction_weight: float,
    domain_weight: float,
    domain_alpha: float,
    rbpt_reaction_weights: torch.Tensor,
    sat_reaction_weights: torch.Tensor,
    reaction_ordinal_weight: float,
    freeze_backbone: bool = False,
) -> dict:
    training = optimizer is not None
    model.train(training)
    raw_model = getattr(model, "_orig_mod", model)
    if training and freeze_backbone:
        raw_model.rbpt_encoder.eval()
        raw_model.sat_encoder.eval()
    raw_model.set_domain_alpha(domain_alpha)
    totals = collections.Counter()
    targets, predictions, probabilities, sample_ids = [], [], [], []
    rbpt_targets, rbpt_predictions, sat_targets, sat_predictions = [], [], [], []
    rbpt_reaction_probabilities, sat_reaction_probabilities, modality_present_rows = [], [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, cpu_batch in enumerate(loader):
        batch = move_batch(cpu_batch, device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    batch["rbpt_global"],
                    batch["rbpt_roi"],
                    batch["sat_global"],
                    batch["sat_roi"],
                    batch["modality_present"],
                )
                diagnosis_per_case = diagnosis_loss(outputs["diagnosis"], batch["label"])
                case_weights = batch["sample_weight"].float()
                loss_diagnosis = (diagnosis_per_case * case_weights).sum() / case_weights.sum().clamp_min(1e-6)
                rbpt_loss = masked_reaction_loss(
                    outputs["rbpt_reaction"],
                    batch["reaction_targets"][:, 0],
                    batch["reaction_valid"][:, 0],
                    rbpt_reaction_weights,
                    reaction_ordinal_weight,
                )
                sat_loss = masked_reaction_loss(
                    outputs["sat_reactions"].flatten(0, 1),
                    batch["reaction_targets"][:, 1:].flatten(),
                    batch["reaction_valid"][:, 1:].flatten(),
                    sat_reaction_weights,
                    reaction_ordinal_weight,
                )
                loss_reaction = 0.5 * (rbpt_loss + sat_loss)
                loss_domain = nn.functional.cross_entropy(outputs["domain"], batch["center"])
                loss = loss_diagnosis + reaction_weight * loss_reaction + domain_weight * loss_domain
            if training:
                scaler.scale(loss / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
        batch_size = int(batch["label"].shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        totals["diagnosis_loss"] += float(loss_diagnosis.detach()) * batch_size
        totals["reaction_loss"] += float(loss_reaction.detach()) * batch_size
        totals["domain_loss"] += float(loss_domain.detach()) * batch_size
        probs = simplex_from_logits(outputs["diagnosis"], 1).cpu()
        targets.extend(batch["label"].detach().cpu().tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())
        sample_ids.extend(cpu_batch["sample_id"])
        rbpt_reaction_probs = simplex_from_logits(outputs["rbpt_reaction"], 1).cpu()
        sat_reaction_probs = simplex_from_logits(outputs["sat_reactions"], 2).cpu()
        rbpt_reaction_probabilities.extend(rbpt_reaction_probs.tolist())
        sat_reaction_probabilities.extend(sat_reaction_probs.tolist())
        modality_present_rows.extend(batch["modality_present"].detach().cpu().tolist())
        rbpt_valid = batch["reaction_valid"][:, 0].detach().cpu()
        rbpt_targets.extend(batch["reaction_targets"][:, 0].detach().cpu()[rbpt_valid].tolist())
        rbpt_predictions.extend(rbpt_reaction_probs.argmax(1)[rbpt_valid].tolist())
        sat_valid = batch["reaction_valid"][:, 1:].flatten().detach().cpu()
        sat_targets.extend(batch["reaction_targets"][:, 1:].flatten().detach().cpu()[sat_valid].tolist())
        sat_predictions.extend(sat_reaction_probs.argmax(2).flatten()[sat_valid].tolist())
    metrics = metrics_from_predictions(targets, predictions)
    count = max(1, len(targets))
    metrics.update({key: value / count for key, value in totals.items()})
    metrics["rbpt_reaction_macro_f1"] = metrics_from_predictions(rbpt_targets, rbpt_predictions)["macro_f1"]
    metrics["sat_reaction_macro_f1"] = metrics_from_predictions(sat_targets, sat_predictions)["macro_f1"]
    metrics["targets"] = targets
    metrics["probabilities"] = probabilities
    metrics["sample_ids"] = sample_ids
    metrics["rbpt_reaction_probabilities"] = rbpt_reaction_probabilities
    metrics["sat_reaction_probabilities"] = sat_reaction_probabilities
    metrics["modality_present_rows"] = modality_present_rows
    return metrics


def cosine_schedule(total_steps: int, warmup_steps: int):
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return multiplier


def build_optimizer(model, args, device):
    groups = [
        {"params": list(backbone_parameters(model)), "lr": args.learning_rate * args.backbone_lr_multiplier},
        {"params": list(head_parameters(model)), "lr": args.learning_rate},
    ]
    kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    # Fused AdamW can select kernels that are not covered by strict deterministic
    # execution on the target Windows runtime.  It remains available only for an
    # explicitly non-deterministic diagnostic run, never for the V2 protocol.
    if device.type == "cuda" and not args.deterministic:
        try:
            return AdamW(groups, fused=True, **kwargs)
        except TypeError:
            pass
    return AdamW(groups, **kwargs)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(
    model,
    config,
    fold,
    epoch,
    global_image_size,
    roi_image_size,
    roi_boxes,
    metrics,
    objective,
):
    non_scalar_metric_keys = {
        "targets",
        "probabilities",
        "sample_ids",
        "rbpt_reaction_probabilities",
        "sat_reaction_probabilities",
        "modality_present_rows",
    }
    return {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "fold": fold,
        "epoch": epoch,
        "global_image_size": global_image_size,
        "roi_image_size": roi_image_size,
        "roi_boxes": {role: list(roi_boxes[role]) for role in ROLE_NAMES},
        "tuning_metrics": {
            key: value for key, value in metrics.items() if key not in non_scalar_metric_keys
        },
        "checkpoint_objective": objective,
        "diagnosis_names": DIAGNOSIS_NAMES,
        "reaction_names": REACTION_NAMES,
        "locked_validation_used": False,
        "weak_agglutination_pixel_segmentation": False,
        "roi_reaction_classification": True,
        "high_resolution_roi": True,
        "reaction_evidence_fusion": True,
        "model_version": "AIRBPTSAT_V2_CANDIDATE_20260717",
        "source_recipe": "R5-B technical-hardening only",
        "hard_mining_enabled": False,
        "case_weight_policy": "uniform_1.0_for_every_case",
    }


def objective_value(name: str, metrics: dict) -> float:
    if name == "weak_sensitivity":
        return float(metrics["weak_sensitivity"] + 0.05 * metrics["macro_f1"])
    return float(metrics[name])


def predict_checkpoint(checkpoint_path, records, device, args):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DualViewModelConfig(**checkpoint["model_config"])
    config = DualViewModelConfig(**{**config.to_dict(), "pretrained_backbone": False})
    model = AIRBPTSATDualViewModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    dataset = AIRBPTSATDualViewDataset(
        records,
        roi_boxes=checkpoint["roi_boxes"],
        global_image_size=int(checkpoint["global_image_size"]),
        roi_image_size=int(checkpoint["roi_image_size"]),
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    amp_enabled, amp_dtype = resolve_amp(device, args.amp_dtype)
    rbpt_weights = torch.ones(3, device=device)
    sat_weights = torch.ones(3, device=device)
    dummy_loss = nn.CrossEntropyLoss(reduction="none")
    with torch.inference_mode():
        metrics = run_epoch(
            model,
            loader,
            device,
            dummy_loss,
            optimizer=None,
            scheduler=None,
            scaler=grad_scaler(False),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            accumulation=1,
            reaction_weight=0.0,
            domain_weight=0.0,
            domain_alpha=0.0,
            rbpt_reaction_weights=rbpt_weights,
            sat_reaction_weights=sat_weights,
            reaction_ordinal_weight=0.0,
        )
    return metrics


def tune_weak_threshold(oof_rows: list[dict], specificity_floor: float, macro_retention: float):
    targets = [int(row["target_class_id"]) for row in oof_rows]
    probabilities = np.asarray(
        [[row["probability_negative"], row["probability_weak_positive"], row["probability_positive"]] for row in oof_rows]
    )
    baseline = metrics_from_predictions(targets, probabilities.argmax(1).tolist())
    rows = []
    for threshold in np.linspace(0.01, 0.95, 95):
        predictions = []
        for probability in probabilities:
            if probability[1] >= threshold:
                predictions.append(1)
            else:
                predictions.append(0 if probability[0] >= probability[2] else 2)
        metrics = metrics_from_predictions(targets, predictions)
        rows.append({"weak_threshold": round(float(threshold), 4), **{key: metrics[key] for key in (
            "accuracy", "macro_f1", "balanced_accuracy", "weak_precision", "weak_sensitivity",
            "weak_specificity", "weak_f1", "minimum_class_recall", "clinical_utility"
        )}})
    balanced = max(rows, key=lambda row: (row["macro_f1"], row["weak_sensitivity"]))
    guarded = [
        row for row in rows
        if row["weak_specificity"] >= specificity_floor
        and row["macro_f1"] >= baseline["macro_f1"] * macro_retention
    ]
    if not guarded:
        guarded = [row for row in rows if row["macro_f1"] >= baseline["macro_f1"] * macro_retention]
    sensitivity = max(
        guarded or rows,
        key=lambda row: (row["weak_sensitivity"], row["weak_precision"], row["macro_f1"]),
    )
    return baseline, balanced, sensitivity, rows


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, targets: list[int]) -> tuple[float, float]:
    targets_array = np.asarray(targets, dtype=np.int64)
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.50, 3.00, 101):
        calibrated = apply_temperature(probabilities, float(temperature))
        nll = -float(np.log(np.clip(calibrated[np.arange(len(targets_array)), targets_array], 1e-8, 1.0)).mean())
        if nll < best_nll:
            best_temperature, best_nll = float(temperature), nll
    return best_temperature, best_nll


def rows_from_prediction_metrics(metrics: dict, *, fold: int, repeat_seed: int) -> list[dict]:
    rows = []
    for sample_id, target, probability in zip(
        metrics["sample_ids"], metrics["targets"], metrics["probabilities"]
    ):
        probability = normalize_probability_vector(
            probability, context=f"inner_prediction:{sample_id}"
        )
        rows.append(
            {
            "sample_id": sample_id,
            "fold": fold,
            "repeat_seed": repeat_seed,
            "target_class_id": int(target),
            "probability_negative": float(probability[0]),
            "probability_weak_positive": float(probability[1]),
            "probability_positive": float(probability[2]),
        }
        )
    return rows


def threshold_prediction(probability: list[float] | np.ndarray, weak_threshold: float) -> int:
    probability = np.asarray(probability, dtype=np.float64)
    if probability[1] >= weak_threshold:
        return 1
    return 0 if probability[0] >= probability[2] else 2


def apply_review_workflow(
    oof_rows: list[dict], weak_threshold: float | None, review_policy: dict
) -> list[dict]:
    reviewed_rows = []
    for source in oof_rows:
        row = dict(source)
        diagnosis_probability = [
            float(row["probability_negative"]),
            float(row["probability_weak_positive"]),
            float(row["probability_positive"]),
        ]
        reaction_probabilities = [
            [float(row[f"{role}_probability_{reaction}"]) for reaction in CLINICAL_REACTION_NAMES]
            for role in ROLE_NAMES
        ]
        present = [value == "1" for value in str(row["modality_present"]).split(";")]
        quality = json.loads(row["quality_assessment_json"])
        row_threshold = float(row.get("inner_weak_threshold") or weak_threshold)
        decision = make_review_decision(
            diagnosis_probability,
            row_threshold,
            reaction_probabilities,
            present,
            review_policy,
            quality_assessment=quality,
        )
        prediction = threshold_prediction(diagnosis_probability, row_threshold)
        row.update(
            {
                "predicted_class_id": prediction,
                "requires_manual_review": str(decision["requires_manual_review"]).lower(),
                "workflow_status": decision["workflow_status"],
                "review_reason_codes": ";".join(decision["review_reason_codes"]),
                "review_reason_cn": "；".join(decision["review_reason_cn"]),
                "review_trigger_weak_threshold_proximity": str(
                    decision["trigger_weak_threshold_proximity"]
                ).lower(),
                "review_trigger_assay_discordance": str(
                    decision["trigger_assay_discordance"]
                ).lower(),
                "review_trigger_image_quality": str(decision["trigger_image_quality"]).lower(),
                "assay_discordance_pattern": decision["assay_discordance_pattern"],
                "review_policy_version": decision["review_policy_version"],
                "weak_positive_threshold": row_threshold,
                "weak_threshold_distance": abs(
                    float(row["probability_weak_positive"]) - row_threshold
                ),
            }
        )
        reviewed_rows.append(row)
    return reviewed_rows


def review_workflow_metrics(rows: list[dict]) -> dict:
    targets = [int(row["target_class_id"]) for row in rows]
    predictions = [int(row["predicted_class_id"]) for row in rows]
    referred = [str(row["requires_manual_review"]).lower() == "true" for row in rows]
    automatic_indices = [index for index, value in enumerate(referred) if not value]
    errors = [target != prediction for target, prediction in zip(targets, predictions)]
    referred_error_count = sum(error and review for error, review in zip(errors, referred))
    reason_counts = collections.Counter()
    for row in rows:
        reason_counts.update(code for code in str(row["review_reason_codes"]).split(";") if code)
    class_referral = {}
    for class_id, class_name in enumerate(DIAGNOSIS_NAMES):
        indices = [index for index, target in enumerate(targets) if target == class_id]
        class_referral[class_name] = {
            "count": len(indices),
            "referred": sum(referred[index] for index in indices),
            "referral_rate": sum(referred[index] for index in indices) / max(1, len(indices)),
        }
    automatic_metrics = None
    if automatic_indices:
        automatic_metrics = metrics_from_predictions(
            [targets[index] for index in automatic_indices],
            [predictions[index] for index in automatic_indices],
        )
    return {
        "development_oof_only": True,
        "case_count": len(rows),
        "review_count": sum(referred),
        "review_rate": sum(referred) / max(1, len(rows)),
        "automatic_decision_count": len(automatic_indices),
        "automatic_decision_coverage": len(automatic_indices) / max(1, len(rows)),
        "overall_three_class_metrics_before_review": metrics_from_predictions(targets, predictions),
        "automatic_subset_metrics": automatic_metrics,
        "total_model_errors_before_review": sum(errors),
        "model_errors_referred": referred_error_count,
        "error_capture_rate": referred_error_count / max(1, sum(errors)),
        "referral_rate_by_true_class": class_referral,
        "review_reason_counts": dict(sorted(reason_counts.items())),
        "interpretation": (
            "Automatic-subset performance is conditional on referral and must be reported together "
            "with coverage/review rate. Expert review is not assumed to be error-free."
        ),
        "locked_validation_used": False,
    }


def aggregate_repeated_oof(rows: list[dict], expected_repeats: int) -> list[dict]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    probability_columns = [
        "probability_negative",
        "probability_weak_positive",
        "probability_positive",
        *[
            f"{role}_probability_{reaction}"
            for role in ROLE_NAMES
            for reaction in CLINICAL_REACTION_NAMES
        ],
    ]
    result = []
    for sample_id, sample_rows in sorted(grouped.items()):
        if len(sample_rows) != expected_repeats:
            raise ValueError(
                f"{sample_id} has {len(sample_rows)} repeated outer predictions; "
                f"expected {expected_repeats}"
            )
        for key in ("target_class_id", "fold", "center_id", "modality_present"):
            if len({str(row[key]) for row in sample_rows}) != 1:
                raise ValueError(f"inconsistent {key} across repeated OOF rows for {sample_id}")
        row = {
            "sample_id": sample_id,
            "fold": int(sample_rows[0]["fold"]),
            "center_id": sample_rows[0]["center_id"],
            "target_class_id": int(sample_rows[0]["target_class_id"]),
            "repeat_count": len(sample_rows),
            "repeat_seeds": ";".join(str(item["repeat_seed"]) for item in sample_rows),
            "temperature_mean": float(np.mean([float(item["temperature"]) for item in sample_rows])),
            "inner_weak_threshold": float(
                np.mean([float(item["inner_weak_threshold"]) for item in sample_rows])
            ),
            "modality_present": sample_rows[0]["modality_present"],
            "quality_policy_scope": "inner_train_only_union_across_repeats",
        }
        assessments = [json.loads(item["quality_assessment_json"]) for item in sample_rows]
        merged_roles = {}
        merged_reason_codes = []
        for role in ROLE_NAMES:
            role_reasons = list(
                dict.fromkeys(
                    reason
                    for assessment in assessments
                    for reason in assessment["roles"][role].get("reason_codes", [])
                )
            )
            merged_roles[role] = {
                "metrics": assessments[0]["roles"][role].get("metrics", {}),
                "reason_codes": role_reasons,
            }
            merged_reason_codes.extend(f"{role}:{reason}" for reason in role_reasons)
        merged_quality = {
            "poor_quality": bool(merged_reason_codes),
            "poor_quality_roles": [
                role for role in ROLE_NAMES if merged_roles[role]["reason_codes"]
            ],
            "reason_codes": merged_reason_codes,
            "roles": merged_roles,
        }
        row["poor_quality_roles"] = ";".join(merged_quality["poor_quality_roles"])
        row["quality_assessment_json"] = compact_quality_json(merged_quality)
        for column in probability_columns:
            row[column] = float(np.mean([float(item[column]) for item in sample_rows]))
        normalize_probability_groups(row, context=f"repeated_mean:{sample_id}")
        row["raw_predicted_class_id"] = int(
            np.argmax(
                [
                    row["probability_negative"],
                    row["probability_weak_positive"],
                    row["probability_positive"],
                ]
            )
        )
        result.append(row)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_fold(
    fold,
    repeat_seed,
    records,
    assignments,
    center_to_index,
    run_dir,
    device,
    args,
    review_policy,
):
    # The fold seed is reset before every model so model initialization,
    # augmentation and DataLoader streams do not inherit another fold's state.
    seed_everything(repeat_seed + fold * 1_000_003, args.deterministic)
    fold_dir = run_dir / f"seed_{repeat_seed}" / f"outer_fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    outer_train_records = [record for record in records if assignments[record.sample_id] != fold]
    outer_val_records = [record for record in records if assignments[record.sample_id] == fold]
    inner_train_records, inner_val_records = stratified_patient_split(
        outer_train_records,
        val_fraction=args.inner_val_fraction,
        seed=repeat_seed + fold * 1009,
    )
    fold_review_policy = fit_review_policy(
        inner_train_records,
        weak_threshold_margin=args.review_weak_margin,
        reaction_confidence_floor=args.review_reaction_confidence,
        quality_lower_quantile=args.quality_lower_quantile,
        quality_upper_quantile=args.quality_upper_quantile,
    )
    (fold_dir / "review_policy_inner_train_only.json").write_text(
        json.dumps(fold_review_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    roi_boxes = estimate_fold_roi_boxes(
        inner_train_records,
        padding=args.roi_padding,
        roi_source=args.roi_source,
    )
    (fold_dir / "roi_boxes_inner_train_only.json").write_text(
        json.dumps({role: list(box) for role, box in roi_boxes.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "repeat_seed": repeat_seed,
        "outer_fold": fold,
        "inner_train_count": len(inner_train_records),
        "inner_tuning_count": len(inner_val_records),
        "outer_evaluation_count": len(outer_val_records),
        "inner_train_class_distribution": class_distribution(inner_train_records),
        "inner_tuning_class_distribution": class_distribution(inner_val_records),
        "outer_evaluation_class_distribution": class_distribution(outer_val_records),
        "checkpoint_selection_data": "inner_tuning_only",
        "temperature_and_threshold_selection_data": "inner_tuning_only",
        "outer_fold_used_for_selection": False,
        "roi_source": args.roi_source,
        "imbalance_mode": args.imbalance_mode,
        "case_weight_policy": "uniform_1.0_for_every_case",
        "global_image_size": args.global_image_size,
        "roi_image_size": args.roi_image_size,
        "reaction_ordinal_weight": args.reaction_ordinal_weight,
        "head_warmup_epochs": args.head_warmup_epochs,
    }
    (fold_dir / "nested_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"seed={repeat_seed} outer_fold={fold} inner_train={len(inner_train_records)} "
        f"inner_tune={len(inner_val_records)} outer_eval={len(outer_val_records)}"
    )
    train_dataset = AIRBPTSATDualViewDataset(
        inner_train_records,
        roi_boxes=roi_boxes,
        global_image_size=args.global_image_size,
        roi_image_size=args.roi_image_size,
        training=True,
        modality_dropout=args.modality_dropout,
        base_seed=repeat_seed + fold * 100_003,
    )
    val_dataset = AIRBPTSATDualViewDataset(
        inner_val_records,
        roi_boxes=roi_boxes,
        global_image_size=args.global_image_size,
        roi_image_size=args.roi_image_size,
        training=False,
        base_seed=repeat_seed + fold * 100_003,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        # Workers are recreated each epoch so dataset.set_epoch() changes the
        # augmentation stream deterministically.
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    sampler = (
        create_sampler(inner_train_records, repeat_seed + fold)
        if args.imbalance_mode in {"weighted_sampler", "both"}
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=len(inner_train_records) >= args.batch_size * 2,
        generator=torch.Generator().manual_seed(repeat_seed + fold * 10_003 + 1),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(repeat_seed + fold * 10_003 + 2),
        **loader_kwargs,
    )
    config = DualViewModelConfig(
        num_centers=len(center_to_index),
        pretrained_backbone=not args.no_pretrained_backbone,
    )
    model = AIRBPTSATDualViewModel(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, args, device)
    updates = max(1, math.ceil(len(train_loader) / args.accumulation))
    total_steps = max(1, args.epochs * updates)
    scheduler = LambdaLR(
        optimizer,
        cosine_schedule(total_steps, min(total_steps - 1, args.warmup_epochs * updates)),
    )
    amp_enabled, amp_dtype = resolve_amp(device, args.amp_dtype)
    scaler = grad_scaler(amp_enabled and amp_dtype == torch.float16)
    diagnosis_loss = diagnosis_loss_for_mode(inner_train_records, device, args.imbalance_mode)
    rbpt_weights, sat_weights = reaction_weights(inner_train_records, device)
    best_scores = {name: -float("inf") for name in CHECKPOINT_OBJECTIVES}
    no_clinical_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.head_warmup_epochs
        for parameter in backbone_parameters(model):
            parameter.requires_grad_(not freeze_backbone)
        train_dataset.set_epoch(epoch)
        started = time.time()
        progress = (epoch - 1) / max(1, args.epochs - 1)
        domain_alpha = 2 / (1 + math.exp(-10 * progress)) - 1
        train_metrics = run_epoch(
            model, train_loader, device, diagnosis_loss,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, accumulation=args.accumulation,
            reaction_weight=args.reaction_weight, domain_weight=args.domain_weight,
            domain_alpha=domain_alpha, rbpt_reaction_weights=rbpt_weights,
            sat_reaction_weights=sat_weights,
            reaction_ordinal_weight=args.reaction_ordinal_weight,
            freeze_backbone=freeze_backbone,
        )
        val_metrics = run_epoch(
            model, val_loader, device, diagnosis_loss,
            optimizer=None, scheduler=None, scaler=scaler,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, accumulation=1,
            reaction_weight=args.reaction_weight, domain_weight=args.domain_weight,
            domain_alpha=domain_alpha, rbpt_reaction_weights=rbpt_weights,
            sat_reaction_weights=sat_weights,
            reaction_ordinal_weight=args.reaction_ordinal_weight,
        )
        clinical_improved = False
        for objective in CHECKPOINT_OBJECTIVES:
            score = objective_value(objective, val_metrics)
            if score > best_scores[objective] + 1e-6:
                best_scores[objective] = score
                payload = checkpoint_payload(
                    model,
                    config,
                    fold,
                    epoch,
                    args.global_image_size,
                    args.roi_image_size,
                    roi_boxes,
                    val_metrics,
                    objective,
                )
                save_checkpoint(fold_dir / f"best_{objective}.pth", payload)
                if objective == "robust_clinical_utility":
                    clinical_improved = True
        no_clinical_improvement = 0 if clinical_improved else no_clinical_improvement + 1
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[-1]["lr"],
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weak_sensitivity": train_metrics["weak_sensitivity"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_weak_precision": val_metrics["weak_precision"],
            "val_weak_sensitivity": val_metrics["weak_sensitivity"],
            "val_weak_specificity": val_metrics["weak_specificity"],
            "val_weak_f1": val_metrics["weak_f1"],
            "val_minimum_class_recall": val_metrics["minimum_class_recall"],
            "val_clinical_utility": val_metrics["clinical_utility"],
            "val_robust_clinical_utility": val_metrics["robust_clinical_utility"],
            "val_rbpt_reaction_macro_f1": val_metrics["rbpt_reaction_macro_f1"],
            "val_sat_reaction_macro_f1": val_metrics["sat_reaction_macro_f1"],
            "seconds": round(time.time() - started, 2),
        }
        history.append(row)
        write_csv(fold_dir / "history.csv", history)
        save_checkpoint(
            fold_dir / "last.pth",
            checkpoint_payload(
                model,
                config,
                fold,
                epoch,
                args.global_image_size,
                args.roi_image_size,
                roi_boxes,
                val_metrics,
                "last",
            ),
        )
        print(
            f"seed={repeat_seed} outer_fold={fold} epoch={epoch:03d} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"inner_f1={val_metrics['macro_f1']:.4f} weak_sens={val_metrics['weak_sensitivity']:.4f} "
            f"weak_spec={val_metrics['weak_specificity']:.4f} "
            f"robust_utility={val_metrics['robust_clinical_utility']:.4f} "
            f"time={row['seconds']:.1f}s confusion={val_metrics['confusion_matrix']}"
        )
        if no_clinical_improvement >= args.patience:
            print(
                f"seed={repeat_seed} outer_fold={fold} early stopping: "
                f"inner robust clinical utility unchanged for {args.patience} epochs"
            )
            break
    checkpoint = fold_dir / "best_robust_clinical_utility.pth"
    inner_predictions = predict_checkpoint(checkpoint, inner_val_records, device, args)
    inner_probabilities = np.asarray(inner_predictions["probabilities"], dtype=np.float64)
    temperature, calibration_nll = fit_temperature(
        inner_probabilities, inner_predictions["targets"]
    )
    calibrated_inner = apply_temperature(inner_probabilities, temperature)
    inner_rows = rows_from_prediction_metrics(
        inner_predictions, fold=fold, repeat_seed=repeat_seed
    )
    for row, probability in zip(inner_rows, calibrated_inner):
        row["probability_negative"] = float(probability[0])
        row["probability_weak_positive"] = float(probability[1])
        row["probability_positive"] = float(probability[2])
    inner_baseline, inner_balanced, inner_sensitivity, inner_sweep = tune_weak_threshold(
        inner_rows, args.weak_specificity_floor, args.macro_retention
    )
    write_csv(fold_dir / "inner_threshold_sweep.csv", inner_sweep)
    calibration = {
        "temperature": temperature,
        "temperature_grid_nll": calibration_nll,
        "inner_raw_argmax_metrics_after_temperature": inner_baseline,
        "inner_balanced_threshold": inner_balanced,
        "inner_selected_sensitivity_threshold": inner_sensitivity,
        "selection_data": "inner_tuning_only",
        "outer_fold_used": False,
    }
    (fold_dir / "inner_calibration_and_threshold.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    oof = predict_checkpoint(checkpoint, outer_val_records, device, args)
    calibrated_outer = apply_temperature(np.asarray(oof["probabilities"]), temperature)
    rows = []
    record_by_id = {record.sample_id: record for record in outer_val_records}
    for index, (sample_id, target, probability) in enumerate(
        zip(oof["sample_ids"], oof["targets"], calibrated_outer)
    ):
        record = record_by_id[sample_id]
        rbpt_reaction = oof["rbpt_reaction_probabilities"][index]
        sat_reactions = oof["sat_reaction_probabilities"][index]
        reaction_probabilities = [rbpt_reaction, *sat_reactions]
        quality = assess_case_quality(record.images, fold_review_policy)
        row = {
            "sample_id": sample_id,
            "fold": fold,
            "repeat_seed": repeat_seed,
            "center_id": record.center_name,
            "target_class_id": target,
            "probability_negative": float(probability[0]),
            "probability_weak_positive": float(probability[1]),
            "probability_positive": float(probability[2]),
            "raw_predicted_class_id": int(np.argmax(probability)),
            "temperature": temperature,
            "inner_weak_threshold": float(inner_sensitivity["weak_threshold"]),
            "modality_present": ";".join(
                "1" if value else "0" for value in oof["modality_present_rows"][index]
            ),
            "poor_quality_roles": ";".join(quality["poor_quality_roles"]),
            "quality_assessment_json": compact_quality_json(quality),
            "quality_policy_scope": "inner_train_only",
        }
        for role, role_probabilities in zip(ROLE_NAMES, reaction_probabilities):
            for reaction_name, value in zip(CLINICAL_REACTION_NAMES, role_probabilities):
                row[f"{role}_probability_{reaction_name}"] = float(value)
        rows.append(row)
    write_csv(fold_dir / "outer_oof_predictions.csv", rows)
    return rows, checkpoint, calibration


def synthetic_smoke(device: torch.device, args) -> None:
    """Exercise the actual V2 dual-view sizes, AMP path and ordinal BCE path."""
    model = AIRBPTSATDualViewModel(
        DualViewModelConfig(pretrained_backbone=False, num_centers=2)
    ).to(device)
    model.train()
    amp_enabled, amp_dtype = resolve_amp(device, args.amp_dtype)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        outputs = model(
            torch.randn(1, 3, 192, 192, device=device),
            torch.randn(1, 3, 320, 320, device=device),
            torch.randn(1, 4, 3, 192, 192, device=device),
            torch.randn(1, 4, 3, 320, 320, device=device),
            torch.ones(1, 5, dtype=torch.bool, device=device),
        )
        diagnosis = nn.functional.cross_entropy(
            outputs["diagnosis"], torch.tensor([1], device=device)
        )
        rbpt = masked_reaction_loss(
            outputs["rbpt_reaction"],
            torch.tensor([1], device=device),
            torch.tensor([True], device=device),
            torch.ones(3, device=device),
            ordinal_weight=0.25,
        )
        sat = masked_reaction_loss(
            outputs["sat_reactions"].flatten(0, 1),
            torch.tensor([0, 1, 2, 0], device=device),
            torch.tensor([True, True, True, True], device=device),
            torch.ones(3, device=device),
            ordinal_weight=0.25,
        )
        loss = diagnosis + 0.5 * (rbpt + sat)
    loss.backward()
    diagnostic = simplex_from_logits(outputs["diagnosis"], 1)
    reaction = simplex_from_logits(outputs["sat_reactions"], 2)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("V2 synthetic smoke produced non-finite loss")
    if not torch.allclose(diagnostic.sum(1), torch.ones(1, device=device), atol=1e-6):
        raise AssertionError("V2 diagnostic smoke probability normalization failed")
    if not torch.allclose(reaction.sum(2), torch.ones((1, 4), device=device), atol=1e-6):
        raise AssertionError("V2 reaction smoke probability normalization failed")
    print(
        f"v2_dual_view_amp_reaction_smoke_ok amp={amp_enabled} dtype={amp_dtype} "
        f"loss={float(loss.detach()):.4f}"
    )


def main():
    args = parse_args()
    repeat_seeds = parse_seeds(args.seeds)
    determinism = seed_everything(repeat_seeds[0], args.deterministic)
    device = select_device(args.allow_cpu or args.smoke_test)
    if args.smoke_test:
        synthetic_smoke(device, args)
        return
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    records, center_to_index = load_development_records(args.manifest)
    assignments = load_fixed_folds(args.fold_manifest, records, args.folds)
    if len(records) != 618:
        raise ValueError(f"V2 requires exactly 618 eligible development cases, found {len(records)}")
    input_fingerprint_path = BUNDLE_ROOT / "provenance" / "V2_INPUT_FILE_SHA256.json"
    if not input_fingerprint_path.is_file():
        raise FileNotFoundError(
            "V2 input fingerprint is missing; run code_common/preflight_v2.py before training"
        )
    review_policy = fit_review_policy(
        records,
        weak_threshold_margin=args.review_weak_margin,
        reaction_confidence_floor=args.review_reaction_confidence,
        quality_lower_quantile=args.quality_lower_quantile,
        quality_upper_quantile=args.quality_upper_quantile,
    )
    review_policy_path = run_dir / "review_policy_prespecified.json"
    review_policy_path.write_text(
        json.dumps(review_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        **vars(args),
        "manifest": str(Path(args.manifest).resolve()),
        "fold_manifest": str(Path(args.fold_manifest).resolve()),
        "repeat_seeds": repeat_seeds,
        "eligible_development_cases": len(records),
        "class_distribution": class_distribution(records),
        "center_to_index": center_to_index,
        "locked_validation_used": False,
        "patient_level": True,
        "fixed_outer_folds_from_r3": True,
        "r4_data_version": "AIRBPTSAT_R4_BlindedExpertAudit_HighResROI_20260715",
        "r4_blinded_audit_cases": 56,
        "r4_expert_label_changes": 20,
        "r4_reaction_overlay_images": 295,
        "model_version": "AIRBPTSAT_V2_CANDIDATE_20260717",
        "development_status": "exploratory V2 candidate based on R5-B with technical reproducibility hardening",
        "confirmatory_claim_allowed": False,
        "nested_cross_validation": True,
        "outer_fold_role": "evaluation_only",
        "inner_tuning_role": "early_stopping_checkpoint_temperature_threshold",
        "repeated_seed_count": len(repeat_seeds),
        "imbalance_mode": args.imbalance_mode,
        "weak_agglutination_pixel_segmentation": False,
        "roi_reaction_classification": True,
        "high_resolution_roi": True,
        "reaction_evidence_fusion": True,
        "v2_source_recipe": "R5-B uniform-weight architecture; no new clinical feature or loss branch",
        "hard_mining_enabled": False,
        "class_weighted_ce": args.imbalance_mode in {"weighted_ce", "both"},
        "case_weight_policy": "all eligible cases have sample_weight=1.0; class-weighted CE is retained",
        "sample_weight_range": [1.0, 1.0],
        "strict_determinism": determinism,
        "input_fingerprint_sha256": sha256_file(input_fingerprint_path),
        "trainer_source_sha256": sha256_file(Path(__file__)),
        "roi_estimation_source": (
            f"{args.roi_source} annotations from inner-training cases only; "
            "fixed boxes applied to inner tuning and outer evaluation"
        ),
        "clinical_review_policy": review_policy,
        "clinical_review_policy_sha256": sha256_file(review_policy_path),
        "nested_oof_quality_assessment_scope": "inner-training images only; union across repeated seeds",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    folds = [args.only_fold] if args.only_fold >= 0 else list(range(args.folds))
    if any(fold < 0 or fold >= args.folds for fold in folds):
        raise SystemExit("--only-fold is outside the configured fold range")
    if args.only_fold >= 0:
        repeat_seeds = repeat_seeds[:1]
    all_replicate_oof, checkpoints = [], []
    for repeat_seed in repeat_seeds:
        seed_everything(repeat_seed, args.deterministic)
        for fold in folds:
            fold_rows, checkpoint, calibration = train_fold(
                fold,
                repeat_seed,
                records,
                assignments,
                center_to_index,
                run_dir,
                device,
                args,
                review_policy,
            )
            all_replicate_oof.extend(fold_rows)
            checkpoints.append(
                {
                    "repeat_seed": repeat_seed,
                    "outer_fold": fold,
                    "checkpoint": str(checkpoint.relative_to(run_dir)).replace("\\", "/"),
                    "sha256": sha256_file(checkpoint),
                    "temperature": calibration["temperature"],
                    "inner_weak_threshold": calibration["inner_selected_sensitivity_threshold"][
                        "weak_threshold"
                    ],
                }
            )
    write_csv(run_dir / "outer_oof_predictions_all_repeats.csv", all_replicate_oof)
    if len(folds) != args.folds:
        print("Single-fold diagnostic mode complete; no aggregate R4 OOF summary was created.")
        return
    all_oof = aggregate_repeated_oof(all_replicate_oof, len(repeat_seeds))
    write_csv(run_dir / "oof_predictions_repeated_mean.csv", all_oof)
    targets = [int(row["target_class_id"]) for row in all_oof]
    raw_predictions = [int(row["raw_predicted_class_id"]) for row in all_oof]
    nested_threshold_predictions = [
        threshold_prediction(
            [
                row["probability_negative"],
                row["probability_weak_positive"],
                row["probability_positive"],
            ],
            float(row["inner_weak_threshold"]),
        )
        for row in all_oof
    ]
    baseline = metrics_from_predictions(targets, raw_predictions)
    nested_threshold_metrics = metrics_from_predictions(targets, nested_threshold_predictions)
    reviewed_oof = apply_review_workflow(all_oof, None, review_policy)
    write_csv(run_dir / "oof_predictions_with_clinical_review.csv", reviewed_oof)
    review_summary = review_workflow_metrics(reviewed_oof)
    (run_dir / "oof_clinical_review_metrics.json").write_text(
        json.dumps(review_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    thresholds = {
        "decision_rule": (
            "Each outer prediction uses temperature and weak-positive threshold chosen only "
            "from its corresponding inner tuning split; repeated seeds are averaged."
        ),
        "raw_argmax_oof_metrics": baseline,
        "nested_inner_selected_threshold_oof_metrics": nested_threshold_metrics,
        "inner_threshold_distribution": {
            "count": len(checkpoints),
            "minimum": min(float(item["inner_weak_threshold"]) for item in checkpoints),
            "maximum": max(float(item["inner_weak_threshold"]) for item in checkpoints),
            "mean": float(np.mean([float(item["inner_weak_threshold"]) for item in checkpoints])),
            "values_by_seed_and_fold": [
                {
                    "repeat_seed": item["repeat_seed"],
                    "outer_fold": item["outer_fold"],
                    "inner_weak_threshold": item["inner_weak_threshold"],
                    "temperature": item["temperature"],
                }
                for item in checkpoints
            ],
        },
        "guardrails": {
            "weak_specificity_floor": args.weak_specificity_floor,
            "macro_f1_retention_vs_raw_argmax": args.macro_retention,
        },
        "clinical_review_policy": review_policy,
        "clinical_review_oof_metrics": review_summary,
        "nested_oof_quality_assessment_scope": "inner-training images only; union across repeated seeds",
        "outer_labels_used_for_threshold_selection": False,
        "locked_validation_used": False,
    }
    (run_dir / "oof_thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ensemble = {
        "model_version": "AIRBPTSAT_V2_CANDIDATE_20260717",
        "architecture": "dual_view_highres_roi_reaction_evidence_uniformweight_v2",
        "source_recipe": "R5-B technical-hardening only",
        "outer_fold_count": args.folds,
        "repeat_seed_count": len(repeat_seeds),
        "aggregation": "arithmetic mean across repeated outer-fold predictions per patient",
        "checkpoint_objective": "inner-tuning robust_clinical_utility",
        "review_policy": review_policy,
        "review_policy_sha256": sha256_file(review_policy_path),
        "members": checkpoints,
        "nested_cv_evaluation_complete": True,
        "deployment_model_frozen": False,
        "deployment_note": (
            "These checkpoints estimate exploratory development performance only. Mac-side audit "
            "must freeze the policy before any separate deployment fit or new-cohort evaluation."
        ),
        "locked_validation_used": False,
        "historical_160_200_validation_reused": False,
        "input_fingerprint_sha256": sha256_file(input_fingerprint_path),
        "trainer_source_sha256": sha256_file(Path(__file__)),
    }
    (run_dir / "V2_NESTED_EVALUATION_MANIFEST.json").write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"V2 nested repeated OOF complete: {len(all_oof)} development patients, "
        f"{len(repeat_seeds)} seeds, {len(checkpoints)} fitted models"
    )
    print(
        f"Raw repeated-mean OOF macro-F1={baseline['macro_f1']:.4f}; "
        f"weak sensitivity={baseline['weak_sensitivity']:.4f}. "
        f"Inner-selected thresholds: macro-F1={nested_threshold_metrics['macro_f1']:.4f}; "
        f"weak sensitivity={nested_threshold_metrics['weak_sensitivity']:.4f}; "
        f"weak specificity={nested_threshold_metrics['weak_specificity']:.4f}."
    )
    print(
        f"Clinical review OOF: review rate={review_summary['review_rate']:.4f}; "
        f"automatic-decision coverage={review_summary['automatic_decision_coverage']:.4f}; "
        f"error capture={review_summary['error_capture_rate']:.4f}."
    )
    print("Historical internal/external locked validation was not opened or used.")
    print("Return the V2 results package to Mac for audit; do not use historical locked validation for tuning.")


if __name__ == "__main__":
    main()
