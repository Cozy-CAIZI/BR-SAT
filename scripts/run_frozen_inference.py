#!/usr/bin/env python3
"""Run privacy-minimised BR-SAT frozen ensemble inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from br_sat_decision import (  # noqa: E402
    BINARY_NAMES,
    THREE_CLASS_NAMES,
    aggregate_member_probabilities,
    direct_argmax,
    reactive_score,
    threshold_binary_decision,
)
from multimodal_cv_data import AIRBPTSATDualViewDataset  # noqa: E402
from multimodal_cv_model import AIRBPTSATDualViewModel, DualViewModelConfig  # noqa: E402
from multimodal_data_nvidia import CaseRecord, ROLE_NAMES  # noqa: E402


REQUIRED_COLUMNS = {"sample_id", *(f"{role}_image" for role in ROLE_NAMES)}
OPTIONAL_COLUMNS = {f"{role}_sha256" for role in ROLE_NAMES}
EXPECTED_MEMBERS = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[CaseRecord]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        unexpected = sorted(columns - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)
        if missing:
            raise ValueError(f"missing required manifest columns: {missing}")
        if unexpected:
            raise ValueError(
                "public inference rejects clinical, truth, and unrecognised metadata columns: "
                f"{unexpected}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("inference manifest is empty")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample_id values must be unique")

    records: list[CaseRecord] = []
    for row in rows:
        images: list[Path] = []
        for role in ROLE_NAMES:
            image = Path(row[f"{role}_image"])
            if not image.is_absolute():
                image = path.parent / image
            image = image.resolve()
            if not image.is_file():
                raise FileNotFoundError(f"missing image for {row['sample_id']}:{role}: {image}")
            expected = row.get(f"{role}_sha256", "").strip().lower()
            if expected and sha256_file(image).lower() != expected:
                raise ValueError(f"SHA-256 mismatch for {row['sample_id']}:{role}")
            images.append(image)
        records.append(
            CaseRecord(
                sample_id=row["sample_id"],
                label=None,
                diagnosis_class="",
                center_name="unknown",
                center_index=0,
                images=tuple(images),
                jsons=(None, None, None, None, None),
            )
        )
    return rows, records


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def configure_determinism(device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(20260717)
    np.random.seed(20260717)
    torch.use_deterministic_algorithms(True, warn_only=device.type != "cuda")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260717)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_members(config_path: Path, weights_root: Path, member_limit: int | None) -> tuple[dict, list[dict]]:
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    members = list(payload["members"])
    if len(members) != EXPECTED_MEMBERS:
        raise ValueError(f"expected {EXPECTED_MEMBERS} members, found {len(members)}")
    if member_limit is not None:
        if not 1 <= member_limit <= EXPECTED_MEMBERS:
            raise ValueError("member-limit must be between 1 and 15")
        members = members[:member_limit]
    for member in members:
        checkpoint = weights_root / member["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual = sha256_file(checkpoint)
        if actual.lower() != member["sha256"].lower():
            raise ValueError(f"checkpoint SHA-256 mismatch: {checkpoint}")
        member["resolved_checkpoint"] = str(checkpoint)
    return payload, members


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument(
        "--ensemble-config",
        type=Path,
        default=ROOT / "config" / "frozen_ensemble_v1.0.0.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--member-limit",
        type=int,
        help="non-study smoke-test mode; omit to require all 15 frozen members",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    weights_root = args.weights_root.expanduser().resolve()
    config_path = args.ensemble_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    run_manifest_path = output_dir / "run_manifest.json"
    if prediction_path.exists() or run_manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing prediction or run manifest")

    _, records = read_manifest(manifest_path)
    device = resolve_device(args.device)
    configure_determinism(device)
    ensemble, members = load_members(config_path, weights_root, args.member_limit)
    member_outputs: list[np.ndarray] = []
    started = datetime.now(timezone.utc).isoformat()

    for index, member in enumerate(members, start=1):
        checkpoint_path = Path(member.pop("resolved_checkpoint"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config_values = dict(checkpoint["model_config"])
        config_values["pretrained_backbone"] = False
        model = AIRBPTSATDualViewModel(DualViewModelConfig(**config_values))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval().to(device)
        dataset = AIRBPTSATDualViewDataset(
            records,
            roi_boxes={key: tuple(value) for key, value in checkpoint["roi_boxes"].items()},
            global_image_size=int(checkpoint["global_image_size"]),
            roi_image_size=int(checkpoint["roi_image_size"]),
            training=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(0, args.workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                model_inputs = {
                    key: batch[key].to(device, non_blocking=device.type == "cuda")
                    for key in ("rbpt_global", "rbpt_roi", "sat_global", "sat_roi", "modality_present")
                }
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(**model_inputs)["diagnosis"]
                probability = torch.softmax(logits.float() / float(member["temperature"]), dim=1)
                probabilities.append(probability.cpu().numpy().astype(np.float64))
        member_outputs.append(np.concatenate(probabilities, axis=0))
        print(f"member {index:02d}/{len(members):02d} complete", flush=True)
        del model, checkpoint, dataset, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean = aggregate_member_probabilities(np.stack(member_outputs, axis=0))
    three_class = direct_argmax(mean)
    binary = threshold_binary_decision(mean)
    q_r = reactive_score(mean)
    output_rows = []
    for row_index, record in enumerate(records):
        output_rows.append(
            {
                "sample_id": record.sample_id,
                "ensemble_member_count": len(members),
                "p_negative": f"{mean[row_index, 0]:.10f}",
                "p_weak_positive": f"{mean[row_index, 1]:.10f}",
                "p_positive": f"{mean[row_index, 2]:.10f}",
                "q_reactive": f"{q_r[row_index]:.10f}",
                "frozen_three_class_prediction_id": int(three_class[row_index]),
                "frozen_three_class_prediction": THREE_CLASS_NAMES[int(three_class[row_index])],
                "binary_prediction_id": int(binary[row_index]),
                "binary_prediction": BINARY_NAMES[int(binary[row_index])],
                "decision_rule": "member_temperature_softmax_then_probability_mean_then_q_reactive_ge_0.572",
            }
        )
    write_csv(prediction_path, output_rows)
    run_manifest = {
        "status": "study_compatible_frozen_ensemble" if len(members) == EXPECTED_MEMBERS else "non_study_smoke_test",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": ensemble["model_version"],
        "case_count": len(records),
        "ensemble_member_count": len(members),
        "manifest_sha256": sha256_file(manifest_path),
        "ensemble_config_sha256": sha256_file(config_path),
        "prediction_sha256": sha256_file(prediction_path),
        "decision_rule": {
            "per_member_temperature_calibration": True,
            "probability_aggregation": "arithmetic mean",
            "three_class_decision": "direct argmax retained as a traceability output",
            "binary_decision": "reactive when q_R >= 0.572; non-reactive otherwise",
            "q_R_role": "primary binary decision and continuous discrimination analyses",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "members": [
            {
                "repeat_seed": member["repeat_seed"],
                "outer_fold": member["outer_fold"],
                "checkpoint": member["checkpoint"],
                "sha256": member["sha256"],
                "temperature": member["temperature"],
            }
            for member in members
        ],
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": run_manifest["status"], "predictions": str(prediction_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
