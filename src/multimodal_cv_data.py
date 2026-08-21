from __future__ import annotations

import collections
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from multimodal_data_nvidia import CaseRecord
from multimodal_data_nvidia import IMAGENET_MEAN, IMAGENET_STD
from multimodal_data_nvidia import ROLE_NAMES
from multimodal_data_nvidia import reaction_target_from_json


ROI_ZONE_LABELS = {
    "rbpt": "rbt_reaction_zone",
    "sat25": "sat_bottom_zone",
    "sat50": "sat_bottom_zone",
    "sat100": "sat_bottom_zone",
    "sat200": "sat_bottom_zone",
}
ROI_REACTION_LABELS = {
    "rbpt": {"rbt_agglutinate_weak", "rbt_agglutinate_strong"},
    "sat25": {"sat_agglutinate_weak", "sat_agglutinate_strong"},
    "sat50": {"sat_agglutinate_weak", "sat_agglutinate_strong"},
    "sat100": {"sat_agglutinate_weak", "sat_agglutinate_strong"},
    "sat200": {"sat_agglutinate_weak", "sat_agglutinate_strong"},
}
DEFAULT_ROI_BOXES = {
    "rbpt": (0.12, 0.12, 0.88, 0.88),
    "sat25": (0.18, 0.38, 0.82, 0.98),
    "sat50": (0.18, 0.38, 0.82, 0.98),
    "sat100": (0.18, 0.38, 0.82, 0.98),
    "sat200": (0.18, 0.38, 0.82, 0.98),
}


def hospital_stratified_patient_folds(
    records: list[CaseRecord],
    *,
    n_splits: int = 5,
    seed: int = 20260714,
) -> dict[str, int]:
    """Deterministic patient-level folds stratified jointly by hospital and label."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("duplicate patient/sample_id in development records")
    strata: dict[tuple[str, int], list[CaseRecord]] = collections.defaultdict(list)
    for record in records:
        if record.label is None:
            raise ValueError("cross-validation received an unlabeled record")
        strata[(record.center_name, int(record.label))].append(record)

    assignments: dict[str, int] = {}
    fold_sizes = [0] * n_splits
    fold_class_counts = [collections.Counter() for _ in range(n_splits)]
    rng = random.Random(seed)
    # Large and rare strata first. For each stratum, start at the currently least
    # represented fold for that class and then round-robin.
    for (center, label), group in sorted(
        strata.items(), key=lambda item: (-len(item[1]), item[0][1], item[0][0])
    ):
        group = sorted(group, key=lambda record: record.sample_id)
        rng.shuffle(group)
        fold_order = list(range(n_splits))
        rng.shuffle(fold_order)
        fold_order.sort(key=lambda fold: (fold_class_counts[fold][label], fold_sizes[fold]))
        for index, record in enumerate(group):
            fold = fold_order[index % n_splits]
            assignments[record.sample_id] = fold
            fold_sizes[fold] += 1
            fold_class_counts[fold][label] += 1

    labels = sorted({int(record.label) for record in records})
    for fold in range(n_splits):
        fold_records = [record for record in records if assignments[record.sample_id] == fold]
        if not fold_records:
            raise ValueError(f"fold {fold} is empty")
        missing = [label for label in labels if not any(int(record.label) == label for record in fold_records)]
        if missing:
            raise ValueError(f"fold {fold} is missing diagnosis classes {missing}")
    return assignments


def write_cv_manifest(
    path: str | Path,
    records: list[CaseRecord],
    assignments: dict[str, int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "fold", "diagnosis_class", "class_id", "center_id"),
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.sample_id):
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "fold": assignments[record.sample_id],
                    "diagnosis_class": record.diagnosis_class,
                    "class_id": record.label,
                    "center_id": record.center_name,
                }
            )


def _shape_bbox(shape: dict, width: float, height: float) -> tuple[float, float, float, float] | None:
    points = shape.get("points") or []
    if not points:
        return None
    if shape.get("shape_type") == "circle" and len(points) == 2:
        cx, cy = float(points[0][0]), float(points[0][1])
        px, py = float(points[1][0]), float(points[1][1])
        radius = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        return (
            (cx - radius) / max(1.0, width),
            (cy - radius) / max(1.0, height),
            (cx + radius) / max(1.0, width),
            (cy + radius) / max(1.0, height),
        )
    xs = [float(point[0]) / max(1.0, width) for point in points]
    ys = [float(point[1]) / max(1.0, height) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def estimate_fold_roi_boxes(
    train_records: list[CaseRecord],
    *,
    padding: float = 0.10,
    roi_source: str = "zone",
) -> dict[str, tuple[float, float, float, float]]:
    """Estimate one normalized crop per modality from inner-training annotations only.

    No held-out patient annotation is needed to apply these boxes. The robust
    medians reduce sensitivity to individual annotators and camera framing.
    """
    if roi_source not in {"zone", "reaction"}:
        raise ValueError("roi_source must be 'zone' or 'reaction'")
    boxes_by_role: dict[str, list[tuple[float, float, float, float]]] = collections.defaultdict(list)
    for record in train_records:
        for role, json_path in zip(ROLE_NAMES, record.jsons):
            if json_path is None or not json_path.is_file():
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            width = float(payload.get("imageWidth") or 1)
            height = float(payload.get("imageHeight") or 1)
            reaction_boxes = []
            zone_boxes = []
            for shape in payload.get("shapes", []):
                label = str(shape.get("label", "")).strip()
                if label not in ROI_REACTION_LABELS[role] and label != ROI_ZONE_LABELS[role]:
                    continue
                box = _shape_bbox(shape, width, height)
                if box is not None:
                    (reaction_boxes if label in ROI_REACTION_LABELS[role] else zone_boxes).append(box)
            # R3 defaults to assay-zone geometry for every role so negative and
            # non-agglutinating images contribute to ROI estimation. The old
            # reaction-derived SAT crop remains available only as a prespecified
            # sensitivity analysis.
            shape_boxes = zone_boxes if roi_source == "zone" or role == "rbpt" else reaction_boxes
            if shape_boxes:
                boxes_by_role[role].append(
                    (
                        min(box[0] for box in shape_boxes),
                        min(box[1] for box in shape_boxes),
                        max(box[2] for box in shape_boxes),
                        max(box[3] for box in shape_boxes),
                    )
                )

    # A rare fold/modality may contain no requested annotation. In that case,
    # repeat the pass using assay-zone labels from inner-training patients.
    missing_roles = [role for role in ROLE_NAMES if not boxes_by_role.get(role)]
    pooled_sat_boxes = [
        box
        for role in ROLE_NAMES[1:]
        for box in boxes_by_role.get(role, [])
    ]
    for role in list(missing_roles):
        if role != "rbpt" and pooled_sat_boxes:
            boxes_by_role[role].extend(pooled_sat_boxes)
            missing_roles.remove(role)
    if missing_roles:
        for record in train_records:
            for role, json_path in zip(ROLE_NAMES, record.jsons):
                if role not in missing_roles or json_path is None or not json_path.is_file():
                    continue
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                width = float(payload.get("imageWidth") or 1)
                height = float(payload.get("imageHeight") or 1)
                zone_boxes = [
                    box
                    for shape in payload.get("shapes", [])
                    if str(shape.get("label", "")).strip() == ROI_ZONE_LABELS[role]
                    for box in [_shape_bbox(shape, width, height)]
                    if box is not None
                ]
                if zone_boxes:
                    boxes_by_role[role].append(
                        (
                            min(box[0] for box in zone_boxes),
                            min(box[1] for box in zone_boxes),
                            max(box[2] for box in zone_boxes),
                            max(box[3] for box in zone_boxes),
                        )
                    )

    result = {}
    for role in ROLE_NAMES:
        values = boxes_by_role.get(role, [])
        if values:
            median = np.median(np.asarray(values, dtype=np.float64), axis=0).tolist()
            x1, y1, x2, y2 = median
        else:
            x1, y1, x2, y2 = DEFAULT_ROI_BOXES[role]
        width = max(0.10, x2 - x1)
        height = max(0.10, y2 - y1)
        x1 = max(0.0, x1 - width * padding)
        y1 = max(0.0, y1 - height * padding)
        x2 = min(1.0, x2 + width * padding)
        y2 = min(1.0, y2 + height * padding)
        min_width, min_height = ((0.42, 0.42) if role == "rbpt" else (0.45, 0.18))
        max_width, max_height = ((0.78, 0.72) if role == "rbpt" else (0.68, 0.38))

        def bounded_axis(low: float, high: float, minimum: float, maximum: float):
            center = 0.5 * (low + high)
            size = min(maximum, max(minimum, high - low))
            low = max(0.0, min(1.0 - size, center - size / 2))
            return low, low + size

        x1, x2 = bounded_axis(x1, x2, min_width, max_width)
        y1, y2 = bounded_axis(y1, y2, min_height, max_height)
        result[role] = (float(x1), float(y1), float(x2), float(y2))
    return result


def _open_image(path: Path | None, image_size: int) -> tuple[Image.Image, bool]:
    if path is None or not path.is_file():
        return Image.new("RGB", (image_size, image_size), (128, 128, 128)), False
    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB"), True
    except (OSError, ValueError):
        # Inference must return a reviewable case instead of aborting the whole
        # batch when one uploaded image is corrupt. Development fitting still
        # rejects unreadable images when the quality reference is frozen.
        return Image.new("RGB", (image_size, image_size), (128, 128, 128)), False


def _crop_normalized(
    image: Image.Image,
    box: tuple[float, float, float, float],
    jitter: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    x1, y1, x2, y2 = box
    if jitter is not None:
        x1, y1, x2, y2 = (
            max(0.0, x1 + jitter[0]),
            max(0.0, y1 + jitter[1]),
            min(1.0, x2 + jitter[2]),
            min(1.0, y2 + jitter[3]),
        )
    if x2 <= x1 + 0.05 or y2 <= y1 + 0.05:
        x1, y1, x2, y2 = box
    width, height = image.size
    return image.crop(
        (
            int(round(x1 * width)),
            int(round(y1 * height)),
            max(1, int(round(x2 * width))),
            max(1, int(round(y2 * height))),
        )
    )


def _transform(
    image: Image.Image,
    image_size: int,
    *,
    flip: bool,
    rotation: float,
    brightness: float,
    contrast: float,
    saturation: float,
) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if flip:
        image = ImageOps.mirror(image)
    if rotation:
        image = image.rotate(rotation, Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).float()


class AIRBPTSATDualViewDataset(Dataset):
    """Five global views plus five leakage-safe high-resolution assay crops."""

    def __init__(
        self,
        records: list[CaseRecord],
        *,
        roi_boxes: dict[str, tuple[float, float, float, float]],
        global_image_size: int = 192,
        roi_image_size: int = 320,
        training: bool = False,
        modality_dropout: float = 0.0,
        sample_weights: dict[str, float] | None = None,
        base_seed: int = 20260715,
    ):
        if not records:
            raise ValueError("dataset requires at least one case")
        self.records = records
        self.roi_boxes = {role: tuple(roi_boxes[role]) for role in ROLE_NAMES}
        self.global_image_size = int(global_image_size)
        self.roi_image_size = int(roi_image_size)
        if self.global_image_size < 96 or self.roi_image_size < self.global_image_size:
            raise ValueError("ROI size must be at least the global size and global size must be >=96")
        self.training = bool(training)
        self.modality_dropout = float(modality_dropout)
        self.sample_weights = dict(sample_weights or {})
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        rng = random.Random(self.base_seed + self.epoch * 1_000_003 + index * 104729)
        flip = self.training and rng.random() < 0.5
        rotation = rng.uniform(-4.0, 4.0) if self.training else 0.0
        brightness = rng.uniform(0.88, 1.12) if self.training else 1.0
        contrast = rng.uniform(0.88, 1.12) if self.training else 1.0
        saturation = rng.uniform(0.92, 1.08) if self.training else 1.0
        global_images, roi_images = [], []
        modality_present, reaction_targets, reaction_valid = [], [], []
        for role_index, (role, image_path, json_path) in enumerate(
            zip(ROLE_NAMES, record.images, record.jsons)
        ):
            image, present = _open_image(image_path, self.global_image_size)
            jitter = None
            if self.training:
                scale = 0.025
                jitter = tuple(rng.uniform(-scale, scale) for _ in range(4))
            roi = _crop_normalized(image, self.roi_boxes[role], jitter)
            kwargs = {
                "flip": flip,
                "rotation": rotation,
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
            }
            global_images.append(_transform(image, self.global_image_size, **kwargs))
            roi_images.append(_transform(roi, self.roi_image_size, **kwargs))
            reaction = reaction_target_from_json(json_path, role_index)
            modality_present.append(present)
            reaction_targets.append(reaction)
            reaction_valid.append(reaction >= 0 and present)

        if self.training and self.modality_dropout > 0 and rng.random() < self.modality_dropout:
            candidates = [idx for idx, present in enumerate(modality_present) if present]
            if len(candidates) > 1:
                dropped = rng.choice(candidates)
                global_images[dropped].zero_()
                roi_images[dropped].zero_()
                modality_present[dropped] = False
                reaction_valid[dropped] = False

        label = -1 if record.label is None else int(record.label)
        return {
            "sample_id": record.sample_id,
            "rbpt_global": global_images[0],
            "rbpt_roi": roi_images[0],
            "sat_global": torch.stack(global_images[1:]),
            "sat_roi": torch.stack(roi_images[1:]),
            "modality_present": torch.tensor(modality_present, dtype=torch.bool),
            "reaction_targets": torch.tensor(reaction_targets, dtype=torch.long),
            "reaction_valid": torch.tensor(reaction_valid, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "center": torch.tensor(record.center_index, dtype=torch.long),
            "sample_weight": torch.tensor(
                float(self.sample_weights.get(record.sample_id, 1.0)), dtype=torch.float32
            ),
        }
