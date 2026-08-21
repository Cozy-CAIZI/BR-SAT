from __future__ import annotations

import base64
import collections
import csv
import hashlib
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
from PIL import ImageOps
from torch.utils.data import Dataset

from multimodal_model_nvidia import RBPT_MASK_NAMES
from multimodal_model_nvidia import SAT_MASK_NAMES


DIAGNOSIS_TO_ID = {"阴性": 0, "弱阳性": 1, "阳性": 2}
ROLE_NAMES = ("rbpt", "sat25", "sat50", "sat100", "sat200")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)


@dataclass(frozen=True)
class CaseRecord:
    sample_id: str
    label: int | None
    diagnosis_class: str
    center_name: str
    center_index: int
    images: tuple[Path | None, ...]
    jsons: tuple[Path | None, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(bundle_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else bundle_root / path


def load_development_records(manifest_path: str | Path) -> tuple[list[CaseRecord], dict[str, int]]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    bundle_root = manifest_path.parent.parent
    raw_rows = _read_csv(manifest_path)
    eligible = [
        row for row in raw_rows
        if row.get("five_image_complete", "").lower() == "yes"
        and row.get("exclude_from_default_training", "").upper() != "YES"
    ]
    center_names = sorted({row.get("center_id") or row.get("hospital") or "unknown" for row in eligible})
    center_to_index = {name: index for index, name in enumerate(center_names)}
    records = []
    for row in eligible:
        diagnosis = row["diagnosis_class"].strip()
        if diagnosis not in DIAGNOSIS_TO_ID:
            raise ValueError(f"unsupported diagnosis_class {diagnosis!r} for {row['sample_id']}")
        center_name = row.get("center_id") or row.get("hospital") or "unknown"
        images = tuple(_resolve(bundle_root, row.get(f"{role}_image")) for role in ROLE_NAMES)
        jsons = tuple(_resolve(bundle_root, row.get(f"{role}_json")) for role in ROLE_NAMES)
        if any(path is None or not path.is_file() for path in images):
            raise FileNotFoundError(f"eligible five-image case has a missing file: {row['sample_id']}")
        records.append(
            CaseRecord(
                sample_id=row["sample_id"],
                label=DIAGNOSIS_TO_ID[diagnosis],
                diagnosis_class=diagnosis,
                center_name=center_name,
                center_index=center_to_index[center_name],
                images=images,
                jsons=jsons,
            )
        )
    if not records:
        raise ValueError(f"no eligible development cases in {manifest_path}")
    return records, center_to_index


def load_inference_records(manifest_path: str | Path) -> list[CaseRecord]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    bundle_root = manifest_path.parent.parent
    rows = _read_csv(manifest_path)
    records = []
    for row in rows:
        images = tuple(_resolve(bundle_root, row.get(f"{role}_image")) for role in ROLE_NAMES)
        records.append(
            CaseRecord(
                sample_id=row["sample_id"],
                label=None,
                diagnosis_class="",
                center_name="unknown",
                center_index=0,
                images=images,
                jsons=(None, None, None, None, None),
            )
        )
    if not records:
        raise ValueError(f"empty inference manifest: {manifest_path}")
    return records


def stratified_patient_split(
    records: list[CaseRecord],
    *,
    val_fraction: float = 0.20,
    seed: int = 20260714,
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    """Class-stratified split with center round-robin selection.

    Every case stays intact. Internal and external validation manifests are never
    read by this function.
    """
    if not 0.05 <= val_fraction <= 0.40:
        raise ValueError("val_fraction must be between 0.05 and 0.40")
    by_class: dict[int, list[CaseRecord]] = collections.defaultdict(list)
    for record in records:
        if record.label is None:
            raise ValueError("training split received an unlabeled record")
        by_class[record.label].append(record)
    val_ids: set[str] = set()
    for label, class_records in sorted(by_class.items()):
        target = max(1, min(len(class_records) - 1, round(len(class_records) * val_fraction)))
        by_center: dict[str, list[CaseRecord]] = collections.defaultdict(list)
        for record in class_records:
            by_center[record.center_name].append(record)
        rng = random.Random(seed + label * 1009)
        center_queues = []
        for center_name in sorted(by_center):
            queue = sorted(by_center[center_name], key=lambda record: record.sample_id)
            rng.shuffle(queue)
            center_queues.append(queue)
        rng.shuffle(center_queues)
        selected_for_class = 0
        while selected_for_class < target:
            progressed = False
            for queue in center_queues:
                if queue and selected_for_class < target:
                    val_ids.add(queue.pop().sample_id)
                    selected_for_class += 1
                    progressed = True
            if not progressed:
                break
    train = [record for record in records if record.sample_id not in val_ids]
    val = [record for record in records if record.sample_id in val_ids]
    for label in sorted(by_class):
        if not any(record.label == label for record in train) or not any(record.label == label for record in val):
            raise ValueError(f"class {label} missing from train or tuning split")
    return train, val


def write_split_manifest(path: str | Path, train: list[CaseRecord], val: list[CaseRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_by_id = {record.sample_id: "train" for record in train}
    split_by_id.update({record.sample_id: "tuning" for record in val})
    records = sorted(train + val, key=lambda record: record.sample_id)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "split", "diagnosis_class", "center_id"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "split": split_by_id[record.sample_id],
                    "diagnosis_class": record.diagnosis_class,
                    "center_id": record.center_name,
                }
            )


def _scaled_points(points, width: float, height: float, image_size: int):
    return [
        (
            min(image_size - 1, max(0, float(x) / max(1.0, width) * image_size)),
            min(image_size - 1, max(0, float(y) / max(1.0, height) * image_size)),
        )
        for x, y in points
    ]


def _draw_shape(layer: Image.Image, shape: dict, width: int, height: int, image_size: int) -> None:
    points = _scaled_points(shape.get("points", []), width, height, image_size)
    kind = shape.get("shape_type", "polygon")
    draw = ImageDraw.Draw(layer)
    if kind == "polygon" and len(points) >= 3:
        draw.polygon(points, fill=255)
    elif kind == "rectangle" and len(points) == 2:
        draw.rectangle((points[0][0], points[0][1], points[1][0], points[1][1]), fill=255)
    elif kind == "circle" and len(points) == 2:
        cx, cy = points[0]
        px, py = points[1]
        radius = math.hypot(px - cx, py - cy)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    elif kind == "mask" and len(points) == 2 and shape.get("mask"):
        decoded = base64.b64decode(shape["mask"])
        mask = Image.open(io.BytesIO(decoded)).convert("L")
        x1, y1 = points[0]
        x2, y2 = points[1]
        left, right = sorted((round(x1), round(x2)))
        top, bottom = sorted((round(y1), round(y2)))
        target_width = max(1, right - left + 1)
        target_height = max(1, bottom - top + 1)
        mask = mask.resize((target_width, target_height), Image.Resampling.NEAREST)
        thresholded = mask.point(lambda value: 255 if value > 0 else 0)
        layer.paste(Image.new("L", thresholded.size, 255), (left, top), thresholded)


def _mask_cache_key(json_path: Path, labels: tuple[str, ...], image_size: int) -> str:
    stat = json_path.stat()
    value = f"{json_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{labels}|{image_size}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def reaction_target_from_json(
    json_path: Path | None,
    role_index: int,
    payload: dict | None = None,
) -> int:
    """Return 0=no agglutination, 1=weak, 2=strong, -1=unknown.

    Each modality is interpreted independently. In particular, no SAT result is
    inferred from RBPT and no SAT dilution is inferred from another dilution.
    """
    if json_path is None or not json_path.is_file():
        return -1
    if payload is None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    explicit_status = str(
        payload.get("reaction_supervision")
        or (payload.get("annotation_task") or {}).get("reaction_status")
        or ""
    ).strip().lower()
    explicit_targets = {
        "no_agglutination": 0,
        "weak": 1,
        "strong": 2,
        "unknown": -1,
    }
    if explicit_status:
        if explicit_status not in explicit_targets:
            raise ValueError(
                f"unsupported reaction_supervision {explicit_status!r} in {json_path}"
            )
        # Expert image-level supervision may be categorical even when no lesion
        # polygon can be drawn reliably. Unknown is always ignored by the
        # auxiliary loss, even if a broad assay-zone polygon is present.
        return explicit_targets[explicit_status]
    labels = {str(shape.get("label", "")).strip() for shape in payload.get("shapes", [])}
    if role_index == 0:
        weak_label = "rbt_agglutinate_weak"
        strong_label = "rbt_agglutinate_strong"
        zone_label = "rbt_reaction_zone"
    else:
        weak_label = "sat_agglutinate_weak"
        strong_label = "sat_agglutinate_strong"
        zone_label = "sat_bottom_zone"
    has_weak = weak_label in labels
    has_strong = strong_label in labels
    if has_weak and has_strong:
        raise ValueError(f"weak and strong reaction labels coexist in {json_path}")
    if has_strong:
        return 2
    if has_weak:
        return 1
    flags = payload.get("flags") or {}
    explicit_no_agglutination = flags.get("no_agglutination") is True
    if explicit_no_agglutination or zone_label in labels:
        return 0
    # Unflagged empty JSON means unfinished/unknown, not a negative reaction.
    return -1


def render_multilabel_mask(
    json_path: Path | None,
    labels: tuple[str, ...],
    image_size: int,
    cache_dir: Path | None,
    role_index: int,
) -> tuple[np.ndarray, bool, int]:
    if json_path is None or not json_path.is_file():
        return np.zeros((len(labels), image_size, image_size), dtype=np.uint8), False, -1
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_mask_cache_key(json_path, labels, image_size)}.npz"
        if cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=False)
            reaction = int(cached["reaction"]) if "reaction" in cached.files else reaction_target_from_json(json_path, role_index)
            return cached["mask"], True, reaction
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    reaction = reaction_target_from_json(json_path, role_index, payload)
    width = int(payload.get("imageWidth") or image_size)
    height = int(payload.get("imageHeight") or image_size)
    label_to_index = {label: index for index, label in enumerate(labels)}
    layers = [Image.new("L", (image_size, image_size), 0) for _ in labels]
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label in label_to_index:
            _draw_shape(layers[label_to_index[label]], shape, width, height, image_size)
    mask = np.stack([(np.asarray(layer, dtype=np.uint8) > 0).astype(np.uint8) for layer in layers])
    if cache_path is not None:
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, mask=mask, reaction=np.asarray(reaction, dtype=np.int8))
            os.replace(temporary, cache_path)
        except OSError:
            temporary.unlink(missing_ok=True)
    return mask, True, reaction


def _load_image(path: Path | None, image_size: int) -> tuple[Image.Image, bool]:
    if path is None or not path.is_file():
        return Image.new("RGB", (image_size, image_size), (128, 128, 128)), False
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR), True


def _transform_image(
    image: Image.Image,
    *,
    horizontal_flip: bool,
    rotation: float,
    brightness: float,
    contrast: float,
    saturation: float,
) -> torch.Tensor:
    if horizontal_flip:
        image = ImageOps.mirror(image)
    if rotation:
        image = image.rotate(rotation, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).float()


def _transform_mask(mask: np.ndarray, horizontal_flip: bool, rotation: float) -> torch.Tensor:
    transformed = []
    for channel in mask:
        image = Image.fromarray(channel * 255, mode="L")
        if horizontal_flip:
            image = ImageOps.mirror(image)
        if rotation:
            image = image.rotate(rotation, resample=Image.Resampling.NEAREST, fillcolor=0)
        transformed.append(torch.from_numpy((np.asarray(image, dtype=np.uint8) > 0).astype(np.float32)))
    return torch.stack(transformed)


class AIRBPTSATDataset(Dataset):
    def __init__(
        self,
        records: list[CaseRecord],
        *,
        image_size: int = 224,
        training: bool = False,
        mask_cache_dir: str | Path | None = None,
        modality_dropout: float = 0.0,
    ):
        if not records:
            raise ValueError("dataset requires at least one case")
        self.records = records
        self.image_size = int(image_size)
        self.training = bool(training)
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else None
        self.modality_dropout = float(modality_dropout)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        rng = random.Random(torch.initial_seed() + index * 104729)
        flip = self.training and rng.random() < 0.50
        rotation = rng.uniform(-5.0, 5.0) if self.training else 0.0
        brightness = rng.uniform(0.85, 1.15) if self.training else 1.0
        contrast = rng.uniform(0.85, 1.15) if self.training else 1.0
        saturation = rng.uniform(0.90, 1.10) if self.training else 1.0
        image_tensors = []
        modality_present = []
        mask_valid = []
        reaction_targets = []
        reaction_valid = []
        masks = []
        for role_index, (image_path, json_path) in enumerate(zip(record.images, record.jsons)):
            image, present = _load_image(image_path, self.image_size)
            image_tensors.append(
                _transform_image(
                    image,
                    horizontal_flip=flip,
                    rotation=rotation,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                )
            )
            labels = RBPT_MASK_NAMES if role_index == 0 else SAT_MASK_NAMES
            mask, valid, reaction_target = render_multilabel_mask(
                json_path,
                labels,
                self.image_size,
                self.mask_cache_dir,
                role_index,
            )
            masks.append(_transform_mask(mask, flip, rotation))
            modality_present.append(present)
            mask_valid.append(valid and present)
            reaction_targets.append(reaction_target)
            reaction_valid.append(reaction_target >= 0 and present)

        if self.training and self.modality_dropout > 0 and rng.random() < self.modality_dropout:
            candidates = [idx for idx, present in enumerate(modality_present) if present]
            if len(candidates) > 1:
                dropped = rng.choice(candidates)
                image_tensors[dropped].zero_()
                modality_present[dropped] = False
                mask_valid[dropped] = False
                reaction_valid[dropped] = False

        rbpt_mask = masks[0]
        sat_masks = torch.stack(masks[1:])
        label = -1 if record.label is None else record.label
        return {
            "sample_id": record.sample_id,
            "rbpt_image": image_tensors[0],
            "sat_images": torch.stack(image_tensors[1:]),
            "modality_present": torch.tensor(modality_present, dtype=torch.bool),
            "rbpt_mask": rbpt_mask,
            "sat_masks": sat_masks,
            "mask_valid": torch.tensor(mask_valid, dtype=torch.bool),
            "reaction_targets": torch.tensor(reaction_targets, dtype=torch.long),
            "reaction_valid": torch.tensor(reaction_valid, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "center": torch.tensor(record.center_index, dtype=torch.long),
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
