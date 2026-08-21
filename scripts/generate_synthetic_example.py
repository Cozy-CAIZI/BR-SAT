#!/usr/bin/env python3
"""Generate a non-clinical five-image example and integrity-checked manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic"
ROLES = ("rbpt", "sat25", "sat50", "sat100", "sat200")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    image_dir = EXAMPLE / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "rbpt": (181, 92, 112),
        "sat25": (82, 137, 157),
        "sat50": (91, 151, 146),
        "sat100": (118, 155, 120),
        "sat200": (153, 147, 105),
    }
    row = {"sample_id": "SYNTHETIC-001"}
    for index, role in enumerate(ROLES):
        image = Image.new("RGB", (512, 512), (244, 242, 235))
        draw = ImageDraw.Draw(image)
        color = palette[role]
        draw.rounded_rectangle((72, 72, 440, 440), radius=36, outline=color, width=16)
        draw.ellipse((145, 145 + index * 8, 367, 367 - index * 8), fill=color)
        draw.text((22, 22), f"SYNTHETIC {role.upper()}", fill=(30, 38, 42))
        path = image_dir / f"SYNTHETIC-001_{role}.png"
        image.save(path, format="PNG")
        row[f"{role}_image"] = str(path.relative_to(EXAMPLE))
        row[f"{role}_sha256"] = sha256(path)
    fields = ["sample_id"]
    for role in ROLES:
        fields.extend([f"{role}_image", f"{role}_sha256"])
    with (EXAMPLE / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    print(EXAMPLE / "manifest.csv")


if __name__ == "__main__":
    main()
