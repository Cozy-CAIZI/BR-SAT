#!/usr/bin/env python3
"""Verify and, only when explicitly requested, stage the 15 frozen weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "frozen_ensemble_v1.0.0.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_checkpoint(value: str) -> Path:
    checkpoint = Path(value)
    if checkpoint.is_absolute() or ".." in checkpoint.parts:
        raise ValueError(f"unsafe checkpoint path in manifest: {value!r}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--copy", action="store_true", help="copy verified files into repository models/; otherwise verify only")
    parser.add_argument("--write-report", type=Path, help="write a path-free JSON verification report")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8-sig"))
    members = config.get("members", [])
    if len(members) != 15:
        raise SystemExit(f"expected 15 ensemble members, found {len(members)}")

    verified: list[tuple[Path, Path, str, int]] = []
    for member in members:
        relative = safe_relative_checkpoint(member["checkpoint"])
        source = source_root / relative
        if not source.is_file():
            raise SystemExit(f"missing source weight: {relative}")
        expected_hash = member["sha256"]
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"source weight hash mismatch for {relative}: expected {expected_hash}, found {actual_hash}"
            )
        verified.append((relative, source, expected_hash, source.stat().st_size))

    if args.copy:
        for relative, source, expected_hash, _ in verified:
            destination = ROOT / "models" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != expected_hash:
                raise SystemExit(f"refusing to overwrite non-matching destination weight: models/{relative}")
            if not destination.exists():
                shutil.copy2(source, destination)
            if sha256(destination) != expected_hash:
                raise SystemExit(f"destination verification failed: models/{relative}")

    total_bytes = sum(item[3] for item in verified)
    action = "verified and staged" if args.copy else "verified without copying"
    if args.write_report:
        report = {
            "status": "PASS",
            "verification_scope": "controlled-source weights; read-only hash and size verification",
            "member_count": len(verified),
            "total_bytes": total_bytes,
            "largest_file_bytes": max(item[3] for item in verified),
            "github_object_limit_bytes": 100 * 1024 * 1024,
            "all_files_below_github_object_limit": all(item[3] <= 100 * 1024 * 1024 for item in verified),
            "members": [
                {"checkpoint": str(relative), "bytes": size, "sha256": expected_hash}
                for relative, _, expected_hash, size in verified
            ],
        }
        destination = args.write_report.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{action}: {len(verified)} frozen weights, {total_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
