#!/usr/bin/env python3
"""Audit the BR-SAT public repository boundary and produce a file manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".md", ".py", ".toml", ".txt", ".json", ".csv", ".yml", ".yaml",
    ".cff", ".xml", ".ini", ".cfg", ".sh", ".ps1", ".bat", ".gitignore",
    ".gitattributes",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"}
WEIGHT_SUFFIXES = {".pth", ".pt", ".ckpt", ".onnx", ".safetensors"}
PROHIBITED_SUFFIXES = {".xlsx", ".xls", ".docx", ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".dcm"}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
PATTERNS = {
    "local_macos_path": re.compile(r"/Users/[^/\s]+/"),
    "local_windows_path": re.compile(r"[A-Za-z]:\\+"),
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "credential_assignment": re.compile(r"(?i)(api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"][^'\"]{8,}"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & IGNORED_PARTS)
    )


def text_findings(text: str) -> list[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]


def inspect_git_history() -> tuple[list[dict], list[dict]]:
    """Inspect objects reachable from refs; unreachable local blobs are not publishable history."""
    findings: list[dict] = []
    metadata_findings: list[dict] = []
    if not (ROOT / ".git").is_dir():
        return findings, metadata_findings

    objects = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for record in objects:
        fields = record.split(" ", 1)
        if len(fields) != 2:
            continue
        object_id, object_path = fields
        suffix = Path(object_path).suffix.lower()
        object_type = subprocess.run(
            ["git", "cat-file", "-t", object_id],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if object_type != "blob":
            continue
        size = int(subprocess.run(
            ["git", "cat-file", "-s", object_id],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout)
        if size > 100 * 1024 * 1024:
            findings.append({"severity": "block", "path": object_path, "finding": "historical blob exceeds GitHub 100 MB object limit"})
        if suffix in IMAGE_SUFFIXES and tuple(Path(object_path).parts[:3]) != ("examples", "synthetic", "images"):
            findings.append({"severity": "block", "path": object_path, "finding": "historical image outside the synthetic example boundary"})
        if suffix in PROHIBITED_SUFFIXES:
            findings.append({"severity": "block", "path": object_path, "finding": "prohibited historical clinical/document/archive type"})
        if (suffix in TEXT_SUFFIXES or Path(object_path).name in {".gitignore", ".gitattributes"}) and object_path != "scripts/audit_release.py":
            payload = subprocess.run(
                ["git", "cat-file", "blob", object_id],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout.decode("utf-8-sig", errors="replace")
            for name in text_findings(payload):
                findings.append({"severity": "block", "path": object_path, "finding": f"historical_{name}"})

    identities = subprocess.run(
        ["git", "log", "--all", "--format=%H%x09%an%x09%ae%x09%cn%x09%ce"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for row in identities:
        commit, author_name, author_email, committer_name, committer_email = row.split("\t")
        for role, name, email in (
            ("author", author_name, author_email),
            ("committer", committer_name, committer_email),
        ):
            if email and not email.lower().endswith("@users.noreply.github.com"):
                metadata_findings.append({
                    "severity": "review",
                    "commit": commit,
                    "role": role,
                    "name": name,
                    "finding": "public commit metadata contains a non-noreply email address",
                })
    return findings, metadata_findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--expect-final", action="store_true")
    parser.add_argument("--audit-git-history", action="store_true")
    args = parser.parse_args()
    all_files = files()
    findings: list[dict] = []
    manifest = []
    for path in all_files:
        relative = path.relative_to(ROOT)
        if args.write_manifest and path.resolve() == args.write_manifest.resolve():
            continue
        if path.is_symlink():
            findings.append({"severity": "block", "path": str(relative), "finding": "symbolic links are prohibited in the public release"})
            continue
        manifest.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": sha256(path)})
        if path.stat().st_size > 100 * 1024 * 1024:
            findings.append({"severity": "block", "path": str(relative), "finding": "file exceeds GitHub 100 MB object limit"})
        if path.suffix.lower() in IMAGE_SUFFIXES and relative.parts[:3] != ("examples", "synthetic", "images"):
            findings.append({"severity": "block", "path": str(relative), "finding": "image outside the synthetic example boundary"})
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            findings.append({"severity": "block", "path": str(relative), "finding": "clinical/document/archive type is prohibited"})
        if (path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore") and relative != Path("scripts/audit_release.py"):
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for name in text_findings(text):
                findings.append({"severity": "block", "path": str(relative), "finding": name})

    config_path = ROOT / "config" / "frozen_ensemble_v1.0.0.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    members = config.get("members", [])
    if len(members) != 15 or len({member["sha256"] for member in members}) != 15:
        findings.append({"severity": "block", "path": str(config_path.relative_to(ROOT)), "finding": "ensemble must contain 15 unique hashes"})
    expected_weights = {Path("models") / member["checkpoint"]: member["sha256"] for member in members}
    if len(expected_weights) != len(members):
        findings.append({"severity": "block", "path": str(config_path.relative_to(ROOT)), "finding": "ensemble checkpoint paths must be unique"})

    synthetic_manifest = ROOT / "examples" / "synthetic" / "manifest.csv"
    with synthetic_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        synthetic_rows = list(csv.DictReader(handle))
    if len(synthetic_rows) != 1:
        findings.append({"severity": "block", "path": str(synthetic_manifest.relative_to(ROOT)), "finding": "expected one synthetic example"})
    for row in synthetic_rows:
        for role in ("rbpt", "sat25", "sat50", "sat100", "sat200"):
            image = synthetic_manifest.parent / row[f"{role}_image"]
            if not image.is_file() or sha256(image) != row[f"{role}_sha256"]:
                findings.append({"severity": "block", "path": str(image.relative_to(ROOT)), "finding": "synthetic image missing or hash mismatch"})

    weight_files = [path for path in all_files if path.suffix.lower() in WEIGHT_SUFFIXES]
    actual_weight_paths = {path.relative_to(ROOT) for path in weight_files}
    for relative, expected_hash in sorted(expected_weights.items(), key=lambda item: str(item[0])):
        path = ROOT / relative
        if not path.is_file():
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            findings.append({"severity": "block", "path": str(relative), "finding": f"weight SHA-256 mismatch: expected {expected_hash}, found {actual_hash}"})
    for relative in sorted(actual_weight_paths - set(expected_weights), key=str):
        findings.append({"severity": "block", "path": str(relative), "finding": "unexpected weight file not declared in frozen ensemble manifest"})

    policy_path = ROOT / "config" / "release_policy_v1.0.0.json"
    if not policy_path.is_file():
        findings.append({"severity": "block", "path": str(policy_path.relative_to(ROOT)), "finding": "release policy is absent"})
        policy = {}
    else:
        policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    public_weights = policy.get("public_model_weights")
    if public_weights not in {True, False}:
        findings.append({"severity": "block", "path": str(policy_path.relative_to(ROOT)), "finding": "public_model_weights must be true or false"})
    if public_weights is False and weight_files:
        findings.append({"severity": "block", "path": "models", "finding": "release policy prohibits model weights in the public repository"})

    history_findings: list[dict] = []
    repository_metadata_findings: list[dict] = []
    if args.audit_git_history:
        history_findings, repository_metadata_findings = inspect_git_history()
        findings.extend(history_findings)

    release_blockers = []
    if not (ROOT / "LICENSE").is_file():
        release_blockers.append("institution-approved LICENSE is absent")
    if not (ROOT / "CITATION.cff").is_file():
        release_blockers.append("verified creator metadata and CITATION.cff are absent")
    if public_weights is True and len(weight_files) != 15:
        release_blockers.append(f"archived frozen weights are incomplete: found {len(weight_files)}/15")
    if public_weights is False and not policy.get("model_weight_restriction_rationale"):
        release_blockers.append("model-weight non-public rationale is not documented")
    if public_weights is False and not policy.get("model_weight_access_conditions"):
        release_blockers.append("model-weight access conditions or request route are not documented")
    if args.expect_final and release_blockers:
        findings.extend({"severity": "block", "path": ".", "finding": item} for item in release_blockers)

    report = {
        "status": "FAIL" if any(item["severity"] == "block" for item in findings) else ("candidate_pass_with_release_blockers" if release_blockers else "PASS"),
        "file_count": len(manifest),
        "manifest": manifest,
        "privacy_and_secret_findings": findings,
        "ensemble_members": len(members),
        "public_model_weights": public_weights,
        "weight_files_in_repository": len(weight_files),
        "weight_files_matching_manifest": sum(
            1 for relative, expected_hash in expected_weights.items()
            if (ROOT / relative).is_file() and sha256(ROOT / relative) == expected_hash
        ),
        "git_history_audited": args.audit_git_history,
        "git_history_findings": history_findings,
        "repository_metadata_findings": repository_metadata_findings,
        "release_blockers": release_blockers,
    }
    if args.write_manifest:
        destination = args.write_manifest.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "file_count", "privacy_and_secret_findings", "ensemble_members", "public_model_weights",
        "weight_files_in_repository", "weight_files_matching_manifest",
        "git_history_audited", "git_history_findings", "repository_metadata_findings",
        "release_blockers",
    )}, indent=2))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
