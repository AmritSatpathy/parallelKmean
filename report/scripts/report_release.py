"""Shared registry and lineage helpers for versioned report releases."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def load_registry(root: Path) -> tuple[Path | None, dict]:
    value = os.environ.get("REPORT_RELEASE_REGISTRY")
    if not value:
        return None, {}
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError(f"Unsupported report release registry: {path}")
    return path, registry


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_id(registry: dict, name: str, default: str) -> str:
    return str(registry.get("ids", {}).get(name, default))


def registry_output(root: Path, registry: dict, name: str, default: str) -> Path:
    return resolve(root, registry.get("outputs", {}).get(name, default))
