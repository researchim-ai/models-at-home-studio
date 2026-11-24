"""Shared path helpers for native and Docker backends."""
from __future__ import annotations

import os
import re
import json
import tempfile
from pathlib import Path


def add_runtime_path():
    """Resolve code separately from the user's mutable workspace, including workers."""
    import sys
    studio=Path(__file__).resolve().parents[1]
    explicit=os.environ.get('MAH_RUNTIME_ROOT')
    candidates=[Path(explicit).expanduser()] if explicit else [studio/'models-at-home',studio.parent/'models-at-home']
    for root in candidates:
        if (root/'homellm').is_dir():
            value=str(root.resolve())
            if value not in sys.path:sys.path.insert(0,value)
            return root
    if explicit:raise ValueError('MAH_RUNTIME_ROOT must contain the homellm package')
    return None


def get_project_root(default: Path) -> Path:
    """Return the models-at-home root, overridable inside Docker."""
    override = os.environ.get("MODELS_AT_HOME_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    sibling = default.parent.parent / "models-at-home"
    return (sibling if not default.exists() and (sibling / "homellm").is_dir() else default).resolve()


def safe_child(root: Path, value: str) -> Path:
    """Resolve a relative child, including symlinks, without allowing the root itself."""
    try:
        if not value or Path(value).is_absolute() or "\\" in value:
            raise ValueError
        path = (root / value).resolve()
        if path == root.resolve() or not path.is_relative_to(root.resolve()):
            raise ValueError
        return path
    except (ValueError, OSError, RuntimeError):
        from fastapi import HTTPException
        raise HTTPException(400, "Invalid workspace path")


def safe_id(root: Path, value: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        from fastapi import HTTPException
        raise HTTPException(400, "Invalid identifier")
    return safe_child(root, value)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
