"""Shared path helpers for native and Docker backends."""
from __future__ import annotations

import os
from pathlib import Path


def get_project_root(default: Path) -> Path:
    """Return the models-at-home root, overridable inside Docker."""
    override = os.environ.get("MODELS_AT_HOME_ROOT")
    return Path(override).resolve() if override else default.resolve()
