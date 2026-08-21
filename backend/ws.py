"""WebSocket endpoint for real-time training metrics streaming."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.paths import get_project_root

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent / "models-at-home")
RUNS_DIR = PROJECT_ROOT / ".runs"


@router.websocket("/ws/metrics/{run_id}")
async def metrics_ws(websocket: WebSocket, run_id: str):
    """Stream training metrics from .runs/<run_id>/metrics.json via WebSocket."""
    await websocket.accept()

    metrics_path = RUNS_DIR / run_id / "metrics.json"
    last_mtime = 0.0
    last_step = -1

    try:
        while True:
            if metrics_path.exists():
                mtime = metrics_path.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    try:
                        data = _normalize_metrics(json.loads(metrics_path.read_text()))
                        step = data.get("step", 0)
                        if step != last_step:
                            last_step = step
                            await websocket.send_json(data)
                    except (json.JSONDecodeError, IOError):
                        pass
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()


def _normalize_metrics(data: dict) -> dict:
    """Normalize legacy worker metrics for the React client."""
    if "step" not in data and "current_step" in data:
        data["step"] = data.get("current_step", 0)
    if "loss" not in data and "current_loss" in data:
        data["loss"] = data.get("current_loss", 0)
    if "learning_rate" not in data and "current_lr" in data:
        data["learning_rate"] = data.get("current_lr", 0)
    if "timestamp" not in data:
        data["timestamp"] = data.get("elapsed_seconds", 0)
    return data
