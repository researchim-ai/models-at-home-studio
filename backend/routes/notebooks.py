"""Notebook endpoints: list local .ipynb, expose templates, check JupyterLab."""
from __future__ import annotations

import urllib.request
import urllib.error
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from backend.paths import get_project_root

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
JUPYTER_PID = PROJECT_ROOT / ".runs" / "jupyter.pid"
JUPYTER_LOG = PROJECT_ROOT / ".runs" / "jupyter.log"
JUPYTER_TOKEN = os.environ.get("JUPYTER_TOKEN", "mah-local")
JUPYTER_PORT = int(os.environ.get("JUPYTER_PORT", "8888"))
JUPYTER_PUBLIC_URL = os.environ.get("JUPYTER_PUBLIC_URL", f"http://127.0.0.1:{JUPYTER_PORT}")

# Curated templates that ship with the project
TEMPLATES = [
    {
        "name": "Inference Playground",
        "description": "Интерактивный inference: попробовать любую модель",
        "file": "inference_playground.ipynb",
    },
    {
        "name": "Dataset Download",
        "description": "Скачать датасет с HuggingFace и положить в datasets/",
        "file": "dataset_download.ipynb",
    },
    {
        "name": "SFT Playground",
        "description": "Fine-tune модели на chat-данных",
        "file": "sft_playground.ipynb",
    },
    {
        "name": "GRPO Mini Demo",
        "description": "RL-обучение через GRPO на небольшом примере",
        "file": "grpo_mini_demo.ipynb",
    },
    {
        "name": "Memory Estimator Demo",
        "description": "Оценка VRAM для разных архитектур",
        "file": "memory_estimator_demo.ipynb",
    },
    {
        "name": "Blueprint from JSON",
        "description": "Собрать модель из blueprint JSON",
        "file": "blueprint_from_json.ipynb",
    },
    {
        "name": "Custom Block Compare",
        "description": "Сравнение кастомных блоков (норм, активаций, MoE)",
        "file": "custom_block_compare.ipynb",
    },
    {
        "name": "MoE Playground",
        "description": "Эксперименты с Mixture of Experts",
        "file": "moe_playground.ipynb",
    },
    {
        "name": "Llama Default Train",
        "description": "Базовая тренировка Llama-архитектуры",
        "file": "llama_default_train.ipynb",
    },
    {
        "name": "Llama Module Playground",
        "description": "Экспериментировать с отдельными модулями Llama",
        "file": "llama_module_playground.ipynb",
    },
    {
        "name": "Optimizers Compare",
        "description": "Сравнение AdamW / SGD / Lion / ...",
        "file": "optimizers_compare.ipynb",
    },
    {
        "name": "Data Formats (SFT/RL)",
        "description": "Форматы данных для SFT и RL тренировки",
        "file": "data_formats_sft_rl.ipynb",
    },
]


@router.get("/list")
async def list_notebooks():
    """List .ipynb files in the notebooks directory."""
    items = []
    if not NOTEBOOKS_DIR.exists():
        return {"notebooks": []}

    for f in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        try:
            stat = f.stat()
            items.append({
                "name": f.name,
                "path": f.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except Exception:
            continue

    return {"notebooks": items}


@router.get("/templates")
async def notebook_templates():
    """Expose curated templates. Only returns those that exist on disk."""
    available = []
    for tmpl in TEMPLATES:
        if (NOTEBOOKS_DIR / tmpl["file"]).exists():
            available.append(tmpl)
    return {"templates": available}


@router.get("/jupyter-status")
async def jupyter_status(url: str = Query(...)):
    """Check if JupyterLab is running by hitting /api/status."""
    try:
        endpoint = f"{_probe_url(url).rstrip('/')}/api/status"
        req = urllib.request.Request(endpoint, headers={"User-Agent": "mah-studio"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return {"running": resp.status == 200, "url": url}
    except urllib.error.HTTPError as e:
        return {"running": e.code in (401, 403), "url": url}
    except Exception:
        return {"running": False, "url": url}


@router.post("/jupyter-start")
async def jupyter_start():
    """Start a managed JupyterLab process rooted at models-at-home/notebooks."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    JUPYTER_PID.parent.mkdir(parents=True, exist_ok=True)

    if _pid_running():
        return {
            "running": True,
            "url": JUPYTER_PUBLIC_URL,
            "token": JUPYTER_TOKEN,
            "root_dir": str(NOTEBOOKS_DIR),
        }

    cmd = [
        "python",
        "-m",
        "jupyterlab",
        "--no-browser",
        "--ip=0.0.0.0",
        f"--port={JUPYTER_PORT}",
        f"--ServerApp.token={JUPYTER_TOKEN}",
        f"--ServerApp.root_dir={NOTEBOOKS_DIR}",
        "--ServerApp.allow_origin=*",
        "--ServerApp.allow_remote_access=True",
        "--allow-root",
    ]

    try:
        log = open(JUPYTER_LOG, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=log,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        JUPYTER_PID.write_text(str(proc.pid))
    except Exception as exc:
        raise HTTPException(500, f"Failed to start JupyterLab: {exc}")

    if not _wait_for_jupyter_ready("http://127.0.0.1:8888", timeout=30):
        tail = ""
        if JUPYTER_LOG.exists():
            tail = "\n".join(JUPYTER_LOG.read_text(errors="ignore").splitlines()[-20:])
        raise HTTPException(500, f"JupyterLab did not become ready. Log tail:\n{tail}")

    return {
        "running": True,
        "url": JUPYTER_PUBLIC_URL,
        "token": JUPYTER_TOKEN,
        "root_dir": str(NOTEBOOKS_DIR),
    }


@router.post("/jupyter-stop")
async def jupyter_stop():
    """Stop the managed JupyterLab process if it was started by the backend."""
    if JUPYTER_PID.exists():
        try:
            pid = int(JUPYTER_PID.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:
            raise HTTPException(500, f"Failed to stop JupyterLab: {exc}")
        finally:
            JUPYTER_PID.unlink(missing_ok=True)
    return {"success": True}


def _pid_running() -> bool:
    if not JUPYTER_PID.exists():
        return False
    try:
        pid = int(JUPYTER_PID.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        JUPYTER_PID.unlink(missing_ok=True)
        return False


def _wait_for_jupyter_ready(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            endpoint = f"{url.rstrip('/')}/api/status"
            req = urllib.request.Request(endpoint, headers={"User-Agent": "mah-studio"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _probe_url(url: str) -> str:
    """Use the container-internal URL when checking the managed Jupyter server."""
    if url.rstrip("/") == JUPYTER_PUBLIC_URL.rstrip("/"):
        return "http://127.0.0.1:8888"
    return url
