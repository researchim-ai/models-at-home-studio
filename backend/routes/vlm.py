"""VLM (Visual Language Model) training + inference endpoints."""
from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.paths import get_project_root

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
RUNS_DIR = PROJECT_ROOT / ".runs"
OUTPUT_DIR = PROJECT_ROOT / "out"

_vlm_state: dict = {"model": None, "processor": None, "path": None}


class VLMTrainingRequest(BaseModel):
    stage: str = "vlm_sft"
    model_path: str = ""
    dataset_path: str = ""
    experiment_name: str = ""
    batch_size: int = 2
    learning_rate: float = 2e-5
    max_steps: int = 500
    num_epochs: int = 1
    seq_len: int = 2048
    warmup_steps: int = 50
    lr_schedule: str = "cosine"
    gradient_accumulation: int = 4
    mixed_precision: str = "bf16"
    output_dir: str = "out"
    use_lora: bool = True
    lora_rank: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    tuning_method: str = ""
    use_flash_attention: bool = True
    gpu_ids: list[int] = []


@router.post("/start")
async def start_vlm_training(req: VLMTrainingRequest):
    run_id = "vlm_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = req.model_dump()
    try:
        _normalize_and_validate_vlm_config(config)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    config["run_id"] = run_id
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({"status": "initializing", "step": 0, "loss": 0}))

    module_map = {
        "vlm_pretrain": "homellm.training.vlm_pretrain",
        "vlm_sft": "homellm.training.vlm_sft",
        "vlm_grpo": "homellm.training.vlm_grpo",
    }
    module = module_map.get(req.stage, "homellm.training.vlm_sft")

    env = os.environ.copy()
    if req.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in req.gpu_ids)

    cmd = ["python", "-m", module, "--config", str(run_dir / "config.json")]

    stdout_log = open(run_dir / "stdout.log", "w")
    stderr_log = open(run_dir / "stderr.log", "w")

    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout_log, stderr=stderr_log)
    (run_dir / "pid").write_text(str(proc.pid))
    _raise_if_exited_early(proc, run_dir)

    (RUNS_DIR / "vlm_active_run.json").write_text(json.dumps({
        "run_id": run_id, "started_at": datetime.now().isoformat(), "config": config,
    }, indent=2))

    return {"run_id": run_id}


@router.post("/stop/{run_id}")
async def stop_vlm_training(run_id: str):
    run_dir = RUNS_DIR / run_id
    pid_file = run_dir / "pid"

    if not pid_file.exists():
        raise HTTPException(404, "Run not found")

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    (RUNS_DIR / "vlm_active_run.json").unlink(missing_ok=True)
    return {"success": True}


@router.get("/runs")
async def list_vlm_runs():
    runs = []
    if not RUNS_DIR.exists():
        return {"runs": []}

    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("vlm_"):
            continue
        config_file = d / "config.json"
        metrics_file = d / "metrics.json"

        config = {}
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
            except Exception:
                pass

        status = "unknown"
        current_step = 0
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                status = metrics.get("status", "unknown")
                current_step = metrics.get("step", 0)
            except Exception:
                pass

        runs.append({
            "run_id": d.name,
            "stage": config.get("stage", "vlm_sft"),
            "status": status,
            "started_at": config.get("started_at", ""),
            "model_type": "vlm",
            "experiment_name": config.get("experiment_name", d.name),
            "config": config,
            "current_step": current_step,
            "total_steps": config.get("max_steps", 0),
            "checkpoints": [],
        })

    return {"runs": runs}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class VLMLoadRequest(BaseModel):
    model_path: str


class VLMGenerateRequest(BaseModel):
    prompt: str
    image_base64: str
    max_tokens: int = 256
    temperature: float = 0.7


@router.post("/load")
async def load_vlm(req: VLMLoadRequest):
    """Load a VLM (Qwen2-VL, LLaVA, etc.) via transformers AutoProcessor/AutoModel."""
    global _vlm_state
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(req.model_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            req.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _vlm_state = {"model": model, "processor": processor, "path": req.model_path}
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to load VLM: {e}")


@router.post("/unload")
async def unload_vlm():
    global _vlm_state
    _vlm_state = {"model": None, "processor": None, "path": None}
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"success": True}


@router.get("/status")
async def vlm_status():
    return {
        "loaded": _vlm_state["model"] is not None,
        "model_path": _vlm_state["path"],
    }


@router.post("/generate")
async def generate_vlm(req: VLMGenerateRequest):
    """Generate a response for an image + text prompt."""
    if _vlm_state["model"] is None:
        raise HTTPException(400, "No VLM loaded — call /api/vlm/load first")

    try:
        from PIL import Image
        import torch

        b64 = req.image_base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        model = _vlm_state["model"]
        processor = _vlm_state["processor"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": req.prompt},
            ],
        }]

        try:
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            prompt_text = req.prompt

        inputs = processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=max(req.temperature, 0.01),
                do_sample=req.temperature > 0,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return {"text": text}
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


def _resolve_existing_path(value: str, *, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required")

    p = Path(value).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()

    if not p.exists():
        raise ValueError(f"{label} does not exist: {p}")
    return str(p)


def _normalize_and_validate_vlm_config(config: dict) -> None:
    dataset_path = config.get("data_path") or config.get("dataset_path")
    resolved_data = _resolve_existing_path(dataset_path, label="dataset_path")
    config["data_path"] = resolved_data
    config["dataset_path"] = resolved_data

    model_path = config.get("base_model_path") or config.get("model_path")
    if not model_path:
        raise ValueError("model_path/base_model_path is required")

    config["base_model_path"] = model_path
    config["model_name_or_path"] = model_path
    config["model_path"] = model_path
    config["lora_r"] = int(config.get("lora_r") or config.get("lora_rank") or 16)
    config["lora_rank"] = int(config.get("lora_rank") or config.get("lora_r") or 16)
    config["tuning_method"] = (
        config.get("tuning_method")
        or ("lora" if config.get("use_lora", True) else "full")
    )
    config["lr_schedule"] = config.get("lr_schedule") or config.get("lr_scheduler") or "cosine"
    config["seq_len"] = int(config.get("seq_len") or 2048)
    config["num_epochs"] = int(config.get("num_epochs") or config.get("epochs") or 1)


def _raise_if_exited_early(proc: subprocess.Popen, run_dir: Path) -> None:
    time.sleep(0.5)
    code = proc.poll()
    if code is None:
        return

    stderr = run_dir / "stderr.log"
    detail = f"VLM process exited immediately with code {code}"
    if stderr.exists():
        tail = stderr.read_text(errors="ignore")[-2000:].strip()
        if tail:
            detail = f"{detail}: {tail}"

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            metrics["status"] = "error"
            metrics["error"] = detail
            metrics_path.write_text(json.dumps(metrics, indent=2))
        except Exception:
            pass

    raise HTTPException(500, detail)
