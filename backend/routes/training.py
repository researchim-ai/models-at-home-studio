"""Training endpoints: start/stop/list runs, continue training."""
from __future__ import annotations

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
RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "out"


class TrainingRequest(BaseModel):
    stage: str = "pretrain"
    model_type: str = "home"
    model_path: str = ""
    dataset_path: str = ""
    data_path: str = ""
    experiment_name: str = ""
    model_id: str = ""
    model_name_input: str = ""
    hidden_size: int = 512
    num_layers: int = 8
    n_heads: int = 8
    num_heads: int = 8
    intermediate_size: int = 0
    vocab_size: int = 50257
    dropout: float = 0.0
    arch_preset: str = ""
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_type: str = "swiglu"
    optimizer: str = "adamw"
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    betas: str = "0.9,0.95"
    eps: float = 1e-8
    min_lr_ratio: float = 0.0
    training_mode: str = "steps"
    max_steps: int = 10000
    warmup_steps: int = 1000
    scheduler_resync_on_resume: bool = True
    lr_scheduler: str = "cosine"
    lr_schedule: str = ""
    mixed_precision: str = "bf16"
    fp16_pure: bool = False
    flash_attention: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    grad_checkpoint: bool = True
    liger_kernel: bool = True
    use_liger: bool = True
    liger_fused_ce: bool = True
    output_dir: str = "out"
    save_every: int = 100
    log_every: int = 10
    export_on_checkpoint: bool = True
    merge_lora: bool = True
    max_grad_norm: float = 1.0
    parallel_mode: str = "single"
    gpu_ids: list[int] = []
    deepspeed_config: str = ""
    training_backend: str = "models_at_home"
    # LoRA
    use_lora: bool = False
    tuning_method: str = "full"
    lora_rank: int = 32
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = []
    # Pretrain / continual
    seq_len: int = 2048
    epochs: int = 1
    num_epochs: int = 1
    base_model_path: str = ""
    # SFT
    sft_chat_template: str = ""
    sft_data_format: str = "chat"
    sft_max_seq_length: int = 2048
    sft_packing: bool = False
    sft_train_on_completions_only: bool = False
    # GRPO
    grpo_algorithm: str = "grpo"
    grpo_group_size: int = 8
    grpo_temperature: float = 0.7
    grpo_max_new_tokens: int = 512
    grpo_kl_weight: float = 0.04
    grpo_clip_eps_high: float = 0.2
    grpo_clip_eps_low: float = 0.2
    grpo_prompt_batch_size: int = 8
    grpo_train_batch_size: int = 2
    grpo_max_prompts: int = 1000
    grpo_learning_rate: float = 0.0
    grpo_max_steps: int = 0
    grpo_max_optim_steps: int = 0
    grpo_reward_fn: str = "exact_match"
    grpo_dataset_path: str = ""


@router.post("/start")
async def start_training(req: TrainingRequest):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = req.model_dump()
    try:
        _normalize_and_validate_training_config(config)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    config["run_id"] = run_id
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    initial_metrics = {
        "status": "initializing",
        "step": 0,
        "loss": 0,
        "learning_rate": config["learning_rate"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(initial_metrics))

    env = os.environ.copy()

    if req.stage == "grpo":
        cmd = ["python", "-m", "homellm.training.rl.train_rl", "--config", str(run_dir / "config.json")]
    elif req.stage.startswith("vlm_"):
        module = f"homellm.training.{req.stage}"
        cmd = ["python", "-m", module, "--config", str(run_dir / "config.json")]
    else:
        cmd = [
            "python",
            "-m",
            "homellm.app.trainer_worker",
            "--config",
            str(run_dir / "config.json"),
            "--metrics",
            str(run_dir / "metrics.json"),
        ]

    if req.parallel_mode != "single" and req.parallel_mode != "":
        accel_config = _resolve_accel_config(req.parallel_mode)
        if accel_config:
            cmd = ["accelerate", "launch", "--config_file", accel_config] + cmd[1:]

    if req.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in req.gpu_ids)

    stdout_log = open(run_dir / "stdout.log", "w")
    stderr_log = open(run_dir / "stderr.log", "w")

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=stdout_log,
        stderr=stderr_log,
    )
    (run_dir / "pid").write_text(str(proc.pid))
    _raise_if_exited_early(proc, run_dir)

    active_run = {"run_id": run_id, "started_at": datetime.now().isoformat(), "config": config}
    (RUNS_DIR / "active_run.json").write_text(json.dumps(active_run, indent=2))

    return {"run_id": run_id}


@router.post("/stop/{run_id}")
async def stop_training(run_id: str):
    run_dir = RUNS_DIR / run_id
    pid_file = run_dir / "pid"

    if not pid_file.exists():
        raise HTTPException(404, "Run not found or no PID file")

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            data["status"] = "stopped"
            metrics_path.write_text(json.dumps(data))
        except Exception:
            pass

    (RUNS_DIR / "active_run.json").unlink(missing_ok=True)
    return {"success": True}


@router.get("/runs")
async def list_runs():
    runs = []
    if not RUNS_DIR.exists():
        return {"runs": []}

    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("."):
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
        total_steps = config.get("max_steps", 0)

        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                status = metrics.get("status", "unknown")
                current_step = metrics.get("step", metrics.get("current_step", 0))
                total_steps = metrics.get("total_steps", total_steps)
            except Exception:
                pass

        checkpoints = []
        out_dir = OUTPUT_DIR / config.get("experiment_name", d.name)
        if out_dir.exists():
            checkpoints = sorted([
                p.name for p in out_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint")
            ])

        runs.append({
            "run_id": d.name,
            "stage": config.get("stage", "unknown"),
            "status": status,
            "started_at": config.get("started_at", d.stat().st_mtime),
            "model_type": config.get("model_type", ""),
            "experiment_name": config.get("experiment_name", d.name),
            "config": config,
            "current_step": current_step,
            "total_steps": total_steps,
            "checkpoints": checkpoints,
        })

    return {"runs": runs}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")

    config = {}
    config_file = run_dir / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())

    metrics = {}
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass

    return {
        "run_id": run_id,
        "stage": config.get("stage", "unknown"),
        "status": metrics.get("status", "unknown"),
        "config": config,
        "current_step": metrics.get("step", metrics.get("current_step", 0)),
        "total_steps": metrics.get("total_steps", config.get("max_steps", 0)),
    }


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    import shutil
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return {"success": True}


@router.get("/runs/{run_id}/logs")
async def run_logs(run_id: str, lines: int = 500):
    """Return the last N lines of stdout/stderr logs for a run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")

    def _tail(path: Path, n: int) -> str:
        if not path.exists():
            return ""
        try:
            text = path.read_text(errors="ignore").splitlines()
            return "\n".join(text[-n:])
        except Exception:
            return ""

    return {
        "run_id": run_id,
        "stdout": _tail(run_dir / "stdout.log", lines),
        "stderr": _tail(run_dir / "stderr.log", lines),
    }


@router.post("/runs/{run_id}/continue")
async def continue_training(run_id: str):
    run_dir = RUNS_DIR / run_id
    config_file = run_dir / "config.json"
    if not config_file.exists():
        raise HTTPException(404, "Run config not found")

    config = json.loads(config_file.read_text())
    req = TrainingRequest(**{k: v for k, v in config.items() if k in TrainingRequest.model_fields})
    return await start_training(req)


@router.get("/runs/{run_id}/grpo-samples")
async def grpo_samples(run_id: str, limit: int = 20):
    """Return recent GRPO sample generations from .runs/{id}/samples/ or samples.jsonl."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")

    samples: list = []
    samples_file = run_dir / "samples.jsonl"
    if samples_file.exists():
        lines = samples_file.read_text().strip().split("\n")
        for line in lines[-limit:]:
            try:
                samples.append(json.loads(line))
            except Exception:
                continue

    samples_dir = run_dir / "samples"
    if samples_dir.exists() and not samples:
        for f in sorted(samples_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                samples.append(json.loads(f.read_text()))
            except Exception:
                continue

    return {"samples": samples}


def _resolve_accel_config(mode: str) -> str | None:
    configs_dir = PROJECT_ROOT / "configs"
    mapping = {
        "multi_gpu": "accelerate_multi_gpu.yaml",
        "fsdp": "accelerate_fsdp.yaml",
        "fsdp_offload": "accelerate_fsdp_offload.yaml",
        "deepspeed_zero2": "accelerate_deepspeed_zero2.yaml",
        "deepspeed_zero3": "accelerate_deepspeed_zero3.yaml",
        "deepspeed_zero3_offload": "accelerate_deepspeed_zero3_offload.yaml",
    }
    filename = mapping.get(mode)
    if filename:
        path = configs_dir / filename
        if path.exists():
            return str(path)
    return None


def _resolve_existing_path(value: str, *, label: str) -> str:
    """Resolve a user-provided path against the models-at-home root."""
    if not value:
        raise ValueError(f"{label} is required")

    p = Path(value).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()

    if not p.exists():
        raise ValueError(f"{label} does not exist: {p}")
    return str(p)


def _normalize_and_validate_training_config(config: dict) -> None:
    """Make UI/API config compatible with the legacy training workers."""
    stage = config.get("stage", "pretrain")

    # React uses dataset_path; legacy workers use data_path.
    dataset_path = config.get("data_path") or config.get("dataset_path")
    if dataset_path:
        resolved = _resolve_existing_path(dataset_path, label="dataset_path")
        config["data_path"] = resolved
        config["dataset_path"] = resolved

    # React often writes model_path; legacy fine-tune/RL workers expect base_model_path.
    model_path = config.get("base_model_path") or config.get("model_path")
    if model_path and stage in ("sft", "continual_pretrain", "grpo"):
        config["base_model_path"] = model_path
        config["model_path"] = model_path

    if stage in ("pretrain", "sft", "continual_pretrain") and not config.get("data_path"):
        raise ValueError(f"{stage} requires dataset_path")

    if stage in ("sft", "continual_pretrain") and not config.get("base_model_path"):
        raise ValueError(f"{stage} requires base_model_path/model_path")

    if stage == "grpo":
        grpo_path = config.get("grpo_dataset_path") or config.get("data_path")
        if grpo_path:
            resolved = _resolve_existing_path(grpo_path, label="grpo_dataset_path")
            config["grpo_dataset_path"] = resolved
            config["data_path"] = resolved

    # Worker expects seq_len under this exact name.
    config["seq_len"] = int(config.get("seq_len") or config.get("sft_max_seq_length") or 2048)
    config["epochs"] = int(config.get("epochs") or config.get("num_epochs") or 1)
    config["num_epochs"] = int(config.get("num_epochs") or config.get("epochs") or 1)
    config["lr_schedule"] = config.get("lr_schedule") or config.get("lr_scheduler") or "cosine"
    config["lr_scheduler"] = config["lr_schedule"]
    config["hidden_size"] = int(config.get("hidden_size") or 512)
    config["num_layers"] = int(config.get("num_layers") or 8)
    config["n_heads"] = int(config.get("n_heads") or config.get("num_heads") or 8)
    config["num_heads"] = int(config.get("num_heads") or config.get("n_heads") or 8)
    config["use_flash_attention"] = bool(config.get("use_flash_attention", config.get("flash_attention", True)))
    config["use_liger"] = bool(config.get("use_liger", config.get("liger_kernel", False)))
    config["grad_checkpoint"] = bool(config.get("grad_checkpoint", config.get("gradient_checkpointing", False)))
    config["lora_r"] = int(config.get("lora_r") or config.get("lora_rank") or 32)
    config["lora_rank"] = int(config.get("lora_rank") or config.get("lora_r") or 32)
    if config.get("use_lora") and config.get("tuning_method") in ("", "full", None):
        config["tuning_method"] = "lora"
    if config.get("tuning_method") in ("lora", "qlora"):
        config["use_lora"] = True
    config["training_backend"] = str(config.get("training_backend") or "models-at-home").replace("_", "-")

    # Worker expects betas as a [beta1, beta2] list; the UI sends "0.9,0.95".
    betas = config.get("betas")
    if isinstance(betas, str):
        try:
            parts = [float(x.strip()) for x in betas.split(",")[:2]]
            config["betas"] = parts if len(parts) == 2 else [0.9, 0.95]
        except ValueError:
            config["betas"] = [0.9, 0.95]
    elif not isinstance(betas, (list, tuple)):
        config["betas"] = [0.9, 0.95]

    config["weight_decay"] = float(config.get("weight_decay", 0.1))
    config["min_lr_ratio"] = float(config.get("min_lr_ratio", 0.0) or 0.0)
    config["liger_fused_ce"] = bool(config.get("liger_fused_ce", True))
    config["fp16_pure"] = bool(config.get("fp16_pure", False))

    # Epochs vs steps: the worker uses max_steps only when it is truthy, and
    # otherwise derives the schedule from epochs. Honour the UI's explicit mode.
    if config.get("training_mode") == "epochs":
        config["max_steps"] = 0

    if stage == "grpo":
        config["grpo_learning_rate"] = float(config.get("grpo_learning_rate") or config.get("learning_rate") or 5e-6)
        config["learning_rate"] = config["grpo_learning_rate"]
        grpo_steps = int(
            config.get("grpo_max_optim_steps")
            or config.get("grpo_max_steps")
            or config.get("max_steps")
            or 0
        )
        if grpo_steps > 0:
            config["grpo_max_optim_steps"] = grpo_steps
            config["grpo_max_steps"] = grpo_steps
            config["max_steps"] = grpo_steps
        if config.get("grpo_reward_fn") and not config.get("grpo_reward_rules"):
            reward_fn = str(config.get("grpo_reward_fn"))
            config["grpo_reward_rules"] = [{"type": reward_fn}]


def _raise_if_exited_early(proc: subprocess.Popen, run_dir: Path) -> None:
    """Catch immediate subprocess failures before telling the UI the run started."""
    code = None
    for _ in range(10):
        time.sleep(0.2)
        code = proc.poll()
        if code is not None:
            break
    if code is None:
        return

    stderr = run_dir / "stderr.log"
    detail = f"Training process exited immediately with code {code}"
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
