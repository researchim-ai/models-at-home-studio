"""Model management endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.paths import get_project_root

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "out"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class DownloadRequest(BaseModel):
    repo_id: str
    save_name: str = ""


@router.get("/local")
async def list_local_models():
    """List models downloaded to models/ directory."""
    models = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                arch = _detect_architecture(d)
                models.append({
                    "name": d.name,
                    "path": str(d),
                    "size_bytes": size,
                    "type": "local",
                    "architecture": arch,
                    "modified_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                })
    return {"models": models}


@router.get("/trained")
async def list_trained_models():
    """List trained models from out/ directory."""
    models = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                final = d / "final_model"
                target = final if final.exists() else d
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                arch = _detect_architecture(target)

                model_type = "trained"
                if "lora" in d.name.lower() or (d / "adapter_config.json").exists():
                    model_type = "lora"

                models.append({
                    "name": d.name,
                    "path": str(target),
                    "size_bytes": size,
                    "type": model_type,
                    "architecture": arch,
                    "modified_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                })
    return {"models": models}


@router.post("/download")
async def download_model(req: DownloadRequest):
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise HTTPException(500, "huggingface_hub not installed")

    save_name = req.save_name or req.repo_id.split("/")[-1]
    save_path = MODELS_DIR / save_name

    try:
        snapshot_download(
            repo_id=req.repo_id,
            local_dir=str(save_path),
        )
        return {"success": True, "path": str(save_path)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/local/{name:path}")
async def delete_local_model(name: str):
    """Delete a downloaded model from the models/ directory."""
    import shutil
    path = MODELS_DIR / name
    try:
        path = path.resolve()
        path.relative_to(MODELS_DIR.resolve())
    except (ValueError, OSError):
        raise HTTPException(400, "Invalid model path")
    if not path.exists():
        raise HTTPException(404, "Model not found")
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    return {"success": True}


@router.post("/estimate-memory")
async def estimate_memory(config: dict):
    """Estimate VRAM usage (simple)."""
    try:
        from homellm.models.memory_estimator import estimate_memory_footprint
        result = estimate_memory_footprint(
            config,
            batch_size=int(config.get("batch_size", 1) or 1),
            distributed_mode=config.get("parallel_mode", "default") or "default",
            num_gpus=max(1, len(config.get("gpu_ids") or [])) or 1,
        )
        return result
    except Exception as e:
        return {"vram_gb": 0, "params": 0, "error": str(e)}


class EstimateMemoryDetailedRequest(BaseModel):
    config: dict
    batch_size: int = 1
    distributed_mode: str = "default"
    num_gpus: int = 1


@router.post("/estimate-memory-detailed")
async def estimate_memory_detailed(req: EstimateMemoryDetailedRequest):
    """Full memory estimator — returns breakdown (weights/grads/optim/activations)."""
    try:
        from homellm.models.memory_estimator import estimate_memory_footprint

        cfg = dict(req.config or {})

        # Подтягиваем hidden_size/num_layers/... из HF config.json если указан base_model_path
        if cfg.get("model_type") == "hf" and cfg.get("base_model_path"):
            base_path = Path(cfg["base_model_path"])
            cfg_path = base_path / "config.json" if base_path.is_dir() else None
            if cfg_path and cfg_path.exists():
                try:
                    hf_cfg = json.loads(cfg_path.read_text())
                    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                             "intermediate_size", "max_position_embeddings", "vocab_size"):
                        if k in hf_cfg and k not in cfg:
                            cfg[k] = hf_cfg[k]
                    if "num_hidden_layers" in cfg and "num_layers" not in cfg:
                        cfg["num_layers"] = cfg["num_hidden_layers"]
                    if "num_attention_heads" in cfg and "n_heads" not in cfg:
                        cfg["n_heads"] = cfg["num_attention_heads"]
                except Exception:
                    pass

        # Разумные дефолты для Home-модели, чтобы estimator не падал
        cfg.setdefault("hidden_size", 512)
        cfg.setdefault("num_layers", 8)
        cfg.setdefault("n_heads", 8)
        cfg.setdefault("seq_len", cfg.get("sft_max_seq_length") or 2048)
        cfg.setdefault("vocab_size", 50257)

        return estimate_memory_footprint(
            cfg,
            batch_size=req.batch_size,
            distributed_mode=req.distributed_mode,
            num_gpus=req.num_gpus,
        )
    except Exception as e:
        return {"error": str(e)}


@router.get("/chat-template")
async def get_chat_template(model_path: str):
    """Return the chat_template from a model's tokenizer_config.json if present."""
    p = Path(model_path)
    if not p.exists():
        return {"has_template": False, "template": None, "error": "Path not found"}

    candidates = [p / "tokenizer_config.json", p / "chat_template.json"]
    for c in candidates:
        if c.exists():
            try:
                data = json.loads(c.read_text())
                tmpl = data.get("chat_template")
                if tmpl:
                    return {"has_template": True, "template": tmpl}
            except Exception:
                continue
    return {"has_template": False, "template": None}


def _detect_architecture(path: Path) -> str | None:
    config_file = path / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            return cfg.get("model_type") or cfg.get("architectures", [None])[0]
        except Exception:
            pass
    return None
