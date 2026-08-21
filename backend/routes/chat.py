"""Chat endpoints: load model, generate text with streaming."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

_loaded_backend = None
_loaded_model_path = None
_loaded_backend_name = None


class LoadRequest(BaseModel):
    model_path: str
    backend: str = "transformers"


class GenerateRequest(BaseModel):
    messages: list[dict]
    model_path: str
    backend: str = "transformers"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    stream: bool = True


@router.post("/load")
async def load_model(req: LoadRequest):
    """Load a model for chat inference."""
    global _loaded_backend, _loaded_model_path, _loaded_backend_name

    try:
        if req.backend == "vllm":
            from homellm.app.vllm_chat import VLLMChatBackend
            _loaded_backend = VLLMChatBackend(req.model_path)
        elif req.backend == "llama.cpp":
            from homellm.app.llama_cpp_chat import LlamaCppChatBackend
            _loaded_backend = LlamaCppChatBackend(req.model_path)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(req.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                req.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            _loaded_backend = {"model": model, "tokenizer": tokenizer}
        _loaded_model_path = req.model_path
        _loaded_backend_name = req.backend
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/unload")
async def unload_model():
    """Unload the currently loaded model."""
    global _loaded_backend, _loaded_model_path, _loaded_backend_name
    _loaded_backend = None
    _loaded_model_path = None
    _loaded_backend_name = None

    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {"success": True}


@router.get("/backends")
async def available_backends():
    backends = ["transformers"]
    try:
        from homellm.app.vllm_chat import is_vllm_available
        if is_vllm_available():
            backends.append("vllm")
    except Exception:
        # vllm может конфликтовать с transformers (aimv2 etc.) — не валим эндпойнт
        pass
    try:
        from homellm.app.llama_cpp_chat import is_llama_cpp_available
        if is_llama_cpp_available():
            backends.append("llama.cpp")
    except Exception:
        pass
    return {"backends": backends}


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Generate a response. Supports SSE streaming."""
    if req.stream:
        return StreamingResponse(
            _stream_generate(req),
            media_type="text/event-stream",
        )
    else:
        text = await _generate_full(req)
        return {"text": text}


async def _stream_generate(req: GenerateRequest):
    """SSE generator for streaming tokens."""
    try:
        if req.backend in ("vllm", "llama.cpp"):
            backend = _ensure_chat_backend(req)
            prompt = _apply_backend_chat_template(backend, req.messages)
            text = backend.generate(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            )
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from threading import Thread

        model_path = req.model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if _loaded_backend and isinstance(_loaded_backend, dict) and _loaded_model_path == model_path:
            model = _loaded_backend["model"]
            tokenizer = _loaded_backend["tokenizer"]
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        chat_text = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": req.max_tokens,
            "temperature": max(req.temperature, 0.01),
            "top_p": req.top_p,
            "top_k": req.top_k,
            "do_sample": req.temperature > 0,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"

        thread.join()
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


async def _generate_full(req: GenerateRequest) -> str:
    """Non-streaming generation."""
    try:
        if req.backend in ("vllm", "llama.cpp"):
            backend = _ensure_chat_backend(req)
            prompt = _apply_backend_chat_template(backend, req.messages)
            return backend.generate(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(req.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            req.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        chat_text = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01),
            top_p=req.top_p,
            do_sample=req.temperature > 0,
        )

        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text

    except Exception as e:
        raise HTTPException(500, str(e))


def _ensure_chat_backend(req: GenerateRequest):
    """Return the requested loaded backend, loading it lazily if needed."""
    global _loaded_backend, _loaded_model_path, _loaded_backend_name

    if (
        _loaded_backend is not None
        and _loaded_model_path == req.model_path
        and _loaded_backend_name == req.backend
    ):
        return _loaded_backend

    if req.backend == "vllm":
        from homellm.app.vllm_chat import VLLMChatBackend
        _loaded_backend = VLLMChatBackend(req.model_path)
    elif req.backend == "llama.cpp":
        from homellm.app.llama_cpp_chat import LlamaCppChatBackend
        _loaded_backend = LlamaCppChatBackend(req.model_path)
    else:
        raise ValueError(f"Unsupported chat backend: {req.backend}")

    _loaded_model_path = req.model_path
    _loaded_backend_name = req.backend
    return _loaded_backend


def _apply_backend_chat_template(backend, messages: list[dict]) -> str:
    if hasattr(backend, "apply_chat_template"):
        return backend.apply_chat_template(messages, add_generation_prompt=True)

    chunks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        chunks.append(f"{role}: {content}")
    chunks.append("assistant:")
    return "\n".join(chunks)
