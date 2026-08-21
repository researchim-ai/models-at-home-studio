"""
FastAPI backend — thin API layer between Electron/React frontend and Python ML code.
Replaces Streamlit as the UI communication layer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.paths import get_project_root

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent / "models-at-home")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes import (
    training, models, datasets, chat, agent, system, vlm, study,
    blueprints, notebooks,
)
from backend.ws import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Models at Home Studio API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(agent.router, prefix="/api/agent", tags=["agent"])
app.include_router(vlm.router, prefix="/api/vlm", tags=["vlm"])
app.include_router(study.router, prefix="/api/study", tags=["study"])
app.include_router(blueprints.router, prefix="/api/blueprints", tags=["blueprints"])
app.include_router(notebooks.router, prefix="/api/notebooks", tags=["notebooks"])
app.include_router(ws_router)
