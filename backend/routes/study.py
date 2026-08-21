"""Study Center endpoints: list and serve markdown documents."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from backend.paths import get_project_root

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
STUDY_DIR = PROJECT_ROOT / "study_materials"


@router.get("/index")
async def document_index(lang: str = Query("en")):
    """List available study documents."""
    documents = []

    lang_dir = STUDY_DIR / lang
    if not lang_dir.exists():
        lang_dir = STUDY_DIR / "en"

    if lang_dir.exists():
        for f in sorted(lang_dir.rglob("*.md")):
            rel = f.relative_to(STUDY_DIR)
            documents.append({
                "name": f.stem.replace("_", " ").replace("-", " ").title(),
                "path": str(rel),
                "source": "local",
            })

    return {"documents": documents}


@router.get("/document")
async def get_document(path: str = Query(...)):
    """Get content of a specific document."""
    full_path = STUDY_DIR / path

    if not full_path.exists():
        raise HTTPException(404, "Document not found")

    if not str(full_path.resolve()).startswith(str(STUDY_DIR.resolve())):
        raise HTTPException(403, "Access denied")

    content = full_path.read_text(encoding="utf-8")
    return {"content": content}
