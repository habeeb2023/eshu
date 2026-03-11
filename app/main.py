"""
app/main.py — FastAPI application.
Endpoints: /query  /ingest  /vaults  /health  /upload
"""
from __future__ import annotations

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.init import init_db
from app.db.queries import list_all_vaults, get_vault_stats
from app.models.schemas import (
    QueryRequest, QueryResponse,
    IngestResponse,
    VaultCreateRequest, VaultDeleteRequest,
    HealthResponse,
)
from app.services.ingestion import ingest_file
from app.services.router import classify_upload, route_query, generate_fingerprint_summary
from app.services.retrieval import retrieve_multi_vault, retrieve_global, retrieve_from_vault
from app.services.generation import generate_answer
from app.services.session import get_history, append_turn
from app.services.learning import log_correction
from app.services import vault_registry


# ─── App init ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="eshu API",
    description="eshu — Local multi-vault RAG system. Knowledge router at the crossroads.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initializes the SQLite database and loads the sqlite-vec extension on boot."""
    init_db()


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Returns the system health status. 
    Checks if Ollama is reachable and aggregates vault statistics from the database.
    """
    # Check Ollama
    ollama_ok = False
    try:
        r = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    vaults = list_all_vaults()
    total_chunks = sum(get_vault_stats(v)["chunk_count"] for v in vaults)

    return HealthResponse(
        status="ok",
        ollama_reachable=ollama_ok,
        db_path=str(settings.sqlite_path),
        vault_count=len(vaults),
        total_chunks=total_chunks,
    )


# ─── Query ────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    """
    Primary endpoint for user interaction.
    1. Loads session history.
    2. Determines retrieval strategy (Global vs Single Vault vs Multi-Vault Route).
    3. Fetches relevant SourceChunks.
    4. Passes chunks to Llama 3.1 for synthesis.
    """
    history = get_history(req.session_id)

    if req.global_search:
        sources = retrieve_global(req.query)
    elif req.vault_name:
        sources = retrieve_from_vault(req.vault_name, req.query)
    else:
        route = route_query(req.query, history)
        sources = retrieve_multi_vault(route.vaults, req.query)

    result = generate_answer(req.query, sources, history)
    append_turn(req.session_id, req.query, result.answer)

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        vault_attribution=result.vault_attribution,
        conflicts=result.conflicts,
        routing_confidence=None,
    )


# ─── Upload (multipart) ───────────────────────────────────────────────────────

@app.post("/upload", response_model=IngestResponse, tags=["Ingestion"])
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vault_name: str = Form(...),
    tags: str = Form("[]"),
):
    """
    Accepts a multipart file upload from the UI.
    Offloads the heavy ingestion process (parsing, chunking, embedding) to a 
    FastAPI BackgroundTask so the UI doesn't block waiting for a response.
    After ingestion, it triggers a background fingerprint update for the vault.
    """
    parsed_tags: list[str] = json.loads(tags) if tags else []

    # Save temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Ensure vault exists in registry
    vault_registry.create_vault(vault_name)

    # Ingest in background to avoid timeout on large files
    def _do_ingest():
        count = ingest_file(tmp_path, vault_name, parsed_tags)
        os.unlink(tmp_path)
        # Regenerate fingerprint
        from app.db.queries import search_global as _sg
        # grab sample chunks for fingerprint
        from app.services.retrieval import embed_query
        try:
            vec = embed_query(vault_name)
            from app.db.queries import search_vault as _sv
            rows = _sv(vault_name, vec, top_k=10)
            samples = [r["document"] for r in rows]
            fp = generate_fingerprint_summary(vault_name, samples)
            vault_registry.update_fingerprint(
                vault_name,
                fp.get("topic_summary", ""),
                fp.get("key_themes", []),
                [],
            )
        except Exception:
            pass

    background_tasks.add_task(_do_ingest)

    return IngestResponse(
        vault_name=vault_name,
        source_file=file.filename,
        chunks_inserted=-1,   # actual count will be computed in background
    )


# ─── Vaults ───────────────────────────────────────────────────────────────────

@app.get("/vaults", tags=["Vaults"])
async def list_vaults():
    """Lists all configured vaults from the registry."""
    return {"vaults": vault_registry.list_vaults()}


@app.post("/vaults/create", tags=["Vaults"])
async def create_vault(req: VaultCreateRequest):
    """Manually provisions a new, empty vault."""
    info = vault_registry.create_vault(req.name, req.topic_summary)
    return {"vault": info}


@app.delete("/vaults/delete", tags=["Vaults"])
async def delete_vault(req: VaultDeleteRequest):
    """Hard deletes a vault from both the SQLite database and the JSON registry."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to delete a vault.")
    from app.db.queries import delete_vault as db_delete
    db_delete(req.name)
    vault_registry.remove_vault(req.name)
    return {"deleted": req.name}


@app.get("/vaults/{vault_name}/stats", tags=["Vaults"])
async def vault_stats(vault_name: str):
    """Retrieves document and chunk counts for a specific vault from SQLite."""
    return get_vault_stats(vault_name)


# ─── Correction endpoint ──────────────────────────────────────────────────────

@app.post("/correction", tags=["Learning"])
async def submit_correction(
    vault_name: str,
    original_input: str,
    phi_suggestion: str,
    phi_confidence: float,
    user_choice: str,
):
    """
    Logs when a user overrides the Phi router's automatic vault suggestion
    during the upload process. Powers the background rule extraction system.
    """
    log_correction(vault_name, original_input, phi_suggestion, phi_confidence, user_choice)
    return {"status": "correction logged"}
