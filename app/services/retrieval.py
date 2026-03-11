"""
app/services/retrieval.py — Filter-then-search retrieval, with robust fallback.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.db.queries import search_vault, search_global, list_all_vaults
from app.models.schemas import SourceChunk, VaultScore


# ─── Embed query ─────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Converts a text string (usually a semantic query) into a 768-D float vector.
    Dispatches to the Ollama embedding API (`nomic-embed-text` by default).
    Used prior to hitting the sqlite-vec cosine distance index.
    """
    resp = httpx.post(
        f"{settings.OLLAMA_HOST}/api/embeddings",
        json={"model": settings.EMBED_MODEL, "prompt": query},
        timeout=settings.EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ─── Single-vault search ──────────────────────────────────────────────────────

def retrieve_from_vault(
    vault_name: str,
    query: str,
    tags: list[str] | None = None,
    top_k: int | None = None,
) -> list[SourceChunk]:
    """
    Retrieves the most semantically relevant chunks strictly bound to a single vault.
    Used when a user explicitly pins a vault in the UI.
    """
    query_vec = embed_query(query)
    rows = search_vault(vault_name, query_vec, top_k, tags)
    return [_row_to_source(r) for r in rows]


# ─── Multi-vault fan-out + re-ranking ────────────────────────────────────────

def retrieve_multi_vault(
    ranked_vaults: list[VaultScore],
    query: str,
    tags: list[str] | None = None,
    top_k_per_vault: int | None = None,
) -> list[SourceChunk]:
    """
    Executes a parallel or sequential fan-out retrieval against multiple candidate vaults.
    
    This is the core of the intelligent multi-vault system:
    1. It takes the vaults suggested by the Phi Router.
    2. Drops any vaults below the MULTI_VAULT_THRESHOLD.
    3. Searches the remaining vaults.
    4. Combines the results and re-ranks them by absolute cosine distance.
    
    CRITICAL FALLBACKS:
    - If Phi yields no vaults above the threshold, it searches ALL vaults.
    - If filtering constraints completely exhaust the results, it fires a final global scan.
    """
    query_vec = embed_query(query)
    k = top_k_per_vault or settings.RETRIEVAL_TOP_K
    threshold = settings.MULTI_VAULT_THRESHOLD

    all_chunks: list[SourceChunk] = []

    # Filter vaults above threshold
    active_vaults = [vs for vs in ranked_vaults if vs.confidence >= threshold]

    # ── CRITICAL FALLBACK: if router gave us nothing usable, search all vaults ──
    if not active_vaults:
        db_vaults = list_all_vaults()
        active_vaults = [VaultScore(vault_name=v, confidence=1.0) for v in db_vaults]

    for vs in active_vaults:
        try:
            rows = search_vault(vs.vault_name, query_vec, k, tags)
            all_chunks.extend([_row_to_source(r) for r in rows])
        except Exception:
            continue  # skip broken vaults, don't crash

    # ── GLOBAL FALLBACK: if all vault searches returned nothing, do global ──
    if not all_chunks:
        rows = search_global(query_vec, k * 2)
        all_chunks = [_row_to_source(r) for r in rows]

    # Re-rank strictly by geometric distance (lower distance = more similar)
    all_chunks.sort(key=lambda c: c.distance)
    return all_chunks[: settings.RETRIEVAL_TOP_K * 2]


# ─── Global search (no vault filter) ─────────────────────────────────────────

def retrieve_global(query: str, top_k: int | None = None) -> list[SourceChunk]:
    """
    Bypasses the vault system entirely to run a brute-force semantic search against
    the entirety of the database. Useful for generic queries like "Find all my recipes".
    """
    query_vec = embed_query(query)
    rows = search_global(query_vec, top_k or settings.RETRIEVAL_TOP_K * 2)
    return [_row_to_source(r) for r in rows]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _row_to_source(row: dict) -> SourceChunk:
    """Converts a raw SQLite dictionary row into a validated SourceChunk Pydantic object."""
    dist = row.get("distance")
    if dist is None:
        dist = 0.0
        
    return SourceChunk(
        chunk_id=row["id"],
        vault_name=row["vault_name"],
        source_file=row.get("source_file") or "",
        page_number=row.get("page_number"),
        text_snippet=row["document"][:600],   # Keep larger snippet for UI context
        distance=float(dist),
        media_type=row.get("media_type", "text"),
    )
