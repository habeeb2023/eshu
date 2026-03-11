"""
app/db/queries.py — All SQL query helpers used throughout the app.
"""
import json
import struct
from typing import Any

from app.db.init import get_connection
from app.config import settings


# ─── Serialisation helpers ───────────────────────────────────────────

def serialize_vec(vec: list[float]) -> bytes:
    """Pack a list of floats into binary for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


# ─── Write Queries ───────────────────────────────────────────────────

def insert_chunk(
    chunk_id: str,
    document: str,
    vault_name: str,
    source_file: str,
    page_number: int,
    chunk_index: int,
    tags: list[str],
    media_type: str,
    ingested_at: str,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO chunks
            (id, document, vault_name, source_file, page_number,
             chunk_index, tags, media_type, ingested_at, extra_meta)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            chunk_id,
            document,
            vault_name,
            source_file,
            page_number,
            chunk_index,
            json.dumps(tags),
            media_type,
            ingested_at,
            json.dumps(extra_meta) if extra_meta else None,
        ),
    )
    con.commit()
    con.close()


def insert_embedding(chunk_id: str, embedding: list[float]) -> None:
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO chunk_embeddings (id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_vec(embedding)),
    )
    con.commit()
    con.close()


def delete_chunks_by_source(vault_name: str, source_file: str) -> None:
    """Remove all chunks (and embeddings) for a given file."""
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM chunks WHERE vault_name=? AND source_file=?",
        (vault_name, source_file),
    )
    ids = [r["id"] for r in cur.fetchall()]
    if ids:
        placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
        cur.execute(
            f"DELETE FROM chunk_embeddings WHERE id IN ({placeholders})", ids
        )
    con.commit()
    con.close()


def delete_vault(vault_name: str) -> None:
    """Remove all data for an entire vault."""
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM chunks WHERE vault_name=?", (vault_name,)
    )
    ids = [r["id"] for r in cur.fetchall()]
    if ids:
        placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
        cur.execute(
            f"DELETE FROM chunk_embeddings WHERE id IN ({placeholders})", ids
        )
    con.commit()
    con.close()


def check_file_exists(vault_name: str, source_file: str) -> bool:
    """Check if any chunks exist for a given file in a specific vault."""
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        "SELECT 1 FROM chunks WHERE vault_name=? AND source_file=? LIMIT 1",
        (vault_name, source_file),
    )
    exists = cur.fetchone() is not None
    con.close()
    return exists


# ─── Read Queries ────────────────────────────────────────────────────

def search_vault(
    vault_name: str,
    query_vec: list[float],
    top_k: int | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Metadata-filtered semantic search inside a single vault.
    
    This query uses the 'Filter-then-Search' pattern:
    1. The SQL `WHERE` clause strictly filters the `chunks` table by `vault_name` (and optional `tags`).
    2. It then `JOIN`s against the `chunk_embeddings` sqlite-vec virtual table.
    3. The vector distance calculation (`vec_distance_cosine`) is only computed on the chunks 
       that survived the metadata filter, making the search extremely fast and strictly scoped.
       
    Args:
        vault_name: The target vault to restrict the search to.
        query_vec: The user's query embedded as a 768-D float vector.
        top_k: Maximum number of chunks to return.
        tags: Optional list of tags. If provided, a chunk must have at least one of these tags.
        
    Returns:
        A list of dictionaries representing the DB rows, including the calculated `distance`.
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    con = get_connection()
    cur = con.cursor()

    tag_filter = ""
    params: list[Any] = [vault_name]

    if tags:
        # Require chunks to contain at least one of the provided tags by checking the JSON array
        tag_clauses = " OR ".join(
            ["EXISTS (SELECT 1 FROM json_each(c.tags) WHERE value=?)"] * len(tags)
        )
        tag_filter = f"AND ({tag_clauses})"
        params.extend(tags)

    params.append(serialize_vec(query_vec))
    params.append(k)

    cur.execute(
        f"""
        SELECT c.id, c.document, c.vault_name, c.source_file,
               c.page_number, c.chunk_index, c.tags, c.media_type,
               c.extra_meta, ce.distance
        FROM chunk_embeddings ce
        JOIN chunks c ON ce.id = c.id
        WHERE c.vault_name = ?
          {tag_filter}
        ORDER BY vec_distance_cosine(ce.embedding, ?) ASC
        LIMIT ?
        """,
        params,
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def search_global(
    query_vec: list[float],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Perform a semantic search across ALL vaults concurrently.
    Used as a fallback when the intelligent router skips all vaults or the pinned vault yields nothing.
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        """
        SELECT c.id, c.document, c.vault_name, c.source_file,
               c.page_number, c.chunk_index, c.tags, c.media_type,
               c.extra_meta, ce.distance
        FROM chunk_embeddings ce
        JOIN chunks c ON ce.id = c.id
        ORDER BY vec_distance_cosine(ce.embedding, ?) ASC
        LIMIT ?
        """,
        (serialize_vec(query_vec), k),
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def get_vault_stats(vault_name: str) -> dict[str, Any]:
    """Retrieve fast aggregate statistics (doc count, chunk count) for a specific vault."""
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        """
        SELECT COUNT(DISTINCT source_file) AS doc_count,
               COUNT(*) AS chunk_count
        FROM chunks WHERE vault_name=?
        """,
        (vault_name,),
    )
    row = dict(cur.fetchone())
    con.close()
    return row


def list_all_vaults() -> list[str]:
    """Retrieve a list of all distinct vault names currently possessing chunks in the database."""
    con = get_connection()
    cur = con.cursor()
    cur.execute("SELECT DISTINCT vault_name FROM chunks ORDER BY vault_name")
    vaults = [r["vault_name"] for r in cur.fetchall()]
    con.close()
    return vaults

def get_recent_chunks(vault_name: str, limit: int = 3) -> list[dict[str, Any]]:
    """Retrieve a sample of the raw text chunks directly from the DB."""
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        "SELECT id, source_file, document, media_type FROM chunks WHERE vault_name=? ORDER BY rowid DESC LIMIT ?",
        (vault_name, limit)
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows
