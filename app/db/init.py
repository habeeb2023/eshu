"""
app/db/init.py — One-shot database setup.
Run directly:  python -m app.db.init
"""
import sqlite3
from pathlib import Path

import sqlite_vec  # type: ignore

from app.config import settings


def get_connection(path: Path | None = None) -> sqlite3.Connection:
    """
    Establishes and returns a SQLite database connection with the `sqlite-vec` 
    vector search extension loaded.
    
    Args:
        path: Optional specific path to the SQLite DB. Defaults to the central 
              SQLITE_PATH defined in settings.
              
    Returns:
        A configured sqlite3.Connection object ready for vector and scalar queries.
    """
    db_path = path or settings.sqlite_path
    
    # Ensure the directory exists before attempting to create/connect to the DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # check_same_thread=False allows FastAPI/Streamlit multithreading
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    
    # Use Row factory to allow dict-like access by column name (e.g., row['id'])
    con.row_factory = sqlite3.Row
    
    # Load the `sqlite-vec` extension for fast local vector embeddings search
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    
    return con


def init_db(path: Path | None = None) -> None:
    """
    Initializes the database schema if it doesn't already exist.
    Creates the main `chunks` table for scalar metadata and text, 
    and the `chunk_embeddings` virtual table for vector distances.
    """
    con = get_connection(path)
    cur = con.cursor()

    # ── chunks: stores raw text + all metadata ────────────────────────
    # id: Unique UUID for the chunk
    # tags: JSON-serialized list of categorical strings
    # extra_meta: Arbitrary JSON blob for dynamic vault fields
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id           TEXT PRIMARY KEY,
            document     TEXT NOT NULL,
            vault_name   TEXT NOT NULL,
            source_file  TEXT,
            page_number  INTEGER,
            chunk_index  INTEGER,
            tags         TEXT,       -- JSON array  '["tag1","tag2"]'
            media_type   TEXT DEFAULT 'text',
            ingested_at  TEXT,
            extra_meta   TEXT        -- JSON blob for vault-specific fields
        );

        -- Indexes for fast filtering BEFORE vector search (Filter-then-Search pattern)
        CREATE INDEX IF NOT EXISTS idx_vault       ON chunks(vault_name);
        CREATE INDEX IF NOT EXISTS idx_source_file ON chunks(source_file);
        CREATE INDEX IF NOT EXISTS idx_media_type  ON chunks(media_type);
    """)

    # ── chunk_embeddings: sqlite-vec virtual table ────────────────────
    # vec0 is the virtual table engine provided by sqlite-vec
    # embedding: stored as a 768-dimensional float array (matches nomic-embed-text)
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
            id         TEXT PRIMARY KEY,
            embedding  FLOAT[768]
        );
    """)

    con.commit()
    con.close()
    print(f"[DB] Initialised at {settings.sqlite_path}")


if __name__ == "__main__":
    init_db()
