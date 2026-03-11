"""
check_db.py - eshu diagnostic script
Run from project root: python check_db.py
"""
import sys
import sqlite3
import struct
from pathlib import Path

ROOT = Path(__file__).parent

def main():
    # 1 - Find the DB
    db_path = ROOT / "data" / "vault.db"
    print(f"=== eshu Diagnostic ===")
    print(f"Project root : {ROOT}")
    print(f"Expected DB  : {db_path}")
    print()

    if not db_path.exists():
        print("ERROR: vault.db NOT FOUND.")
        print("Did you ingest any documents? Check that DATA_DIR=./data in .env")
        return

    print(f"DB found: {db_path}  ({db_path.stat().st_size // 1024} KB)")

    # 2 - Open raw SQLite
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) as n FROM chunks")
    chunk_count = cur.fetchone()["n"]
    print(f"\nchunks table rows : {chunk_count}")

    cur.execute("SELECT DISTINCT vault_name FROM chunks")
    vaults = [r["vault_name"] for r in cur.fetchall()]
    print(f"Vault names found : {vaults}")

    cur.execute("SELECT source_file, COUNT(*) as n FROM chunks GROUP BY source_file")
    for r in cur.fetchall():
        print(f"  - {r['source_file']}  ->  {r['n']} chunks")

    # 3 - Try sqlite-vec
    print("\n--- sqlite-vec test ---")
    try:
        import sqlite_vec
        con2 = sqlite3.connect(str(db_path))
        con2.row_factory = sqlite3.Row
        con2.enable_load_extension(True)
        sqlite_vec.load(con2)
        con2.enable_load_extension(False)
        print("sqlite_vec loaded: OK")

        cur2 = con2.cursor()
        cur2.execute("SELECT COUNT(*) as n FROM chunk_embeddings")
        emb_count = cur2.fetchone()["n"]
        print(f"chunk_embeddings rows: {emb_count}")

        if emb_count > 0 and chunk_count > 0:
            dummy = struct.pack("768f", *([0.1] * 768))
            cur2.execute(
                "SELECT ce.id FROM chunk_embeddings ce"
                " ORDER BY vec_distance_cosine(ce.embedding, ?) ASC LIMIT 1",
                (dummy,),
            )
            row = cur2.fetchone()
            if row:
                print(f"Cosine search test: OK  (matched id starting with {row['id'][:16]})")
            else:
                print("Cosine search returned 0 rows — embeddings may be missing")
        con2.close()
    except ImportError:
        print("sqlite_vec not importable — is it installed? (pip install sqlite-vec)")
    except Exception as e:
        print(f"sqlite-vec error: {e}")

    # 4 - Try embedding
    print("\n--- Embedding test ---")
    try:
        import httpx, os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        r = httpx.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": "test"},
            timeout=30,
        )
        emb = r.json().get("embedding", [])
        print(f"Ollama embed OK: {model}  dim={len(emb)}")
    except Exception as e:
        print(f"Embedding error: {e}")

    con.close()
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
