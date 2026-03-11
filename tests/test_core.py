"""
tests/test_core.py — eshu basic test suite.
Run: pytest tests/ -v
"""
import json
import struct
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─── Config ───────────────────────────────────────────────────────────────────

def test_settings_loaded():
    from app.config import settings
    assert settings.OLLAMA_HOST.startswith("http")
    assert settings.CHUNK_SIZE == 512
    assert settings.CHUNK_OVERLAP == 64
    assert settings.RETRIEVAL_TOP_K == 5


# ─── Database Layer ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test_vault.db"
    from app.db.init import init_db
    init_db(db_path)
    return db_path


def test_db_initialises(tmp_db):
    from app.db.init import get_connection
    con = get_connection(tmp_db)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    con.close()
    assert "chunks" in tables


def test_insert_and_query_chunk(tmp_db, monkeypatch):
    """Insert a dummy chunk + embedding and retrieve it by vault."""
    from app.config import settings
    from app.db import queries

    # Point queries to test DB
    monkeypatch.setattr(settings, "SQLITE_PATH", str(tmp_db))

    import uuid
    import datetime

    chunk_id = str(uuid.uuid4())
    dummy_vec = [0.1] * 768

    queries.insert_chunk(
        chunk_id=chunk_id,
        document="This is a test chunk about AI.",
        vault_name="TestVault",
        source_file="test.pdf",
        page_number=1,
        chunk_index=0,
        tags=["ai", "test"],
        media_type="text",
        ingested_at=datetime.datetime.utcnow().isoformat(),
    )
    queries.insert_embedding(chunk_id, dummy_vec)

    results = queries.search_vault("TestVault", dummy_vec, top_k=5)
    assert len(results) >= 1
    assert results[0]["document"] == "This is a test chunk about AI."


def test_vault_stats(tmp_db, monkeypatch):
    from app.config import settings
    from app.db import queries
    monkeypatch.setattr(settings, "SQLITE_PATH", str(tmp_db))
    stats = queries.get_vault_stats("TestVault")
    assert "doc_count" in stats
    assert "chunk_count" in stats


# ─── Chunking ─────────────────────────────────────────────────────────────────

def test_chunk_text_splits():
    from app.services.ingestion import _chunk_text
    long_text = "Hello world. " * 200  # ~400 tokens
    chunks = _chunk_text(long_text, chunk_size=100, overlap=10)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 0


def test_chunk_short_text():
    from app.services.ingestion import _chunk_text
    short = "This is a short sentence."
    chunks = _chunk_text(short, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == short


# ─── Schemas ─────────────────────────────────────────────────────────────────

def test_phi_classification_schema():
    from app.models.schemas import PhiClassification
    clf = PhiClassification(vault="Recipes", confidence=0.91, tags=["vegan"])
    assert clf.vault == "Recipes"
    assert clf.confidence == 0.91


def test_user_rules_schema():
    from app.models.schemas import UserRule, UserRules
    rule = UserRule(trigger_keywords=["GDPR", "legal"], correct_vault="Legal", generated_from=3)
    rules = UserRules(rules=[rule])
    dumped = rules.model_dump_json()
    reloaded = UserRules.model_validate_json(dumped)
    assert reloaded.rules[0].correct_vault == "Legal"


# ─── Router (mocked Phi) ──────────────────────────────────────────────────────

def test_classify_upload_fallback():
    """Router should fall back to 'General' if Phi call fails."""
    from app.services import router
    with patch.object(router, "_phi_generate", side_effect=Exception("timeout")):
        result = router.classify_upload("unknown_file.pdf", "some preview text")
    assert result.vault == "General"
    assert result.confidence == 0.0


def test_route_query_fallback():
    """Router should return all vaults at 0.5 confidence on failure."""
    from app.services import router
    with (
        patch.object(router, "_phi_generate", side_effect=Exception("timeout")),
        patch.object(router, "get_all_fingerprints", return_value={"A": {}, "B": {}}),
    ):
        result = router.route_query("What is LangChain?")
    assert len(result.vaults) == 2


# ─── Learning System ──────────────────────────────────────────────────────────

def test_keyword_extraction():
    from app.services.learning import _extract_keywords
    keywords = _extract_keywords("neural network training with transformer")
    assert "neural" in keywords or "network" in keywords or "transformer" in keywords


def test_load_user_rules_empty(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    from app.services import learning
    rules = learning.load_user_rules()
    assert rules.rules == []


# ─── FastAPI (smoke test) ─────────────────────────────────────────────────────

def test_api_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "vault_count" in data


def test_api_list_vaults():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/vaults")
    assert resp.status_code == 200
    assert "vaults" in resp.json()
