"""
app/services/ingestion.py — Parse, chunk, embed, and store documents.
Supports: PDF (text + images via LLaVA), standalone images (OCR + LLaVA), audio (Whisper).
"""
from __future__ import annotations

import uuid
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import httpx
import tiktoken

from app.config import settings
from app.db.queries import insert_chunk, insert_embedding, delete_chunks_by_source
from app.services.vault_registry import update_vault_stats, update_fingerprint
from app.db.queries import get_vault_stats


def log_ingest(msg: str):
    """Writes a timestamped log to the ingest status file for real-time UI tracking."""
    log_dir = settings.data_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ingest_stream.log"
    ts = datetime.now().strftime("%H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


# ─── Tokeniser ───────────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")  # works for llama-style models


def _count_tokens(text: str) -> int:
    """Calculates the absolute token length of a given text block."""
    return len(_enc.encode(text))


def _chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits long contiguous text into multiple overlapping pieces to preserve context.
    Uses the `cl100k_base` BPE tokenizer to guarantee no chunk exceeds the context window.
    """
    tokens = _enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ─── Ollama helpers ───────────────────────────────────────────────────────────

def _ollama_embed(text: str) -> list[float]:
    """Call nomic-embed-text via Ollama REST API to compute text embeddings."""
    resp = httpx.post(
        f"{settings.OLLAMA_HOST}/api/embeddings",
        json={"model": settings.EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _ollama_vision(image_bytes: bytes, prompt: str = "Describe this image in detail.") -> str:
    """Call LLaVA via Ollama to generate a rich textual description of an embedded or standalone image."""
    import base64
    b64 = base64.b64encode(image_bytes).decode()
    resp = httpx.post(
        f"{settings.OLLAMA_HOST}/api/generate",
        json={
            "model": settings.VISION_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


# ─── PDF Parsing ─────────────────────────────────────────────────────────────

def _parse_pdf(file_path: Path) -> Generator[dict, None, None]:
    """
    Iterates over a PDF using PyMuPDF (fitz). 
    Yields per-page extractions: {text: <str>, page_number: <int>, images: <list[bytes]>}.
    This powers multimodal RAG by capturing both raw text and graphical payloads.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")

    log_ingest(f"Opening PDF: {file_path.name}")
    doc = fitz.open(str(file_path))
    for page in doc:
        page_num = page.number + 1
        text = page.get_text("text").strip()
        images: list[bytes] = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])
        yield {"text": text, "page_number": page_num, "images": images}
    doc.close()


# ─── Image OCR ───────────────────────────────────────────────────────────────

def _ocr_image(image_bytes: bytes) -> str:
    """Extracts raw text via Tesseract OCR for scans or standalone images."""
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


# ─── Audio Transcription ─────────────────────────────────────────────────────

def _transcribe_audio(file_path: Path) -> list[dict]:
    """Transcribes an audio file locally using `faster-whisper`. Returns timestamped text segments."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        raise RuntimeError("faster-whisper not installed.")

    log_ingest(f"Loading Whisper base model for {file_path.name}...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    log_ingest(f"Transcribing audio...")
    segments, _ = model.transcribe(str(file_path), beam_size=5)
    return [
        {"text": seg.text.strip(), "start": seg.start, "end": seg.end}
        for seg in segments
    ]


# ─── Core Ingest function ────────────────────────────────────────────────────

def ingest_file(
    file_path: str | Path,
    vault_name: str,
    tags: list[str] | None = None,
    extra_meta: dict | None = None,
) -> int:
    """
    The orchestrator for data ingestion. 
    1. Removes any existing chunks for this specific file to support seamless overwrites/updates.
    2. Routes the file to specific sub-handlers (.pdf -> _ingest_pdf, .jpeg -> _ingest_image, etc.).
    3. Handles chunk slicing, vectorization via Ollama, and SQLite storage insertion.
    4. Submits aggregate stat updates to the vault registry.
    Returns the total number of chunks successfully stored.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    now_iso = datetime.now(timezone.utc).isoformat()
    inserted = 0

    log_ingest(f"--- Starting ingestion for {path.name} into '{vault_name}' ---")
    # ── Remove stale chunks for this file ────────────────────────────
    delete_chunks_by_source(vault_name, path.name)

    if suffix == ".pdf":
        log_ingest(f"Detected PDF. Starting multimodal extraction...")
        inserted += _ingest_pdf(path, vault_name, tags or [], extra_meta or {}, now_iso)
    elif suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"):
        inserted += _ingest_image(path, vault_name, tags or [], extra_meta or {}, now_iso)
    elif suffix in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
        inserted += _ingest_audio(path, vault_name, tags or [], extra_meta or {}, now_iso)
    else:
        # Try plain text fallback
        text = path.read_text(encoding="utf-8", errors="ignore")
        inserted += _store_text_chunks(
            text, vault_name, path.name, None, tags or [], "text", extra_meta or {}, now_iso
        )

    # ── Refresh vault stats in registry ──────────────────────────────
    stats = get_vault_stats(vault_name)
    update_vault_stats(vault_name, stats["doc_count"], stats["chunk_count"])
    
    log_ingest(f"Finished {path.name}. Inserted {inserted} total chunks.")

    return inserted


def _ingest_pdf(path: Path, vault_name: str, tags: list, extra_meta: dict, now_iso: str) -> int:
    total = 0
    for page_data in _parse_pdf(path):
        # Text chunks
        if page_data["text"]:
            log_ingest(f"[Page {page_data['page_number']}] Chunking text...")
            total += _store_text_chunks(
                page_data["text"], vault_name, path.name,
                page_data["page_number"], tags, "text", extra_meta, now_iso
            )
        # Image chunks (LLaVA description)
        for idx, img_bytes in enumerate(page_data["images"]):
            log_ingest(f"[Page {page_data['page_number']}] Found image {idx+1}. Asking LLaVA for description...")
            desc = _ollama_vision(img_bytes)
            if desc:
                total += _store_text_chunks(
                    desc, vault_name, path.name,
                    page_data["page_number"], tags, "image_description", extra_meta, now_iso
                )
    return total


def _ingest_image(path: Path, vault_name: str, tags: list, extra_meta: dict, now_iso: str) -> int:
    img_bytes = path.read_bytes()
    total = 0
    log_ingest(f"Detected Image. Running Tesseract OCR...")
    # OCR chunk
    ocr_text = _ocr_image(img_bytes)
    if ocr_text.strip():
        log_ingest(f"OCR found text. Storing chunk...")
        total += _store_text_chunks(
            ocr_text, vault_name, path.name, None, tags, "text", extra_meta, now_iso
        )
    # LLaVA description chunk
    log_ingest(f"Asking LLaVA to visually describe the image...")
    desc = _ollama_vision(img_bytes)
    if desc:
        total += _store_text_chunks(
            desc, vault_name, path.name, None, tags, "image_description", extra_meta, now_iso
        )
    return total


def _ingest_audio(path: Path, vault_name: str, tags: list, extra_meta: dict, now_iso: str) -> int:
    segments = _transcribe_audio(path)
    total = 0
    log_ingest(f"Grouping audio slices into dense 30-second text chunks...")
    # Group into 30-second windows
    window_text = ""
    window_start = 0.0
    idx = 0
    for seg in segments:
        window_text += " " + seg["text"]
        if seg["end"] - window_start >= 30 or seg is segments[-1]:
            if window_text.strip():
                chunk_id = str(uuid.uuid4())
                log_ingest(f"Embedding audio transcript slice {idx+1} ({window_start:.1f}s - {seg['end']:.1f}s)...")
                embedding = _ollama_embed(window_text.strip())
                insert_chunk(
                    chunk_id, window_text.strip(), vault_name, path.name,
                    None, idx, tags, "audio_transcript", now_iso,
                    {**extra_meta, "start_sec": window_start, "end_sec": seg["end"]},
                )
                insert_embedding(chunk_id, embedding)
                total += 1
                idx += 1
            window_text = ""
            window_start = seg["end"]
    return total


def _store_text_chunks(
    text: str,
    vault_name: str,
    source_file: str,
    page_number: int | None,
    tags: list[str],
    media_type: str,
    extra_meta: dict,
    now_iso: str,
) -> int:
    """Chunk text, embed each chunk, and write to DB. Returns count stored."""
    chunks = _chunk_text(text)
    for idx, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        embedding = _ollama_embed(chunk)
        insert_chunk(
            chunk_id, chunk, vault_name, source_file,
            page_number, idx, tags, media_type, now_iso, extra_meta,
        )
        insert_embedding(chunk_id, embedding)
    return len(chunks)
