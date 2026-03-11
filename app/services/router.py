"""
app/services/router.py — Phi-powered vault classifier and query router.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import settings
from app.models.schemas import PhiClassification, PhiQueryRoute, VaultScore
from app.services.vault_registry import get_all_fingerprints
from app.services.learning import load_user_rules


# ─── Shared Ollama call ───────────────────────────────────────────────────────

def _phi_generate(prompt: str) -> str:
    """
    Internal helper to send a synchronous prompt to the local Ollama instance running the Phi model.
    Used exclusively for structural/routing tasks, not conversational generation.
    """
    resp = httpx.post(
        f"{settings.OLLAMA_HOST}/api/generate",
        json={"model": settings.ROUTER_MODEL, "prompt": prompt, "stream": False},
        timeout=settings.ROUTER_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object block from Phi's raw text response.
    Necessary because small models often wrap JSON in markdown blocks or add conversational preamble.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in Phi output:\n{text}")
    return json.loads(match.group())


# ─── Upload Classification ────────────────────────────────────────────────────

def classify_upload(
    filename: str,
    preview: str,  # first ~500 chars of content
) -> PhiClassification:
    """
    Analyzes a newly uploaded document to suggest the most appropriate destination vault.
    
    This function feeds a preview of the document, along with all existing vault fingerprints
    and any user-defined routing rules, to the Phi model. It forces Phi to return a strict 
    JSON structure containing the targeted vault, confidence, tags, and summary.
    
    If parsing fails, it safely defaults to the "General" vault.
    """
    fingerprints = get_all_fingerprints()
    fp_block = json.dumps(fingerprints, indent=2) if fingerprints else "{}"
    rules = load_user_rules()
    rules_block = json.dumps([r.model_dump() for r in rules.rules], indent=2)

    prompt = f"""
You are a document classification assistant. Given a filename, a short content preview,
existing vault fingerprints, and user routing rules, you must decide which vault best fits
this document and extract relevant metadata.

## User Rules (HIGHEST PRIORITY — follow these before anything else)
{rules_block}

## Existing Vault Fingerprints
{fp_block}

## Document to Classify
Filename: {filename}
Content preview (first 500 chars):
\"\"\"{preview[:500]}\"\"\"

## Instructions
Return ONLY a JSON object with these fields:
{{
  "vault": "<vault_name>",
  "confidence": <0.0-1.0>,
  "tags": ["tag1", "tag2"],
  "media_type": "text|image|audio",
  "summary": "<one sentence>",
  "extra_meta": {{}}
}}

If no vault fits, use "General" as the vault name.
""".strip()

    try:
        raw = _phi_generate(prompt)
        data = _extract_json(raw)
        return PhiClassification(**data)
    except Exception as e:
        return PhiClassification(
            vault="General",
            confidence=0.0,
            summary=f"Classification failed: {e}",
        )


# ─── Query Routing ────────────────────────────────────────────────────────────

def route_query(
    query: str,
    session_history: list[dict] | None = None,
) -> PhiQueryRoute:
    """
    Analyzes a user's prompt to determine which vaults contain the relevant information.
    
    It injects the current session history (for context) and vocabulary signatures 
    (vault fingerprints) into the prompt. Phi returns an array of likely vaults and 
    associated confidence scores.
    
    This orchestrates the "Filter-then-Search" pattern by drastically reducing the number
    of chunks that need to be cosine-compared by sqlite-vec.
    """
    fingerprints = get_all_fingerprints()
    fp_block = json.dumps(fingerprints, indent=2) if fingerprints else "{}"
    rules = load_user_rules()
    rules_block = json.dumps([r.model_dump() for r in rules.rules], indent=2)

    history_block = ""
    if session_history:
        recent = session_history[-settings.CONTEXT_WINDOW_TURNS:]
        history_block = "\n".join(
            f"User: {t['query']}\nAssistant: {t['answer'][:200]}"
            for t in recent
        )

    prompt = f"""
You are a query routing assistant for a multi-vault knowledge system.
Given a user question, conversation history, vault fingerprints, and user rules,
return a ranked list of vaults that should be searched.

## User Rules (HIGHEST PRIORITY)
{rules_block}

## Vault Fingerprints
{fp_block}

## Recent Conversation History
{history_block if history_block else "None"}

## User Query
{query}

## Instructions
Return ONLY a JSON object:
{{
  "vaults": [
    {{"vault_name": "<name>", "confidence": <0.0-1.0>}},
    ...
  ],
  "decomposed_queries": ["<sub-query1>", "<sub-query2>"]
}}

Order by confidence descending. Include multiple vaults if the query spans domains.
If only one vault is relevant, return just that one.
""".strip()

    try:
        raw = _phi_generate(prompt)
        data = _extract_json(raw)
        return PhiQueryRoute(**data)
    except Exception:
        # Fallback: if Phi fails or returns garbage JSON, return all known vaults 
        # so the retrieval layer can scan everything rather than returning nothing.
        all_vaults = list(fingerprints.keys())
        return PhiQueryRoute(
            vaults=[VaultScore(vault_name=v, confidence=0.5) for v in all_vaults],
            decomposed_queries=[query],
        )


# ─── Fingerprint Regeneration ─────────────────────────────────────────────────

def generate_fingerprint_summary(vault_name: str, sample_chunks: list[str]) -> dict:
    """
    Background task called by the ingestion pipeline.
    Samples recently ingested chunks from a vault and asks Phi to synthesize a 
    high-level topic summary and extract key themes. This summary forms the 
    'fingerprint' that Phi later uses to accurately route queries.
    """
    combined = "\n\n".join(sample_chunks[:10])  # use up to 10 chunks for summary
    prompt = f"""
Summarise the main topics covered in the following document excerpts from the vault "{vault_name}".

Excerpts:
{combined}

Return ONLY a JSON object:
{{
  "topic_summary": "<3-5 sentence summary>",
  "key_themes": ["theme1", "theme2", "theme3"]
}}
""".strip()

    try:
        raw = _phi_generate(prompt)
        return _extract_json(raw)
    except Exception:
        return {"topic_summary": "", "key_themes": []}
