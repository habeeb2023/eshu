"""
app/services/vault_registry.py — Manages registry.json and per-vault metadata files.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from app.config import settings
from app.models.schemas import Registry, VaultInfo


# ─── Load / Save ─────────────────────────────────────────────────────────────

def _load_registry() -> Registry:
    """Internal: Load the main registry.json file containing all vaults' metadata."""
    p = settings.registry_path
    if not p.exists():
        return Registry()
    return Registry.model_validate_json(p.read_text(encoding="utf-8"))


def _save_registry(reg: Registry) -> None:
    """Internal: Persist the Registry object to registry.json."""
    settings.registry_path.write_text(
        reg.model_dump_json(indent=2), encoding="utf-8"
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def list_vaults() -> list[VaultInfo]:
    """Return a list of VaultInfo objects for all currently registered vaults."""
    return list(_load_registry().vaults.values())


def get_vault(name: str) -> VaultInfo | None:
    """Retrieve metadata for a specific vault by name."""
    return _load_registry().vaults.get(name)


def create_vault(name: str, topic_summary: str = "") -> VaultInfo:
    """
    Register a new knowledge vault.
    Creates the necessary Directory structure, updating `registry.json`,
    and initializing empty `fingerprint.json` and `meta.json` files.
    """
    reg = _load_registry()
    if name not in reg.vaults:
        reg.vaults[name] = VaultInfo(
            name=name,
            topic_summary=topic_summary,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        _save_registry(reg)
        # Ensure directory + empty json files exist
        settings.vault_dir(name)
        _write_fingerprint(name, topic_summary, [], [])
        _write_meta(name, {})
    return reg.vaults[name]


def update_vault_stats(name: str, doc_count: int, chunk_count: int) -> None:
    """Update the aggregate statistics (doc/chunk counts) for a vault in the registry."""
    reg = _load_registry()
    if name not in reg.vaults:
        reg.vaults[name] = VaultInfo(name=name)
    reg.vaults[name].doc_count = doc_count
    reg.vaults[name].chunk_count = chunk_count
    reg.vaults[name].last_updated = datetime.now(timezone.utc).isoformat()
    _save_registry(reg)


def update_fingerprint(
    name: str,
    topic_summary: str,
    key_themes: list[str],
    coverage_gaps: list[str],
) -> None:
    """
    Update a vault's semantic fingerprint (summary, themes, gaps).
    Updates both the central `registry.json` and the vault-specific `fingerprint.json`.
    """
    reg = _load_registry()
    if name not in reg.vaults:
        reg.vaults[name] = VaultInfo(name=name)
    reg.vaults[name].topic_summary = topic_summary
    reg.vaults[name].key_themes = key_themes
    reg.vaults[name].coverage_gaps = coverage_gaps
    _save_registry(reg)
    _write_fingerprint(name, topic_summary, key_themes, coverage_gaps)


def remove_vault(name: str) -> None:
    """Remove a vault entirely from the central registry."""
    reg = _load_registry()
    reg.vaults.pop(name, None)
    _save_registry(reg)


def get_all_fingerprints() -> dict[str, dict]:
    """
    Return a mapping of vault_name -> fingerprint dictionary.
    This data structure is injected into the Phi router prompt to provide
    the LLM with the context required to accurately classify new uploads or route queries.
    """
    reg = _load_registry()
    out = {}
    for name in reg.vaults:
        fp_path = settings.fingerprint_path(name)
        if fp_path.exists():
            out[name] = json.loads(fp_path.read_text(encoding="utf-8"))
        else:
            out[name] = {"topic_summary": reg.vaults[name].topic_summary}
    return out


# ─── Internal file writers ──────────────────────────────────────────────────

def _write_fingerprint(
    name: str, topic_summary: str, key_themes: list[str], coverage_gaps: list[str]
) -> None:
    """Internal: Writes the vault's semantic identity to its `fingerprint.json`."""
    data = {
        "vault_name": name,
        "topic_summary": topic_summary,
        "key_themes": key_themes,
        "coverage_gaps": coverage_gaps,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    settings.fingerprint_path(name).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _write_meta(name: str, extra: dict) -> None:
    """Internal: Initializes or updates the vault's dynamic metadata file (`meta.json`)."""
    data = {
        "vault_name": name,
        "correction_log": [],
        "conflict_flags": [],
        **extra,
    }
    settings.meta_path(name).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
