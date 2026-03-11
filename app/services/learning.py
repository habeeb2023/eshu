"""
app/services/learning.py — Correction logging and rule extraction.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models.schemas import CorrectionRecord, UserRule, UserRules


# ─── Load / Save ─────────────────────────────────────────────────────────────

def load_user_rules() -> UserRules:
    """Load the globally extracted manual routing rules from `user_rules.json`."""
    p = settings.rules_path
    if not p.exists():
        return UserRules()
    return UserRules.model_validate_json(p.read_text(encoding="utf-8"))


def _save_user_rules(rules: UserRules) -> None:
    """Persist updated UserRules to disk."""
    settings.rules_path.write_text(
        rules.model_dump_json(indent=2), encoding="utf-8"
    )


# ─── Correction Logging ───────────────────────────────────────────────────────

def log_correction(
    vault_name: str,
    original_input: str,
    phi_suggestion: str,
    phi_confidence: float,
    user_choice: str,
) -> None:
    """
    Append a user's correction to the vault's `meta.json` file.
    This happens when the UI asks Phi for a vault suggestion, but the user
    explicitly chooses a different vault from the dropdown before clicking Ingest.
    
    After logging, it automatically triggers a background rule extraction pass
    to see if this correction has established a new solid pattern.
    """
    meta_path = settings.meta_path(vault_name)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"vault_name": vault_name, "correction_log": [], "conflict_flags": []}

    record = CorrectionRecord(
        original_input=original_input,
        phi_suggestion=phi_suggestion,
        phi_confidence=phi_confidence,
        user_choice=user_choice,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    meta["correction_log"].append(record.model_dump())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _maybe_extract_rules()


# ─── Rule Extraction ──────────────────────────────────────────────────────────

def _maybe_extract_rules() -> None:
    """
    Scans the `correction_log` arrays across all vaults' `meta.json` files.
    If a specific subset of keywords is repeatedly corrected by the user to the 
    same vault (exceeding `RULE_EXTRACTION_THRESHOLD`), this function crystallizes 
    those keywords into a permanent `UserRule` in `user_rules.json`.
    
    Future router calls will automatically skip Phi and obey this rule 
    if the keywords are seen again.
    """
    threshold = settings.RULE_EXTRACTION_THRESHOLD
    # Collect all corrections across all vaults
    all_corrections: list[dict] = []
    data_dir = settings.data_path / "vaults"
    if not data_dir.exists():
        return
    for meta_file in data_dir.glob("*/meta.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        all_corrections.extend(meta.get("correction_log", []))

    # Group by (keyword_fingerprint, correct_vault)
    pattern_counts: Counter = Counter()
    pattern_meta: dict[tuple, dict] = {}
    for rec in all_corrections:
        if rec["phi_suggestion"] == rec["user_choice"]:
            continue  # no correction — skip
        words = _extract_keywords(rec["original_input"])
        key = (tuple(sorted(words)), rec["user_choice"])
        pattern_counts[key] += 1
        pattern_meta[key] = {
            "correct_vault": rec["user_choice"],
            "not_vault": rec["phi_suggestion"],
            "trigger_keywords": list(words),
        }

    existing_rules = load_user_rules()
    existing_triggers = {tuple(sorted(r.trigger_keywords)) for r in existing_rules.rules}

    new_rules = list(existing_rules.rules)
    for (keywords, correct_vault), count in pattern_counts.items():
        if count >= threshold and keywords not in existing_triggers:
            new_rules.append(
                UserRule(
                    trigger_keywords=list(keywords),
                    correct_vault=correct_vault,
                    not_vault=pattern_meta[(keywords, correct_vault)].get("not_vault"),
                    confidence=1.0,
                    generated_from=count,
                )
            )

    _save_user_rules(UserRules(rules=new_rules))


def _extract_keywords(text: str) -> set[str]:
    """
    Extremely simple keyword extractor meant to group similar correction logs.
    Strips punctuation, lowers text, drops common stop words, and ignores short words.
    """
    stop = {"with", "this", "that", "from", "have", "been", "will", "about"}
    words = {
        w.strip(".,;:!?\"'") for w in text.lower().split()
        if len(w) > 4 and w not in stop
    }
    return words


# ─── Conflict Logging ─────────────────────────────────────────────────────────

def log_conflict(vault_name: str, chunk_a_id: str, chunk_b_id: str, description: str) -> None:
    """
    Appends a detected contradiction to the vault's `meta.json` file.
    This is triggered by Llama 3.1 during the generation phase if the `conflicts` JSON block is populated.
    """
    meta_path = settings.meta_path(vault_name)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"vault_name": vault_name, "correction_log": [], "conflict_flags": []}
    meta["conflict_flags"].append({
        "chunk_a_id": chunk_a_id,
        "chunk_b_id": chunk_b_id,
        "description": description,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
