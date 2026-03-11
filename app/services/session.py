"""
app/services/session.py — In-memory session context manager.
Keeps per-session conversation history for context-aware routing.
"""
from __future__ import annotations

from collections import defaultdict
from app.config import settings


# Simple in-memory store: session_id → list of {query, answer}
_sessions: dict[str, list[dict]] = defaultdict(list)


def get_history(session_id: str) -> list[dict]:
    """Retrieve the recent chat history for a session to append as model context."""
    return _sessions[session_id][-settings.CONTEXT_WINDOW_TURNS:]


def append_turn(session_id: str, query: str, answer: str) -> None:
    """Store the user query and assistant response in the session history up to a hard cap of 50 turns."""
    _sessions[session_id].append({"query": query, "answer": answer})
    # Trim to avoid unbounded growth
    if len(_sessions[session_id]) > 50:
        _sessions[session_id] = _sessions[session_id][-50:]


def clear_session(session_id: str) -> None:
    """Remove a session to free memory (e.g., when 'Clear Chat' is clicked)."""
    _sessions.pop(session_id, None)


def list_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_sessions.keys())
