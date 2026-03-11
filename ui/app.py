"""
ui/app.py — eshu Streamlit Interface (standalone, no FastAPI backend needed)
set_page_config() is the very first Streamlit call — all imports above it.
Four panels: 💬 Chat · 📤 Upload · 🗄️ Vaults · 📊 Health
"""
from __future__ import annotations

# ── Streamlit MUST be configured first ───────────────────────────────────────
import streamlit as st
from pathlib import Path

# Logo path for favicon and sidebar (eshu — Yoruba deity of crossroads & communication)
_LOGO_PATH = Path(__file__).resolve().parent.parent / "logo" / "eshu-logo.svg"

st.set_page_config(
    page_title="eshu",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else "🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Standard library ─────────────────────────────────────────────────────────
import io, json, os, sys, time, tempfile
from datetime import datetime, timezone

# Add project root so `app.*` imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── App modules ───────────────────────────────────────────────────────────────
SERVICES_OK = True
_SERVICE_ERROR = ""
try:
    from app.config import settings
    from app.db.init import init_db
    from app.db.queries import (
        list_all_vaults, get_vault_stats, get_recent_chunks,
        delete_vault as db_delete_vault, check_file_exists
    )
    from app.services.ingestion import ingest_file
    from app.services.router import classify_upload, route_query, generate_fingerprint_summary
    from app.services.retrieval import retrieve_multi_vault, retrieve_global, retrieve_from_vault, embed_query
    from app.services.generation import generate_answer, generate_answer_stream, clear_memory
    from app.services.session import get_history, append_turn, clear_session
    from app.services.learning import log_correction, load_user_rules
    from app.services import vault_registry
    from app.db.queries import search_vault as _search_vault
    import httpx
    init_db()
except Exception as _e:
    SERVICES_OK = False
    _SERVICE_ERROR = str(_e)


# ═════════════════════════════════════════════════════════════════════════════
#  CSS / THEME — Sleek professional dark theme
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"]{font-family:'DM Sans',system-ui,sans-serif;color:#e2e8f0;-webkit-font-smoothing:antialiased}
h1,h2,h3,h4,h5,h6{font-family:'DM Sans',system-ui,sans-serif;}

/* Main — refined gradient, subtle depth */
.main{background: linear-gradient(165deg, #0b0b14 0%, #0d0d1a 35%, #080810 100%) !important;}
.main .block-container{padding:1.75rem 2.5rem 4rem 2.5rem;max-width:1100px}

/* Sidebar — cleaner, minimal */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0c0c14 0%, #0a0a12 100%) !important;
  border-right: 1px solid rgba(99,102,241,.12);
}
section[data-testid="stSidebar"] *{color:#cbd5e1 !important}

/* Headings — tighter hierarchy */
h1{font-size:1.85rem!important;font-weight:700!important;letter-spacing:-0.03em;
   color:#f1f5f9!important;margin-bottom:.4rem!important;}
h2{font-size:1.25rem!important;font-weight:600!important;color:#a5b4fc!important;letter-spacing:-0.02em;margin-top:1.5rem!important;}
h3{font-size:.9rem!important;font-weight:600!important;color:#94a3b8!important;letter-spacing:.02em;text-transform:uppercase;}
hr{border:none;border-top:1px solid rgba(99,102,241,.12)!important;margin:1.25rem 0}

/* Buttons — subtle, professional */
.stButton>button{
  background: linear-gradient(135deg, #4f46e5 0%, #5b21b6 100%)!important;
  color:#fff!important;border:none!important;
  border-radius:10px!important;font-weight:500!important;
  padding:.5rem 1rem!important;font-size:.9rem!important;
  transition: all .18s ease;
  box-shadow: 0 2px 8px rgba(79,70,229,.25);
}
.stButton>button:hover{
  background: linear-gradient(135deg, #6366f1 0%, #6d28d9 100%)!important;
  box-shadow: 0 4px 14px rgba(99,102,241,.35);
  transform: translateY(-1px);
}
.stButton>button:active{transform:translateY(0)}

/* Chat bubbles — refined, readable */
.user-bubble{
  background: linear-gradient(135deg, rgba(67,56,202,.85), rgba(88,28,135,.85));
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(129,140,248,.25);
  border-radius: 18px 18px 4px 18px;
  padding: .9rem 1.2rem; margin: .5rem 0 .5rem 22%;
  color: #f8fafc; line-height: 1.6; font-size: .94rem;
  box-shadow: 0 2px 12px rgba(0,0,0,.2);
}
.bot-bubble{
  background: rgba(15,15,35,.85);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(99,102,241,.15);
  border-radius: 18px 18px 18px 4px;
  padding: .9rem 1.2rem; margin: .5rem 18% .5rem 0;
  color: #f1f5f9; line-height: 1.65; font-size: .94rem;
  white-space: pre-wrap;
  box-shadow: 0 2px 12px rgba(0,0,0,.15);
}
.bot-bubble code{background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);border-radius:6px;padding:.15rem .45rem;font-family:'JetBrains Mono',monospace;font-size:.85em;color:#c7d2fe;}
.bot-bubble strong{color:#c7d2fe;font-weight:600;}

/* Badges — compact, crisp */
.badge{display:inline-flex;align-items:center;border-radius:6px;padding:.2rem .6rem;
       font-size:.7rem;font-weight:500;margin:.1rem .08rem;letter-spacing:.02em;}
.b-src{background:rgba(30,27,75,.7);border:1px solid rgba(67,56,202,.5);color:#a5b4fc;}
.b-high{background:rgba(5,46,22,.7);border:1px solid rgba(22,163,74,.5);color:#86efac;}
.b-med{background:rgba(67,20,7,.7);border:1px solid rgba(234,88,12,.5);color:#fed7aa;}
.b-low{background:rgba(69,10,10,.7);border:1px solid rgba(220,38,38,.5);color:#fca5a5;}
.b-vault{background:rgba(30,27,75,.7);border:1px solid rgba(99,102,241,.5);color:#c7d2fe;}

/* Cards — clean, minimal hover */
.card{
  background: rgba(17,17,45,.5);
  border: 1px solid rgba(99,102,241,.12);
  border-radius: 12px;
  padding: 1rem 1.2rem; margin-bottom: .75rem;
  transition: border-color .2s, background .2s;
}
.card:hover{border-color: rgba(99,102,241,.25); background: rgba(17,17,45,.65);}

/* Metric — understated */
.metric{text-align:center;padding:1.25rem 1rem;
  background: rgba(17,17,45,.5);
  border: 1px solid rgba(99,102,241,.12);
  border-radius: 12px;
  transition: border-color .2s;}
.metric:hover{border-color: rgba(99,102,241,.25);}
.metric .val{font-size:2rem;font-weight:700;color:#a5b4fc;line-height:1.2;}
.metric .lbl{font-size:.75rem;color:#64748b;margin-top:.25rem;font-weight:500;text-transform:uppercase;letter-spacing:.08em;}

/* Conflict — subtle pulse */
.conflict{background:rgba(67,20,7,.6);border-left:3px solid #f97316;
  border-radius:8px;padding:.65rem 1rem;color:#ffedd5;font-size:.85rem;margin:.5rem 0;}

/* Inputs — clean */
.stTextInput input,.stTextArea textarea{
  background: rgba(13,13,35,.9)!important;
  border: 1px solid rgba(99,102,241,.2)!important;
  border-radius: 10px!important;
  color: #f8fafc!important; font-size: .94rem;
  transition: border-color .15s, box-shadow .15s;
}
.stTextInput input:focus,.stTextArea textarea:focus{
  border-color: #6366f1!important;
  box-shadow: 0 0 0 2px rgba(99,102,241,.15)!important;
}
[data-testid="stFileUploader"]{
  background: rgba(99,102,241,.04)!important;
  border: 2px dashed rgba(99,102,241,.3)!important;
  border-radius: 12px!important;
  transition: all .2s;
}
[data-testid="stFileUploader"]:hover{
  background: rgba(99,102,241,.08)!important;
  border-color: rgba(99,102,241,.5)!important;
}

/* Scrollbar */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(99,102,241,.35);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(99,102,241,.5)}

/* Expanders */
details,summary{background:rgba(13,13,35,.6)!important;border-radius:10px!important;border:1px solid rgba(99,102,241,.12);}
summary{padding:.5rem 1rem!important;font-weight:500;color:#cbd5e1!important;}

/* Chat input area */
[data-testid="stChatInput"]{padding:.5rem 0}
[data-testid="stChatInput"] textarea{min-height:48px!important}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SERVICE GUARD
# ═════════════════════════════════════════════════════════════════════════════
if not SERVICES_OK:
    st.error("⚠️ eshu services failed to load.")
    st.code(_SERVICE_ERROR)
    st.info("Run `pip install -r requirements.txt` and ensure you are launching from the project root.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
_DEFS = dict(
    chat_history=[],
    session_id="default",
    pinned_vault="(auto-route)",
    global_search=False,
    panel="💬 Chat",
    show_sources=True,
    clf_file_name=None,
    clf_first_name=None,
    clf_raw_bytes=None,
    clf_result=None,
    upload_classified=False,
    last_retrieval_error="",   # survives st.rerun()
    last_retrieval_debug="",   # vault/chunk info
)
for k, v in _DEFS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _all_vault_infos():
    """Fetches all vault metadata objects from the central registry."""
    return vault_registry.list_vaults()

def _vault_names():
    """Returns a list of just the vault string names for UI dropdowns."""
    return [v.name for v in _all_vault_infos()]

def _ollama_ok() -> bool:
    """Verifies that the local Ollama daemon is active and responding to API requests."""
    try:
        r = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def _badge(text: str, cls: str = "b-src") -> str:
    """Renders a styled HTML badge pill for sources, confidence, and vault attribution."""
    return f"<span class='badge {cls}'>{text}</span>"

def _conf_cls(c: float) -> str:
    """Maps a confidence score (0.0 - 1.0) to a UI badge color CSS class."""
    return "b-high" if c >= .85 else ("b-med" if c >= .60 else "b-low")

def _conf_label(c: float) -> str:
    """Generates a human-readable label for the router's confidence float."""
    if c >= .85: return f"✅ High Confidence ({c:.0%})"
    if c >= .60: return f"⚠️ Medium Confidence ({c:.0%})"
    return f"❓ Low Confidence ({c:.0%})"

# Meta-query detector: questions about the system itself (files, vaults, etc.)
_META_KEYWORDS = [
    "what files", "list files", "what documents", "list documents",
    "what have you", "what do you have", "what's uploaded", "show files",
    "how many docs", "how many documents", "show vaults", "what vaults",
    "list vaults",
]

def _is_meta_query(query: str) -> bool:
    """Detects if the user is asking about the system's own state (e.g., 'What files do you have?')."""
    q = query.lower()
    return any(kw in q for kw in _META_KEYWORDS)

def _handle_meta_query(query: str) -> str:
    """Answer questions about the system state (files, vaults, chunk counts)."""
    from app.db.queries import list_all_vaults, get_vault_stats
    vaults = list_all_vaults()
    if not vaults:
        return "You haven't uploaded any documents yet. Go to the **📤 Upload** panel to add files."

    lines = ["Here's what's currently in your knowledge base:\n"]
    for vname in vaults:
        stats = get_vault_stats(vname)
        vi = vault_registry.get_vault(vname)
        lines.append(f"**🗄 Vault: {vname}**")
        lines.append(f"- 📄 {stats.get('doc_count', 0)} document(s), 🧩 {stats.get('chunk_count', 0)} chunks")
        if vi and vi.topic_summary:
            lines.append(f"- 📝 {vi.topic_summary[:120]}…")
        lines.append("")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), use_container_width=True)
    else:
        st.markdown("### eshu")
    st.markdown("<small style='color:#64748b;font-weight:500'>Local · Private · Knowledge Router</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.session_state.panel = st.radio(
        "nav", ["💬 Chat", "📤 Upload", "📝 Live Logs", "🗄️ Vaults", "📊 Health", "⚙️ Settings"],
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.markdown("---")

    vault_infos = _all_vault_infos()
    vnames = [v.name for v in vault_infos]

    st.markdown("**Active Vaults**")
    if vault_infos:
        with st.container(height=240 if len(vault_infos) > 3 else None, border=False):
            for v in vault_infos:
                live = get_vault_stats(v.name)
                st.markdown(
                    f"<div class='card' style='padding:.55rem 1rem;margin-bottom:.35rem'>"
                    f"<b style='color:#a5b4fc'>🗄 {v.name}</b><br>"
                    f"<small style='color:#475569'>{live.get('doc_count', 0)} docs · {live.get('chunk_count', 0)} chunks</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No vaults yet. Upload a document to begin.")

    st.markdown("---")
    st.markdown("**Search Settings**")
    st.session_state.pinned_vault = st.selectbox("📌 Pin vault", ["(auto-route)"] + vnames)
    st.session_state.global_search = st.toggle("🌐 Global Search")
    st.session_state.show_sources = st.toggle("📎 Show Sources", value=True)

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        clear_session(st.session_state.session_id)
        clear_memory(st.session_state.session_id)
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"<small style='color:#334155'>🤖 Chat: <b>{settings.MAIN_MODEL}</b><br>"
        f"🔀 Router: <b>{settings.ROUTER_MODEL}</b></small>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL — CHAT
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.panel == "💬 Chat":
    st.markdown("## Chat")
    st.caption("Chat · Ready" if not list_all_vaults() else f"{len(list_all_vaults())} vault(s) · {sum(get_vault_stats(v).get('chunk_count',0) for v in list_all_vaults())} chunks")
    st.markdown("---")

    # Show persistent retrieval debug info
    if st.session_state.get("last_retrieval_error"):
        st.error(f"🔴 Retrieval error: {st.session_state.last_retrieval_error}")
    if st.session_state.get("last_retrieval_debug"):
        st.caption(f"🔍 Last search: {st.session_state.last_retrieval_debug}")

    # Render conversation history
    for turn in st.session_state.chat_history:
        st.markdown(f"<div class='user-bubble'>🙋&thinsp; {turn['query']}</div>", unsafe_allow_html=True)
        if turn.get("is_meta"):
            st.markdown(f"<div class='bot-bubble'>🤖&thinsp; {turn['answer']}</div>", unsafe_allow_html=True)
        else:
            if turn.get("conflicts"):
                st.markdown("<div class='conflict'>⚠️ <b>Conflict detected</b> — sources may contradict each other.</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-bubble'>🤖&thinsp; {turn['answer']}</div>", unsafe_allow_html=True)

            if st.session_state.show_sources and turn.get("sources"):
                badges = " ".join(
                    _badge(
                        f"[{i+1}] {Path(s['source_file']).name}"
                        + (f" p.{s['page_number']}" if s.get("page_number") else ""),
                        "b-src"
                    )
                    for i, s in enumerate(turn["sources"][:6])
                )
                attr = " · ".join(sorted({s["vault_name"] for s in turn["sources"]}))
                st.markdown(
                    f"<div style='margin:.2rem 0 .8rem;line-height:2'>"
                    f"{badges}<br>"
                    f"<span style='font-size:.7rem;color:#6366f1'>🏷 {attr}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Input
    user_q = st.chat_input("Ask anything about your documents…")

    if user_q:
        # ── Handle meta-queries instantly (no LLM needed) ─────────────────
        if _is_meta_query(user_q):
            answer = _handle_meta_query(user_q)
            st.session_state.last_retrieval_error = ""
            st.session_state.last_retrieval_debug = ""
            st.session_state.chat_history.append({
                "query": user_q, "answer": answer,
                "sources": [], "conflicts": [], "is_meta": True,
            })
            st.rerun()

        # ── Require Ollama ────────────────────────────────────────────────
        elif not _ollama_ok():
            st.error("❌ Ollama is not running. Start it with `ollama serve` in a terminal.")

        else:
            pinned = st.session_state.pinned_vault

            # ── Step 1: Retrieve chunks ───────────────────────────────────
            retrieval_error = ""
            retrieval_debug = ""
            with st.spinner("🔍 Searching knowledge vaults…"):
                try:
                    if st.session_state.global_search:
                        sources = retrieve_global(user_q)
                        retrieval_debug = f"Global search → {len(sources)} chunks"
                    elif pinned and pinned != "(auto-route)":
                        sources = retrieve_from_vault(pinned, user_q)
                        retrieval_debug = f"Pinned vault `{pinned}` → {len(sources)} chunks"
                    else:
                        # Route with Phi, fall back to all-vault search
                        history_list = get_history(st.session_state.session_id)
                        try:
                            route = route_query(user_q, history_list)
                            sources = retrieve_multi_vault(route.vaults, user_q)
                            vault_names_used = [v.vault_name for v in route.vaults]
                            retrieval_debug = f"Routed to {vault_names_used} → {len(sources)} chunks"
                        except Exception as re:
                            sources = retrieve_global(user_q)
                            retrieval_debug = f"Routing failed ({re}), used global → {len(sources)} chunks"
                except Exception as e:
                    retrieval_error = str(e)
                    sources = []
                    # Last-resort: try global search
                    try:
                        sources = retrieve_global(user_q)
                        retrieval_debug = f"Fallback global after error → {len(sources)} chunks"
                    except Exception as e2:
                        retrieval_error += f" | Global also failed: {e2}"

            # Store debug info so it survives rerun
            st.session_state.last_retrieval_error = retrieval_error
            st.session_state.last_retrieval_debug = retrieval_debug

            # ── Step 2: Generate answer (LangChain streaming) ─────────────
            stream_placeholder = st.empty()

            with st.spinner(f"🤖 {settings.MAIN_MODEL} is generating…"):
                try:
                    full_text = ""
                    for token in generate_answer_stream(user_q, sources, st.session_state.session_id):
                        full_text += token
                        stream_placeholder.markdown(
                            f"<div class='bot-bubble'>🤖&thinsp; {full_text}▌</div>",
                            unsafe_allow_html=True,
                        )
                    stream_placeholder.empty()
                    answer_text = full_text.strip()

                    conflicts = []
                    if "⚠️ CONFLICT" in answer_text.upper() and len(sources) >= 2:
                        conflicts = [{"description": "Model flagged a contradiction between sources."}]
                except Exception as e:
                    stream_placeholder.empty()
                    answer_text = f"⚠️ Generation error: {e}"
                    conflicts = []

            append_turn(st.session_state.session_id, user_q, answer_text)
            st.session_state.chat_history.append({
                "query": user_q,
                "answer": answer_text,
                "sources": [s.model_dump() for s in sources],
                "conflicts": conflicts,
                "is_meta": False,
            })
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL — UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.panel == "📤 Upload":
    st.markdown("## Upload")
    st.caption("Drop files to classify with Phi, then ingest into your vaults.")
    st.markdown("---")

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf","jpg","jpeg","png","webp","mp3","wav","m4a","txt","md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            first_file = uploaded_files[0]
            batch_id = "|".join(f.name for f in uploaded_files)
            if batch_id != st.session_state.get("clf_file_name"):
                st.session_state.upload_classified = False
                st.session_state.clf_result = None
                st.session_state.clf_file_name = batch_id
                raw = first_file.read(); first_file.seek(0)
                st.session_state.clf_raw_bytes = raw
                st.session_state.clf_first_name = first_file.name

        if uploaded_files and not st.session_state.upload_classified:
            btn_txt = "🤖 Classify Batch with Phi" if len(uploaded_files) > 1 else "🤖 Classify with Phi"
            if st.button(btn_txt):
                if not _ollama_ok():
                    st.error("❌ Ollama is not running.")
                else:
                    preview = (st.session_state.clf_raw_bytes or b"")[:1000].decode("utf-8", errors="ignore")
                    with st.spinner("Asking Phi to classify…"):
                        try:
                            clf = classify_upload(st.session_state.clf_first_name, preview)
                            st.session_state.clf_result = clf
                            st.session_state.upload_classified = True
                        except Exception as e:
                            st.error(f"Classification failed: {e}")

        if uploaded_files and st.session_state.upload_classified and st.session_state.clf_result:
            clf = st.session_state.clf_result
            st.markdown(
                f"{_badge(_conf_label(clf.confidence), _conf_cls(clf.confidence))}&nbsp;"
                f"{_badge('→ ' + clf.vault, 'b-vault')}",
                unsafe_allow_html=True,
            )
            if clf.summary:
                st.markdown(f"<div class='card'><small style='color:#94a3b8'>📝 {clf.summary}</small></div>", unsafe_allow_html=True)
            if clf.tags:
                st.markdown("**Tags:** " + " ".join(_badge(t) for t in clf.tags), unsafe_allow_html=True)
            if clf.extra_meta:
                with st.expander("📋 Extracted metadata"):
                    st.json(clf.extra_meta)

    with col_r:
        if uploaded_files and st.session_state.upload_classified and st.session_state.clf_result:
            clf = st.session_state.clf_result
            st.markdown("### 📂 Destination Vault")

            cur_vnames = _vault_names()
            options = ["➕ New Vault"] + cur_vnames
            def_idx = options.index(clf.vault) if clf.vault in cur_vnames and clf.confidence >= .60 else 0

            chosen = st.selectbox("Vault", options, index=def_idx)
            is_new = chosen == "➕ New Vault"
            if is_new:
                nname = st.text_input("New vault name", placeholder="e.g. Recipes")
                final_vault = nname.strip() or None
            else:
                final_vault = chosen
                if chosen != clf.vault:
                    st.info(f"Override: `{clf.vault}` → `{chosen}` (will be logged)")

            tag_str = st.text_input("Tags (comma-separated)", value=", ".join(clf.tags))
            final_tags = [t.strip() for t in tag_str.split(",") if t.strip()]

            st.markdown("")
            
            # --- Check for existing files ---
            duplicate_files = []
            if final_vault:
                for uf in uploaded_files:
                    if check_file_exists(final_vault, uf.name):
                        duplicate_files.append(uf.name)
            
            force_reingest = False
            if duplicate_files:
                st.warning(f"⚠️ **Duplicate Files Detected**\nThe following files have already been uploaded, chunked, and embedded in `{final_vault}`:\n" + "\n".join([f"- `{df}`" for df in duplicate_files]))
                force_reingest = st.checkbox("Force re-ingest existing files", value=False)
                
            files_to_ingest = []
            for uf in uploaded_files:
                if uf.name in duplicate_files and not force_reingest:
                    continue
                files_to_ingest.append(uf)
                
            btn_txt2 = f"📥 Ingest {len(files_to_ingest)} Document(s)"
            if len(files_to_ingest) == 0:
                btn_txt2 = "📥 All files are already in the vault"
                
            if st.button(btn_txt2, disabled=not final_vault or len(files_to_ingest) == 0):
                if not _ollama_ok():
                    st.error("❌ Ollama is not running.")
                else:
                    if not is_new and chosen != clf.vault:
                        try:
                            log_correction(final_vault, st.session_state.clf_first_name, clf.vault, clf.confidence, final_vault)
                        except Exception:
                            pass

                    vault_registry.create_vault(final_vault)
                    
                    # Extract raw bytes before threading to avoid Streamlit file IO context issues
                    files_data = [(uf.name, uf.read()) for uf in files_to_ingest]
                    
                    def ingest_worker():
                        log_dir = settings.data_path / "logs"
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_file = log_dir / "ingest_stream.log"
                        log_file.write_text("")  # Clear old logs
                        
                        from app.services.ingestion import log_ingest
                        log_ingest(f"=== Starting background batch ingestion ({len(files_data)} files) ===")
                        
                        total_chunks = 0
                        success_count = 0
                        for idx, (fname, raw) in enumerate(files_data):
                            log_ingest(f"--- Processing {idx+1}/{len(files_data)}: {fname} ---")
                            suffix = Path(fname).suffix
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    tmp.write(raw)
                                    tmp_path = tmp.name

                                count = ingest_file(tmp_path, final_vault, final_tags, extra_meta={"original_filename": fname})
                                try:
                                    from app.db.init import get_connection
                                    con = get_connection()
                                    con.execute("UPDATE chunks SET source_file=? WHERE source_file=? AND vault_name=?", (fname, Path(tmp_path).name, final_vault))
                                    con.commit(); con.close()
                                except Exception:
                                    pass
                                os.unlink(tmp_path)
                                total_chunks += count
                                success_count += 1
                            except Exception as e:
                                try: os.unlink(tmp_path)
                                except: pass
                                log_ingest(f"ERROR processing {fname}: {e}")
                                
                        if success_count > 0:
                            log_ingest("Batch complete. Generating vault topic fingerprint from new metadata...")
                            try:
                                vec = embed_query(final_vault)
                                rows = _search_vault(final_vault, vec, top_k=10)
                                samples = [r["document"] for r in rows]
                                fp = generate_fingerprint_summary(final_vault, samples)
                                vault_registry.update_fingerprint(final_vault, fp.get("topic_summary",""), fp.get("key_themes",[]), [])
                                log_ingest("Fingerprint successfully updated.")
                            except Exception as e:
                                log_ingest(f"Fingerprint generation skipped: {e}")
                                
                            # Force sync live SQLite counts into registry for Streamlit sidebar
                            for vn in list_all_vaults():
                                s = get_vault_stats(vn)
                                vault_registry.update_vault_stats(vn, s["doc_count"], s["chunk_count"])
                                
                        log_ingest(f"=== DONE! Successfully indexed {success_count} files ({total_chunks} chunks). ===")

                    import threading
                    t = threading.Thread(target=ingest_worker, daemon=True)
                    t.start()

                    st.success(f"🚀 Ingestion started in the background! Switch to the **📝 Live Logs** tab to monitor progress.")
                    st.session_state.upload_classified = False
                    st.session_state.clf_result = None
                    st.session_state.clf_file_name = None
                    
        elif not uploaded_files:
            st.markdown(
                "<div class='card' style='text-align:center;padding:2rem 1.5rem;color:#64748b'>"
                "<div style='font-size:2.5rem;opacity:.8'>📄</div>"
                "<div style='margin-top:.5rem;font-size:.9rem;font-weight:500'>Drop a file to begin</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL — LIVE LOGS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.panel == "📝 Live Logs":
    st.markdown("## Live Logs")
    st.caption("Background ingestion status.")
    st.markdown("---")

    log_file = settings.data_path / "logs" / "ingest_stream.log"
    
    col_t1, col_t2 = st.columns([1, 4])
    with col_t1:
        auto_refresh = st.toggle("🔄 Auto-refresh (1s)", value=True)
    with col_t2:
        if st.button("🗑️ Clear Logs"):
            if log_file.exists():
                log_file.write_text("")
                st.rerun()
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    if log_file.exists():
        logs = log_file.read_text(encoding="utf-8")
        st.code(logs or "Waiting for new file uploads...", language="text")
    else:
        st.info("No active ingestion logs found.")
        
    if auto_refresh:
        time.sleep(1)
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL — VAULTS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.panel == "🗄️ Vaults":
    st.markdown("## Vaults")
    st.markdown("---")

    vault_infos = _all_vault_infos()
    if not vault_infos:
        st.info("No vaults yet. Go to **📤 Upload** to ingest your first document.")
    else:
        for v in vault_infos:
            live = get_vault_stats(v.name)
            with st.expander(f"🗄 **{v.name}** &nbsp;·&nbsp; {live.get('doc_count',0)} docs · {live.get('chunk_count',0)} chunks"):
                if v.topic_summary:
                    st.markdown(f"<div class='card'><small>{v.topic_summary}</small></div>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("📄 Docs", live.get("doc_count","—"))
                c2.metric("🧩 Chunks", live.get("chunk_count","—"))
                c3.metric("🕐 Updated", (v.last_updated or "—")[:10])

                if v.key_themes:
                    st.markdown("**Themes:** " + " ".join(_badge(t) for t in v.key_themes), unsafe_allow_html=True)
                if v.coverage_gaps:
                    st.warning("⚠️ Gaps: " + ", ".join(v.coverage_gaps))

                st.markdown("---")
                st.markdown(f"**Physical DB Storage (SQLite `chunks` table):**<br><small style='color:#64748b'>Both text chunks and float vector embeddings are stored locally.</small>", unsafe_allow_html=True)
                samples = get_recent_chunks(v.name, limit=3)
                if samples:
                    for ch in samples:
                        with st.popover(f"📄 {ch['source_file']} (Type: {ch['media_type']})"):
                            st.caption(f"**Chunk ID:** `{ch['id']}`")
                            st.code(ch["document"], language="text")
                else:
                    st.caption("No chunks found in DB.")

                st.markdown("---")
                ck = st.checkbox("Confirm deletion", key=f"chk_{v.name}")
                if st.button("🗑 Delete Vault", key=f"del_{v.name}", disabled=not ck):
                    db_delete_vault(v.name)
                    vault_registry.remove_vault(v.name)
                    st.success(f"Vault **{v.name}** deleted.")
                    time.sleep(.8); st.rerun()

    st.markdown("---")
    st.markdown("### Create Vault")
    nc1, nc2 = st.columns([3, 1])
    with nc1:
        new_vn = st.text_input("Vault name", placeholder="e.g. Legal Research")
        new_vs = st.text_area("Topic summary (optional)", height=72)
    with nc2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("Create"):
            if new_vn.strip():
                vault_registry.create_vault(new_vn.strip(), new_vs)
                st.success(f"✅ **{new_vn}** created!"); time.sleep(.8); st.rerun()
            else:
                st.warning("Enter a vault name.")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL — HEALTH
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.panel == "📊 Health":
    st.markdown("## Health")
    st.markdown("---")

    vault_infos = _all_vault_infos()
    total_chunks = sum(get_vault_stats(v.name).get("chunk_count", 0) for v in vault_infos)
    ollama_up = _ollama_ok()

    for col, val, lbl in zip(
        st.columns(4),
        ["✅" if ollama_up else "❌", settings.MAIN_MODEL, str(len(vault_infos)), str(total_chunks)],
        ["Ollama",              "Chat Model",          "Vaults",               "Total Chunks"],
    ):
        col.markdown(
            f"<div class='metric'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    if not ollama_up:
        st.error("❌ Ollama is unreachable. Run `ollama serve` in a terminal.")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Active Rules")
        rules = load_user_rules()
        if rules.rules:
            for r in rules.rules:
                st.markdown(
                    f"<div class='card'><b style='color:#a5b4fc'>→ {r.correct_vault}</b><br>"
                    f"<small>Keywords: {', '.join(r.trigger_keywords)}</small><br>"
                    f"<small style='color:#475569'>Strength: {r.confidence:.0%} · From {r.generated_from} corrections</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div class='card'><small style='color:#475569'>No rules yet. Override Phi 3+ times for the same pattern to auto-generate a rule.</small></div>", unsafe_allow_html=True)

    with right:
        st.markdown("### Conflicts")
        found = False
        data_dir = settings.data_path / "vaults"
        if data_dir.exists():
            for mf in data_dir.glob("*/meta.json"):
                meta = json.loads(mf.read_text(encoding="utf-8"))
                for flag in meta.get("conflict_flags", []):
                    found = True
                    st.markdown(
                        f"<div class='conflict'><b>{meta.get('vault_name','?')}</b><br>"
                        f"<small>{flag.get('description','')}</small></div>",
                        unsafe_allow_html=True,
                    )
        if not found:
            st.markdown("<div class='card'><small style='color:#475569'>No conflicts detected.</small></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Vault Overview")
    for v in vault_infos:
        st.markdown(
            f"<div class='card'><b style='color:#a5b4fc'>🗄 {v.name}</b>&nbsp;&nbsp;"
            f"<small>{v.doc_count} docs · {v.chunk_count} chunks</small>"
            + (f"&nbsp;·&nbsp;<small style='color:#475569'>{(v.last_updated or '')[:10]}</small>" if v.last_updated else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"<small style='color:#334155'>DB: `{settings.sqlite_path}`</small>", unsafe_allow_html=True)
    if st.button("🔄 Refresh"): st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL — SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.panel == "⚙️ Settings":
    st.markdown("## Settings")
    st.caption("Models, chunking, and routing. Saved to `.env`.")
    st.markdown("---")

    # Fetch available models from Ollama daemon
    available_models = []
    if _ollama_ok():
        try:
            r = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=3)
            available_models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass

    def _update_env(key: str, new_val: str):
        env_path = Path(__file__).parent.parent / ".env"
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        new_lines = []
        found = False
        for line in lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={new_val}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"{key}={new_val}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        
        orig_val = getattr(settings, key, None)
        if isinstance(orig_val, int):
            setattr(settings, key, int(new_val))
        elif isinstance(orig_val, float):
            setattr(settings, key, float(new_val))
        else:
            setattr(settings, key, str(new_val))

    def model_combo(label: str, current_val: str, key_suffix: str) -> str:
        # Prevent duplicates while preserving order
        opts = list(dict.fromkeys([current_val] + available_models + ["(Custom...)"]))
        sel = st.selectbox(label, opts, index=0, key=f"sel_{key_suffix}")
        if sel == "(Custom...)":
            return st.text_input("Enter custom model name", value=current_val, key=f"txt_{key_suffix}").strip()
        return sel

    st.markdown("### Models")
    colLLM1, colLLM2 = st.columns(2)
    with colLLM1:
        st.markdown("**LLM Assignments**")
        curr_main = model_combo("Main Chat Model (Synthesis)", settings.MAIN_MODEL, "main")
        curr_router = model_combo("Router Model (Phi recommended)", settings.ROUTER_MODEL, "router")
    with colLLM2:
        st.markdown("**Embedding & Vision**")
        curr_embed = model_combo("Embedding Model", settings.EMBED_MODEL, "embed")
        curr_vision = model_combo("Vision Classifier (LLaVA)", settings.VISION_MODEL, "vision")

    st.markdown("---")
    st.markdown("### Parameters")
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown("**Chunking Strategy**")
        c_size = st.number_input("Chunk Size (Tokens)", value=settings.CHUNK_SIZE, min_value=64, max_value=2048)
        c_overlap = st.number_input("Chunk Overlap", value=settings.CHUNK_OVERLAP, min_value=0, max_value=512)
        
    with colB:
        st.markdown("**Retrieval Configuration**")
        top_k = st.number_input("Search Top-K", value=settings.RETRIEVAL_TOP_K, min_value=1, max_value=20)
        c_ctx = st.number_input("Memory Window Turns", value=settings.CONTEXT_WINDOW_TURNS, min_value=1, max_value=15)
        
    with colC:
        st.markdown("**Routing Confidence Thresholds**")
        c_high = st.slider("High Confidence Level", min_value=0.5, max_value=0.99, value=settings.ROUTER_CONFIDENCE_HIGH, step=0.05)
        c_med = st.slider("Medium Confidence Level", min_value=0.3, max_value=0.8, value=settings.ROUTER_CONFIDENCE_MED, step=0.05)
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Save All Settings"):
        _update_env("MAIN_MODEL", curr_main)
        _update_env("ROUTER_MODEL", curr_router)
        _update_env("EMBED_MODEL", curr_embed)
        _update_env("VISION_MODEL", curr_vision)
        _update_env("CHUNK_SIZE", str(c_size))
        _update_env("CHUNK_OVERLAP", str(c_overlap))
        _update_env("RETRIEVAL_TOP_K", str(top_k))
        _update_env("CONTEXT_WINDOW_TURNS", str(c_ctx))
        _update_env("ROUTER_CONFIDENCE_HIGH", str(c_high))
        _update_env("ROUTER_CONFIDENCE_MED", str(c_med))
        st.success("✅ Settings saved dynamically to `.env` and applied to active session!")
        time.sleep(1)
        st.rerun()
