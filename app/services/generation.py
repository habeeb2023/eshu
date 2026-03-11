"""
app/services/generation.py
LangChain-powered RAG chain using ChatOllama + ConversationBufferWindowMemory.
Generator: MAIN_MODEL | Router: phi (separate, in router.py)
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever, Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

from app.config import settings
from app.models.schemas import GeneratedAnswer, SourceChunk, ConflictFlag


# ─── Custom retriever that wraps our sqlite-vec search ───────────────────────

class VaultRetriever(BaseRetriever):
    """
    LangChain-compatible retriever backed by our sqlite-vec database.
    
    Rather than doing the vector search inside LangChain, our routing layer
    already fetched the ideal `SourceChunk`s. This class simply wraps those 
    pre-fetched chunks into LangChain `Document` objects so they can be fed 
    into the standard `ConversationalRetrievalChain`.
    """
    source_chunks: list[Any] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        docs = []
        for chunk in self.source_chunks:
            page_info = f" (page {chunk.page_number})" if chunk.page_number else ""
            media_note = f" [{chunk.media_type}]" if chunk.media_type != "text" else ""
            metadata = {
                "source": chunk.source_file,
                "vault": chunk.vault_name,
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "media_type": chunk.media_type,
            }
            docs.append(Document(
                page_content=chunk.text_snippet,
                metadata=metadata,
            ))
        return docs

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self._get_relevant_documents(query)


# ─── Prompt template ─────────────────────────────────────────────────────────

_QA_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant with access to a personal knowledge base.
Use the following retrieved context to answer the user's question accurately and concisely.
Always cite which document/vault your information comes from using inline references like [source_file, vault].
If you detect any contradictions between sources, clearly mark it with ⚠️ CONFLICT.
If the context doesn't contain enough information, say so honestly — do not make things up.

Context from knowledge vaults:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""")


# ─── Per-session memory store (replaces deprecated ConversationBufferWindowMemory) ──

class _BufferWindowMemory:
    """Lightweight sliding-window chat history. Replaces deprecated ConversationBufferWindowMemory."""

    def __init__(self, k: int, memory_key: str = "chat_history"):
        self.k = k
        self.memory_key = memory_key
        self._buffer: list[tuple[str, str]] = []

    def load_memory_variables(self, _: dict) -> dict[str, str]:
        recent = self._buffer[-self.k :] if self.k > 0 else []
        lines = []
        for q, a in recent:
            lines.append(f"Human: {q}")
            lines.append(f"AI: {a}")
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        q = inputs.get("question", inputs.get("input", ""))
        a = outputs.get("answer", outputs.get("output", ""))
        if q or a:
            self._buffer.append((str(q), str(a)))

    def clear(self) -> None:
        self._buffer.clear()


_session_memories: dict[str, _BufferWindowMemory] = {}


def _get_memory(session_id: str) -> _BufferWindowMemory:
    """Retrieves or initializes the conversational memory buffer for a given session."""
    if session_id not in _session_memories:
        _session_memories[session_id] = _BufferWindowMemory(
            k=settings.CONTEXT_WINDOW_TURNS,
            memory_key="chat_history",
        )
    return _session_memories[session_id]


def clear_memory(session_id: str) -> None:
    """Evicts a session's memory buffer."""
    _session_memories.pop(session_id, None)


# ─── Build the LangChain ConversationalRetrievalChain ─────────────────────────

def _build_chain(
    sources: list[SourceChunk],
    session_id: str,
) -> ConversationalRetrievalChain:
    """
    Constructs the LangChain pipeline linking ChatOllama (the LLM), 
    the VaultRetriever (the context), and the ConversationBuffer (the memory).
    """
    llm = ChatOllama(
        model=settings.MAIN_MODEL,
        base_url=settings.OLLAMA_HOST,
        temperature=0.3,
        num_predict=1024,
    )
    retriever = VaultRetriever(source_chunks=sources)
    memory = _get_memory(session_id)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )
    return chain


# ─── Main generation function ─────────────────────────────────────────────────

def generate_answer(
    query: str,
    sources: list[SourceChunk],
    session_history: list[dict] | None = None,
    session_id: str = "default",
) -> GeneratedAnswer:
    """
    Synchronous generation execution. 
    Constructs the chain, invokes the LLM, and inspects the output for "⚠️ CONFLICT" markers.
    Returns a structured GeneratedAnswer object containing the final text, citations, and flags.
    """
    if not sources:
        return GeneratedAnswer(
            answer=(
                "I couldn't find any relevant information in the knowledge vaults for your question. "
                "Try uploading documents first, or switch on **Global Search** in the sidebar."
            ),
            sources=[],
            vault_attribution=[],
        )

    chain = _build_chain(sources, session_id)

    try:
        result = chain.invoke({"question": query})
        answer_text = result.get("answer", "").strip()
    except Exception as e:
        answer_text = f"⚠️ Generation error: {e}"

    # ── Detect conflicts ──────────────────────────────────────────────
    conflicts: list[ConflictFlag] = []
    if "⚠️ CONFLICT" in answer_text.upper() or "CONFLICT" in answer_text.upper():
        if len(sources) >= 2:
            conflicts.append(ConflictFlag(
                chunk_a_id=sources[0].chunk_id,
                chunk_b_id=sources[1].chunk_id,
                description="Model detected a contradiction between retrieved sources.",
            ))

    vault_attribution = list({s.vault_name for s in sources})

    return GeneratedAnswer(
        answer=answer_text,
        sources=sources,
        vault_attribution=vault_attribution,
        conflicts=conflicts,
    )


# ─── Streaming variant (token-by-token for Streamlit) ─────────────────────────

def generate_answer_stream(
    query: str,
    sources: list[SourceChunk],
    session_id: str = "default",
) -> Iterator[str]:
    """
    Generator function that bypasses the high-level ConversationalRetrievalChain to directly
    stream tokens from ChatOllama. This allows the Streamlit UI to render the answer
    gradually instead of waiting for the full response to complete.
    """
    if not sources:
        yield (
            "I couldn't find any relevant information in the knowledge vaults for your question. "
            "Try uploading documents first, or switch on **Global Search** in the sidebar."
        )
        return

    context = "\n\n---\n\n".join(
        f"[{s.source_file} | {s.vault_name}]\n{s.text_snippet}"
        for s in sources
    )

    memory = _get_memory(session_id)
    chat_history_text = memory.load_memory_variables({}).get("chat_history", "")

    prompt = _QA_PROMPT.format(
        context=context,
        chat_history=chat_history_text,
        question=query,
    )

    llm = ChatOllama(
        model=settings.MAIN_MODEL,
        base_url=settings.OLLAMA_HOST,
        temperature=0.3,
        num_predict=1024,
    )

    full_answer = ""
    for chunk in llm.stream(prompt):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        full_answer += token
        yield token

    # save to memory
    memory.save_context({"question": query}, {"answer": full_answer})
