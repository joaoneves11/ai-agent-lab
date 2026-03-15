
from __future__ import annotations

RAG_TOPICS = ["política", "férias", "reembolso", "viagens"]


def route(query: str) -> str:
    """
    Retorna "tool" | "rag" | "direct".
    """
    if not query or not query.strip():
        return "direct"

    lower = query.lower().strip()

    if "2026-" in lower or "gastos" in lower or "despesa" in lower:
        return "tool"

    if any(topic in lower for topic in RAG_TOPICS):
        return "rag"

    return "direct"
