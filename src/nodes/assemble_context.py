"""
Assemble Context Node — Levers 2 + 5: Window Structure + State Passing

The most important node in the pipeline. Takes every prepared layer and
assembles them into the final context string with token budget control.

Layer order (highest to lowest priority):
    1. System context        — always first, always included
    2. Tool definitions      — tracked, registered by LangGraph
    3. Retrieved documents   — largest budget, sorted by relevance score
    4. Tool outputs          — included if budget allows

Nothing enters the context window by accident. Every piece of information
is here because this node decided it should be.

Configuration (via .env):
    TOKEN_BUDGET     Total token budget for the context window (default: 10000)
    SYSTEM_PROMPT    System prompt text (has sensible default)
"""

import os
from ..state import AgentState
from ..utils import estimate_tokens

TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "10000"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise AI assistant. "
    "Answer only from the provided context. "
    "If the context does not contain the answer, say so clearly."
)


def assemble_context_node(state: AgentState) -> dict:
    """Assemble all context layers with token budget enforcement."""
    parts: list[tuple[str, str]] = []
    used = 0

    # ── Layer 1: System context — always first, always included ──────────────
    system = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    parts.append(("SYSTEM", system))
    used += estimate_tokens(system)

    # ── Layer 2: Tool definitions — tracked, added by LangGraph externally ───
    used += state["token_counts"].get("tool_definitions", 400)

    # ── Layer 3: Retrieved documents ─────────────────────────────────────────
    # Reserve 1,500 tokens for conversation history and response headroom
    doc_budget = TOKEN_BUDGET - used - 1500
    doc_parts: list[str] = []

    for doc in sorted(state["retrieved_docs"], key=lambda d: d["score"], reverse=True):
        doc_tokens = doc["tokens"]
        if used + doc_tokens < doc_budget:
            doc_parts.append(
                f"[Source: {doc['source']} | Relevance: {doc['score']:.2f}]\n"
                f"{doc['content']}"
            )
            used += doc_tokens

    if doc_parts:
        parts.append(("RETRIEVED DOCUMENTS", "\n\n".join(doc_parts)))

    # ── Layer 4: Tool outputs — include while budget allows ──────────────────
    for tool in state["tool_outputs"]:
        if used + tool["tokens"] < TOKEN_BUDGET - 500:
            parts.append((f"TOOL OUTPUT ({tool['tool']})", tool["result"]))
            used += tool["tokens"]

    # ── Assemble with clear separators ───────────────────────────────────────
    separator = "\n\n" + "=" * 40 + "\n\n"
    assembled = separator.join(
        f"### {label}\n{content}" for label, content in parts
    )

    token_counts = {**state["token_counts"], "assembled_total": used}

    return {
        "assembled_context": assembled,
        "token_counts": token_counts,
    }
