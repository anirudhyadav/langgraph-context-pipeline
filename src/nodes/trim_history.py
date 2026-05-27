"""
Trim History Node — Lever 3: History Management

Keeps the last N turns of conversation history.
Summarises older turns into a single line to preserve context without bloat.

The most common cause of context window overflow in multi-turn agents is
unbounded history. This node prevents that while keeping the session coherent.

Configuration (via .env):
    MAX_HISTORY_TURNS  Number of recent turns to keep verbatim (default: 6)
    SUMMARY_MODEL      Model to use for summarisation (default: gpt-4o-mini)
"""

import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))


def trim_history_node(state: AgentState) -> dict:
    """Keep recent history. Summarise older turns to prevent context bloat."""
    messages = state["messages"]

    # Nothing to trim
    if len(messages) <= MAX_HISTORY_TURNS:
        return {"messages": messages}

    recent = messages[-MAX_HISTORY_TURNS:]
    older  = messages[:-MAX_HISTORY_TURNS]

    # Use a cheap model for summaries — this runs on every turn
    llm = ChatOpenAI(
        model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    older_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in older
    )

    summary = llm.invoke(
        f"Summarise this conversation history in 2 sentences, "
        f"preserving key decisions, facts, and unanswered questions:\n\n{older_text}"
    ).content

    trimmed = [
        AIMessage(content=f"[Conversation summary: {summary}]"),
        *recent,
    ]

    return {"messages": trimmed}
