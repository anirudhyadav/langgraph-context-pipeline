"""
Generate Node — LLM Call

The simplest node in the pipeline by design.
All the intelligence lives in the upstream nodes.
This node's only job is to call the LLM with the assembled context.

Configuration (via .env):
    LLM_MODEL        Model to use for generation (default: gpt-4o)
    LLM_TEMPERATURE  Sampling temperature (default: 0 for determinism)
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from ..state import AgentState


def generate_node(state: AgentState) -> dict:
    """Call the LLM with the assembled context window."""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )

    # Assembled context is the system message
    # Trimmed history follows as the conversation
    messages = [
        {"role": "system", "content": state["assembled_context"]},
        *[
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in state["messages"]
        ],
    ]

    response = llm.invoke(messages)

    return {"messages": [AIMessage(content=response.content)]}
