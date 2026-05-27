"""
Format Tool Outputs Node — Lever 4: Tool Output Formatting

Trims and structures raw tool outputs before they enter the context window.

Raw API responses are often 3–5x larger than they need to be. Unstructured
JSON dumps confuse the model and waste token budget. This node makes tool
outputs readable and compact.

Configuration (via .env):
    MAX_TOOL_OUTPUT_CHARS  Max characters per tool output (default: 900)
"""

import os
from ..state import AgentState
from ..utils import estimate_tokens

MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "900"))


def format_tool_outputs_node(state: AgentState) -> dict:
    """Trim and structure raw tool outputs."""
    formatted = []

    for output in state["tool_outputs"]:
        content = str(output.get("content", ""))

        # Truncate at word boundary near the character limit
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            truncation_point = content.rfind(" ", 0, MAX_TOOL_OUTPUT_CHARS)
            if truncation_point == -1:
                truncation_point = MAX_TOOL_OUTPUT_CHARS
            content = content[:truncation_point] + "... [truncated]"

        formatted.append({
            "tool":   output.get("tool_name", "unknown"),
            "result": content,
            "tokens": estimate_tokens(content),
        })

    return {"tool_outputs": formatted}
