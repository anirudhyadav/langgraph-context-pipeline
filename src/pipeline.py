"""
Pipeline — Graph Assembly

Builds and compiles the 5-node context assembly pipeline.

Node order:
    retrieve → trim_history → format_tools → assemble_context → generate → END

Each node controls one lever of the context window.
Nothing reaches the LLM without passing through all five nodes.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    retrieve_node,
    trim_history_node,
    format_tool_outputs_node,
    assemble_context_node,
    generate_node,
)


def build_pipeline() -> StateGraph:
    """Build and compile the context assembly pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve",         retrieve_node)
    graph.add_node("trim_history",     trim_history_node)
    graph.add_node("format_tools",     format_tool_outputs_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("generate",         generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",         "trim_history")
    graph.add_edge("trim_history",     "format_tools")
    graph.add_edge("format_tools",     "assemble_context")
    graph.add_edge("assemble_context", "generate")
    graph.add_edge("generate",         END)

    return graph.compile()


def run(
    query: str,
    messages: list = None,
    tool_outputs: list = None,
    tool_definition_tokens: int = 400,
) -> dict:
    """
    Run the pipeline with a single query.

    Args:
        query:                   The user's question.
        messages:                Prior conversation history (list of BaseMessage).
        tool_outputs:            Tool call results from previous steps.
        tool_definition_tokens:  Approximate token count of registered tools.

    Returns:
        Final AgentState including the response in messages[-1].
    """
    pipeline = build_pipeline()

    return pipeline.invoke({
        "messages":          messages or [],
        "query":             query,
        "retrieved_docs":    [],
        "tool_outputs":      tool_outputs or [],
        "assembled_context": "",
        "token_counts":      {"tool_definitions": tool_definition_tokens},
    })
