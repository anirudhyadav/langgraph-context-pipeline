"""
AgentState — the shared state object that flows through every node.

Each node reads from state and returns only the keys it modifies.
Nothing enters the context window without a node deciding it should.
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation history — add_messages handles append logic automatically
    messages: Annotated[list[BaseMessage], add_messages]

    # Current user query (extracted from the latest human message)
    query: str

    # Retrieved documents from vector store — populated by retrieve_node
    retrieved_docs: List[dict]

    # Tool call outputs from previous steps
    tool_outputs: List[dict]

    # Final assembled context string — built by assemble_context_node
    assembled_context: str

    # Token tracking across all layers — used by assemble_context_node
    token_counts: dict
