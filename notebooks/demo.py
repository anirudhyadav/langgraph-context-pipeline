"""
demo.py — End-to-End Walkthrough
=================================
Run this to see the full context assembly pipeline in action.
Each section prints what's happening at each node.

Usage:
    1. Copy .env.example to .env and fill in your credentials
    2. pip install -r requirements.txt
    3. python notebooks/demo.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import build_pipeline


def print_section(title: str, content: str = ""):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    if content:
        print(content)


def run_demo():
    print_section("Context Assembly Pipeline — Demo")
    print("Building pipeline...")

    pipeline = build_pipeline()

    # ── Single-turn example ───────────────────────────────────────────────────
    print_section("Run 1: Single-turn query")

    query = "What is our current policy on third-party data processors?"
    print(f"Query: {query}\n")

    result = pipeline.invoke({
        "messages":          [],
        "query":             query,
        "retrieved_docs":    [],
        "tool_outputs":      [],
        "assembled_context": "",
        "token_counts":      {"tool_definitions": 400},
    })

    print("Response:")
    print(result["messages"][-1].content)
    print(f"\nTokens used: {result['token_counts'].get('assembled_total', 'N/A')}")
    print(f"Documents retrieved: {len(result['retrieved_docs'])}")

    # ── Multi-turn example ────────────────────────────────────────────────────
    print_section("Run 2: Multi-turn (history trimming in action)")
    print("Simulating a session with 8 prior turns to trigger history trimming...")

    from langchain_core.messages import HumanMessage, AIMessage

    # Simulate prior conversation history (8 turns = triggers trim at MAX_HISTORY_TURNS=6)
    prior_messages = []
    for i in range(1, 5):
        prior_messages.append(HumanMessage(content=f"What about regulation {i}?"))
        prior_messages.append(AIMessage(content=f"Regulation {i} requires X, Y, Z..."))

    query2 = "Does regulation 3 apply to our cloud storage vendors?"

    result2 = pipeline.invoke({
        "messages":          prior_messages,
        "query":             query2,
        "retrieved_docs":    [],
        "tool_outputs":      [],
        "assembled_context": "",
        "token_counts":      {"tool_definitions": 400},
    })

    print(f"Prior turns: {len(prior_messages)}")
    print(f"Turns kept after trim: {len(result2['messages']) - 1}")  # -1 for new response
    print(f"\nResponse:\n{result2['messages'][-1].content}")
    print(f"\nTokens used: {result2['token_counts'].get('assembled_total', 'N/A')}")

    # ── Assembled context inspection ──────────────────────────────────────────
    print_section("Assembled context (what the LLM actually saw)")
    print(result2["assembled_context"][:800] + "...\n[truncated for display]")

    print_section("Demo complete")
    print("Check the token_counts dict on any result to see the budget breakdown.")


if __name__ == "__main__":
    run_demo()
