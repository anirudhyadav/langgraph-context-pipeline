"""
Token counting utilities.

Uses tiktoken for accurate token counts rather than word-count approximations.
cl100k_base matches the encoding used by GPT-4 and GPT-4o.
"""

import tiktoken

_encoder = None


def get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def estimate_tokens(text: str) -> int:
    """Count tokens accurately using tiktoken."""
    if not text:
        return 0
    return len(get_encoder().encode(text))


def tokens_remaining(used: int, budget: int) -> int:
    """How many tokens are left in the budget."""
    return max(0, budget - used)
