"""
Retrieve Node — Lever 1: Retrieval Quality

Pulls relevant documents from MongoDB Atlas Vector Search.
Filters by relevance score to ensure only useful documents enter the context window.

Configuration (via .env):
    RETRIEVAL_THRESHOLD  Minimum relevance score to include a document (default: 0.75)
    RETRIEVAL_TOP_K      Number of candidates to retrieve before filtering (default: 8)
    MONGODB_URI          MongoDB Atlas connection string
    MONGODB_NAMESPACE    Database.collection namespace (default: mydb.documents)
"""

import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

from ..state import AgentState
from ..utils import estimate_tokens

RELEVANCE_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))


def retrieve_node(state: AgentState) -> dict:
    """Pull relevant docs from vector store. Filter by relevance score."""
    embeddings = OpenAIEmbeddings()

    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=os.environ["MONGODB_URI"],
        namespace=os.getenv("MONGODB_NAMESPACE", "mydb.documents"),
        embedding=embeddings,
    )

    results = vector_store.similarity_search_with_score(
        query=state["query"],
        k=TOP_K,
        pre_filter={"status": {"$eq": "active"}},
    )

    retrieved_docs = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "tokens": estimate_tokens(doc.page_content),
        }
        for doc, score in results
        if float(score) > RELEVANCE_THRESHOLD
    ]

    return {"retrieved_docs": retrieved_docs}
