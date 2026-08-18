"""
Retriever Node for Verity's Agentic LangGraph Workflow.

Role:
- Receives the query string and extracted metadata filters from state.
- Calls the hybrid retrieve() function from backend.agent.retrieval (Dense Qdrant + Sparse BM25 fused with RRF).
- Populates state['retrieved_chunks'] with the top retrieved context chunks.
"""

from typing import Dict, List
from backend.agent.retrieval import retrieve
from backend.agent.state import AgentState


def retrieve_node(state: AgentState) -> AgentState:
    """
    Executes hybrid retrieval using query and extracted metadata filters from state.
    Populates state['retrieved_chunks'].
    """
    query = state.get("query", "")
    filters = state.get("extracted_filters") or None

    # Retrieve top hybrid fused chunks (default top_k = 8)
    chunks: List[Dict] = retrieve(query=query, top_k=8, filters=filters)

    state["retrieved_chunks"] = chunks
    return state
