"""
Retriever Node for Verity's Agentic LangGraph Workflow.

Role:
- Receives the query string and extracted metadata filters from state.
- Calls the hybrid retrieve() function from backend.agent.retrieval (Dense Qdrant + Sparse BM25 fused with RRF).
- Populates state['retrieved_chunks'] with the top retrieved context chunks.
"""

import logging
from typing import Dict, List
from backend.agent.retrieval import retrieve
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


def retrieve_node(state: AgentState) -> AgentState:
    """
    Executes hybrid retrieval using query and extracted metadata filters from state.
    Includes soft fallback logic if strict filters yield insufficient context chunks.
    Populates state['retrieved_chunks'].
    """
    query = state.get("query", "")
    filters = state.get("extracted_filters") or None

    # Retrieve top hybrid fused chunks (default top_k = 8)
    chunks: List[Dict] = retrieve(query=query, top_k=8, filters=filters)

    # Soft Fallback: If strict metadata filters returned fewer than 3 chunks,
    # perform fallback retrieval without strict filters to ensure sufficient semantic context.
    if filters and len(chunks) < 3:
        logger.info(f"Strict metadata filtering returned {len(chunks)} chunks. Performing soft fallback retrieval without filters...")
        fallback_chunks = retrieve(query=query, top_k=8, filters=None)
        seen_ids = {c.get("id") or c.get("chunk_id") for c in chunks}
        for fc in fallback_chunks:
            fc_id = fc.get("id") or fc.get("chunk_id")
            if fc_id not in seen_ids:
                chunks.append(fc)
                seen_ids.add(fc_id)

    state["retrieved_chunks"] = chunks[:8]
    return state

