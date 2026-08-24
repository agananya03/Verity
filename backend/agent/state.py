"""
LangGraph State Schema for Verity Financial RAG Agent.

Defines the AgentState TypedDict passed across nodes in the LangGraph workflow graph.
"""

from typing import Any, Dict, List, TypedDict


class CriticVerdict(TypedDict):
    """
    Schema for the hallucination/groundedness evaluation verdict produced by the Critic node.
    """
    grounded: bool
    unsupported_claims: List[str]


class AgentState(TypedDict):
    """
    Represents the global state schema for Verity's LangGraph financial Q&A workflow.
    """

    # Field 1: User Query
    # Populated initially at graph entry with the raw financial query string submitted by the user.
    query: str

    # Field 2: Query Classification
    # Populated by the Router Node.
    # Allowed values: "simple_lookup" (factual retrieval), "comparison" (cross-company/year analysis), "numeric" (financial calculation/table analysis).
    classification: str

    # Field 3: Extracted Metadata Filters
    # Populated by the Router / Query Extractor Node.
    # Contains extracted metadata constraints for retrieval (e.g., {"company": "Apple", "fiscal_year": "2024"}).
    extracted_filters: Dict[str, Any]

    # Field 4: Retrieved Text Chunks
    # Populated by the Retriever Node.
    # List of hybrid RRF-ranked context chunks retrieved from Qdrant (dense) and BM25 (sparse) indices.
    retrieved_chunks: List[Dict[str, Any]]

    # Field 5: Draft Answer
    # Populated by the Generator Node.
    # Initial LLM-generated response synthesized from retrieved SEC filing text chunks.
    draft_answer: str

    # Field 6: Source Citations
    # Populated by the Generator / Citation Formatter Node.
    # Structured list of citation references supporting the draft answer (e.g. company, year, section, chunk_id, source_path).
    citations: List[Dict[str, Any]]

    # Field 7: Critic Evaluation Verdict
    # Populated by the Critic Node.
    # Dictionary evaluation containing {"grounded": bool, "unsupported_claims": list[str]} verifying factual alignment.
    critic_verdict: Dict[str, Any]

    # Field 7b: Initial Critic Evaluation Verdict
    # Populated by the Critic Node on its first evaluation pass before any retries.
    first_critic_verdict: Dict[str, Any]

    # Field 8: Retry Counter
    # Incremented by the Router / Conditional Feedback Node during self-correction loops when the Critic finds hallucinations.
    retry_count: int

    # Field 9: Final Grounded Answer
    # Populated by the Finalizer / Output Node.
    # The verified, grounded answer ready for downstream API streaming and client UI rendering.
    final_answer: str
