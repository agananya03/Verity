"""
Verity Agentic RAG Graph Workflow.

Builds and exports `verity_graph`, a LangGraph StateGraph combining query classification,
hybrid dense/sparse retrieval, citation-grounded draft generation, and a self-correcting critic loop.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agent.nodes.classifier import classify
from backend.agent.nodes.critic import critique
from backend.agent.nodes.draft import draft_answer
from backend.agent.nodes.retriever import retrieve_node
from backend.agent.state import AgentState

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()


def finalize_response(state: AgentState) -> AgentState:
    """Populates state['final_answer'] from the verified draft_answer."""
    state["final_answer"] = state.get("draft_answer", "")
    return state


def route_critic(state: AgentState) -> str:
    """
    Conditional routing function for the Critic node.

    Routes back to 'retriever' if critic_verdict['grounded'] is False and retry_count < 2,
    otherwise routes to 'finalizer'.
    """
    verdict = state.get("critic_verdict") or {}
    grounded = verdict.get("grounded", False)
    retry_count = state.get("retry_count", 0)
    unsupported_claims = verdict.get("unsupported_claims", [])

    if not grounded and retry_count < 2:
        if unsupported_claims:
            claims_str = " ".join(unsupported_claims)
            orig_query = state.get("query", "")
            state["query"] = f"{orig_query} Additional search context needed for: {claims_str}"
        logger.info(f"Critic flagged ungrounded claims. Triggering self-correction loop ({retry_count}/2 retries)...")
        return "retriever"

    logger.info(f"Critic approved response (or max retries reached: {retry_count}). Finalizing response.")
    return "finalizer"


# Construct LangGraph StateGraph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("classifier", classify)
builder.add_node("retriever", retrieve_node)
builder.add_node("draft", draft_answer)
builder.add_node("critic", critique)
builder.add_node("finalizer", finalize_response)

# Add Sequential Edges: START -> classifier -> retriever -> draft -> critic
builder.add_edge(START, "classifier")
builder.add_edge("classifier", "retriever")
builder.add_edge("retriever", "draft")
builder.add_edge("draft", "critic")

# Add Conditional Edge from critic
builder.add_conditional_edges(
    "critic",
    route_critic,
    {
        "retriever": "retriever",
        "finalizer": "finalizer",
    },
)

builder.add_edge("finalizer", END)

# Memory Checkpointer
memory = MemorySaver()

# Compile and export graph
verity_graph = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING VERITY LANGGRAPH WORKFLOW END-TO-END")
    print("=" * 80)

    test_queries = [
        "What was Apple's net sales in fiscal year 2024?",
        "What subscription-based services does Apple offer?",
        "According to the filing, what are the principal competitive factors important to the Company?",
    ]

    for idx, test_query in enumerate(test_queries, start=1):
        print(f"\n--- Test Query #{idx}: '{test_query}' ---")

        initial_state: AgentState = {
            "query": test_query,
            "classification": "",
            "extracted_filters": {},
            "retrieved_chunks": [],
            "draft_answer": "",
            "citations": [],
            "critic_verdict": {},
            "first_critic_verdict": {},
            "retry_count": 0,
            "final_answer": "",
        }

        config = {"configurable": {"thread_id": f"test_end_to_end_{idx}"}}

        final_output = verity_graph.invoke(initial_state, config=config)

        print(f"Query Classification:  {final_output.get('classification')}")
        print(f"Extracted Filters:     {final_output.get('extracted_filters')}")
        print(f"Total Retries:         {final_output.get('retry_count')}")
        print(f"Critic Verdict:        {final_output.get('critic_verdict')}")
        print("-" * 80)
        print(f"Final Answer:\n{final_output.get('final_answer')}\n")
        print("Citations:")
        for cite in final_output.get("citations", []):
            print(f"  - [{cite.get('chunk_id')}] {cite.get('company')} ({cite.get('fiscal_year')}) - {cite.get('item_section')}")
        print("=" * 80)

