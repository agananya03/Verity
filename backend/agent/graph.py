"""
Verity Agentic RAG Graph Workflow.

Builds and exports `verity_graph`, a LangGraph StateGraph combining query classification,
hybrid dense/sparse retrieval, citation-grounded draft generation, and a self-correcting critic loop.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from backend.agent.nodes.classifier import classify
from backend.agent.nodes.critic import critique
from backend.agent.nodes.draft import DECLINE_MESSAGE, draft_answer
from backend.agent.nodes.retriever import retrieve_node
from backend.agent.nodes.rewriter import rewrite_query
from backend.agent.state import AgentState

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

# Maximum self-correction retry cycles permitted before forced finalization
MAX_RETRIES = 2


def split_sentences(text: str) -> List[str]:
    """Splits a text paragraph into sentences while preserving trailing inline citations."""
    text = text.strip()
    if not text:
        return []
    pattern = re.compile(r"(.+?[.!?](?:\s*\[[^\]]+\])*)(?:\s+|$)", re.DOTALL)
    sentences = []
    last_idx = 0
    for m in pattern.finditer(text):
        sentences.append(m.group(1).strip())
        last_idx = m.end()
    trailing = text[last_idx:].strip()
    if trailing:
        sentences.append(trailing)
    return sentences or [text]


def filter_unsupported_sentences(text: str, unsupported_claims: List[str]) -> str:
    """
    Filters out sentences from the text that assert or contain any of the flagged unsupported claims.
    """
    if not text.strip() or not unsupported_claims:
        return text

    paragraphs = text.split("\n")
    kept_paragraphs = []

    for para in paragraphs:
        if not para.strip():
            continue
        sentences = split_sentences(para)
        kept_sentences = []
        for sentence in sentences:
            sent_str = sentence.strip()
            if not sent_str:
                continue
            has_flagged = False
            sent_clean = re.sub(r"\[(?:Chunk\s*)?\d+\]", "", sent_str).lower().strip()
            for claim in unsupported_claims:
                claim_clean = re.sub(r"\[(?:Chunk\s*)?\d+\]", "", claim).lower().strip().rstrip(".")
                if not claim_clean:
                    continue
                # Substring match in either direction
                if claim_clean in sent_clean or sent_clean in claim_clean:
                    has_flagged = True
                    break
                # All tokens of claim must be present in sentence
                claim_words = set(re.findall(r"\w+", claim_clean))
                sent_words = set(re.findall(r"\w+", sent_clean))
                if claim_words and claim_words.issubset(sent_words):
                    has_flagged = True
                    break

            if not has_flagged:
                kept_sentences.append(sent_str)

        if kept_sentences:
            kept_paragraphs.append(" ".join(kept_sentences))

    return "\n\n".join(kept_paragraphs).strip()


def finalize_response(state: AgentState) -> AgentState:
    """
    Finalizes the answer for API delivery and evaluation.
    Sets state['final_grounded'] and state['unsupported_claims_final'].
    If still ungrounded after max retries, removes sentences containing flagged claims.
    If removal leaves nothing, returns the helpful DECLINE_MESSAGE.
    """
    verdict = state.get("critic_verdict") or {}
    grounded = bool(verdict.get("grounded", False))
    unsupported = verdict.get("unsupported_claims", [])
    draft = state.get("draft_answer", "")

    if grounded:
        state["final_grounded"] = True
        state["unsupported_claims_final"] = []
        state["final_answer"] = draft
    else:
        state["final_grounded"] = False
        state["unsupported_claims_final"] = list(unsupported)

        cleaned_text = filter_unsupported_sentences(draft, unsupported)
        if not cleaned_text.strip():
            state["final_answer"] = DECLINE_MESSAGE
            state["is_decline"] = True
        else:
            state["final_answer"] = cleaned_text

    return state


def revise(state: AgentState) -> AgentState:
    """
    Revision node: increments retry_count and records critic's unsupported claims
    in state['critique_feedback'] for the draft node.
    """
    current_retries = state.get("retry_count", 0)
    state["retry_count"] = current_retries + 1

    verdict = state.get("critic_verdict") or {}
    unsupported_claims = verdict.get("unsupported_claims", [])
    state["critique_feedback"] = list(unsupported_claims)

    logger.info(
        f"Revise Node: incremented retry_count to {state['retry_count']}/{MAX_RETRIES}. "
        f"Stored critique_feedback: {state['critique_feedback']}"
    )
    return state


def route_critic(state: AgentState) -> str:
    """
    Pure conditional routing function for the Critic node.

    Returns 'revise' if not grounded and retry_count < MAX_RETRIES,
    otherwise returns 'finalizer'. Does NOT mutate state.
    """
    verdict = state.get("critic_verdict") or {}
    grounded = verdict.get("grounded", False)
    retry_count = state.get("retry_count", 0)

    if not grounded and retry_count < MAX_RETRIES:
        logger.info(
            f"Critic flagged ungrounded claims. Routing to revise ({retry_count}/{MAX_RETRIES} retries used)..."
        )
        return "revise"

    logger.info(
        f"Critic approved response or max retries reached ({retry_count}/{MAX_RETRIES}). Routing to finalizer."
    )
    return "finalizer"


def route_revise(state: AgentState) -> str:
    """
    Conditional routing function from the Revise node.

    Skips re-retrieval unless critique_feedback is non-empty,
    then retrieves once more with the original query only.
    """
    feedback = state.get("critique_feedback") or []
    retry_count = state.get("retry_count", 0)

    # Retrieve once more on first retry if critique feedback is non-empty, otherwise skip re-retrieval
    if feedback and retry_count <= 1:
        logger.info("Critique feedback non-empty on first retry. Retrieving once more with original query...")
        return "retriever"

    logger.info("Skipping re-retrieval. Routing revise directly to draft...")
    return "draft"


# Construct LangGraph StateGraph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("rewriter", rewrite_query)
builder.add_node("classifier", classify)
builder.add_node("retriever", retrieve_node)
builder.add_node("draft", draft_answer)
builder.add_node("critic", critique)
builder.add_node("revise", revise)
builder.add_node("finalizer", finalize_response)

# Add Sequential Edges: START -> rewriter -> classifier -> retriever -> draft -> critic
builder.add_edge(START, "rewriter")
builder.add_edge("rewriter", "classifier")
builder.add_edge("classifier", "retriever")
builder.add_edge("retriever", "draft")
builder.add_edge("draft", "critic")

# Add Conditional Edge from critic: route to revise or finalizer
builder.add_conditional_edges(
    "critic",
    route_critic,
    {
        "revise": "revise",
        "finalizer": "finalizer",
    },
)

# Add Edge from revise: skips re-retrieval unless critique_feedback is non-empty,
# then retrieves once more with the original query only
builder.add_conditional_edges(
    "revise",
    route_revise,
    {
        "retriever": "retriever",
        "draft": "draft",
    },
)

builder.add_edge("finalizer", END)

# Compile and export graph without checkpointer (stateless execution: history is provided per request)
verity_graph = builder.compile()


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
            "original_query": test_query,
            "rewritten_query": test_query,
            "conversation_history": [],
            "classification": "",
            "extracted_filters": {},
            "retrieved_chunks": [],
            "draft_answer": "",
            "citations": [],
            "critic_verdict": {},
            "first_critic_verdict": {},
            "critique_feedback": [],
            "retry_count": 0,
            "final_answer": "",
            "final_grounded": False,
            "unsupported_claims_final": [],
            "is_decline": False,
        }

        config = {"configurable": {"thread_id": f"test_end_to_end_{idx}"}}

        final_output = verity_graph.invoke(initial_state, config=config)

        print(f"Query Classification:      {final_output.get('classification')}")
        print(f"Extracted Filters:         {final_output.get('extracted_filters')}")
        print(f"Total Retries:             {final_output.get('retry_count')}")
        print(f"Critic Verdict:            {final_output.get('critic_verdict')}")
        print(f"Critique Feedback:         {final_output.get('critique_feedback')}")
        print(f"Final Grounded:            {final_output.get('final_grounded')}")
        print(f"Unsupported Claims Final:  {final_output.get('unsupported_claims_final')}")
        print("-" * 80)
        print(f"Final Answer:\n{final_output.get('final_answer')}\n")
        print("Citations:")
        for cite in final_output.get("citations", []):
            print(f"  - [{cite.get('chunk_id')}] {cite.get('company')} ({cite.get('fiscal_year')}) - {cite.get('item_section')}")
        print("=" * 80)

