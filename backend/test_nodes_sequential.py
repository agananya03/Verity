"""
Scratch script to sequentially test and verify all Verity agent node functions.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent.state import AgentState
from backend.agent.nodes.classifier import classify
from backend.agent.nodes.retriever import retrieve_node
from backend.agent.nodes.draft import draft_answer
from backend.agent.nodes.critic import critique


def main():
    print("=" * 80)
    print("STARTING SEQUENTIAL AGENT NODE VERIFICATION TEST")
    print("=" * 80)

    # Initial test state
    state: AgentState = {
        "query": "What was Microsoft's revenue in fiscal year 2024?",
        "classification": "",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "retry_count": 0,
        "final_answer": "",
    }

    print(f"\nInitial Query: '{state['query']}'\n")

    # Step 1: Classifier
    print("=" * 80)
    print("STEP 1: CLASSIFIER NODE")
    print("=" * 80)
    state = classify(state)
    print(f"  Result -> Classification: '{state['classification']}'")
    print(f"  Result -> Extracted Filters: {state['extracted_filters']}\n")

    # Step 2: Retriever
    print("=" * 80)
    print("STEP 2: RETRIEVER NODE")
    print("=" * 80)
    state = retrieve_node(state)
    chunks = state.get("retrieved_chunks", [])
    print(f"  Result -> Retrieved Chunks Count: {len(chunks)}")
    if chunks:
        top_chunk = chunks[0]
        snippet = top_chunk.get("text", "").replace("\n", " ")[:150]
        print(f"  Top Chunk ID:      {top_chunk.get('id')}")
        print(f"  Top Chunk Company: {top_chunk.get('company')} ({top_chunk.get('fiscal_year')})")
        print(f"  Top Chunk Section: {top_chunk.get('item_section')}")
        print(f"  Top Chunk Snippet: {snippet}...\n")

    # Step 3: Draft Generator
    print("=" * 80)
    print("STEP 3: DRAFT GENERATOR NODE")
    print("=" * 80)
    state = draft_answer(state)
    print(f"  Result -> Draft Answer:\n{state['draft_answer']}\n")
    print(f"  Result -> Citations Count: {len(state['citations'])}")
    for cite in state["citations"]:
        print(f"    - [{cite.get('chunk_id')}] {cite.get('company')} ({cite.get('fiscal_year')}) - {cite.get('item_section')}")
    print("\n")

    # Step 4: Critic
    print("=" * 80)
    print("STEP 4: CRITIC NODE")
    print("=" * 80)
    state = critique(state)
    print(f"  Result -> Critic Verdict: {state['critic_verdict']}")
    print(f"  Result -> Retry Count:    {state['retry_count']}\n")

    print("=" * 80)
    print("ALL NODE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
