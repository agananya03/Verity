"""
Draft Generator Node for Verity's Agentic LangGraph Workflow.

Role:
- Generates an initial draft answer to the user query based ONLY on retrieved_chunks.
- Requires the LLM to cite every factual claim with inline [chunk_id] markers matching the provided chunks.
- Instructs the LLM to state "I don't have enough information" if the chunks are insufficient.
- Populates state['draft_answer'] and extracts state['citations'] (list of referenced chunk_ids & metadata).
"""

import os
import re
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.agent.state import AgentState

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

# System Prompt for Draft Generation
DRAFT_PROMPT = """You are Verity's financial analyst AI. Your task is to answer the user's financial question accurately and concisely.

STRICT CONSTRAINTS:
1. Answer ONLY using the information provided in the Context Chunks below. Do NOT use outside knowledge or extrapolate.
2. Every factual statement or financial figure MUST be explicitly cited inline with its corresponding `[chunk_id]` marker (e.g., "Microsoft revenue reached $245B in 2024 [0004150a-99c9-4f57-8bce-fec2f0ee5c89]").
3. If the provided Context Chunks do NOT contain sufficient information to answer the question, state explicitly: "I don't have enough information to answer this question based on the retrieved SEC filings."
4. Be precise, professional, and directly address the user's question.

Context Chunks:
{context_text}
"""


def draft_answer(state: AgentState) -> AgentState:
    """
    Generates a grounded draft answer with inline chunk_id citations using Gemini.
    Populates state['draft_answer'] and state['citations'].
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    api_key = os.getenv("GOOGLE_API_KEY")

    if not chunks:
        state["draft_answer"] = "I don't have enough information to answer this question based on the retrieved SEC filings."
        state["citations"] = []
        return state

    # Format context chunks for the prompt
    context_blocks = []
    chunk_map: Dict[str, Dict] = {}
    for c in chunks:
        c_id = c.get("id") or c.get("chunk_id", "")
        chunk_map[str(c_id)] = c
        header = f"[chunk_id: {c_id}] (Company: {c.get('company')}, Year: {c.get('fiscal_year')}, Section: {c.get('item_section')})"
        context_blocks.append(f"{header}\nText: {c.get('text', '')}")

    context_text = "\n\n".join(context_blocks)
    prompt = DRAFT_PROMPT.format(context_text=context_text)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key or None,
        temperature=0.1,
    )

    messages = [
        ("system", prompt),
        ("user", f"Question: {query}"),
    ]

    response = llm.invoke(messages)
    if isinstance(response.content, list):
        answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content]).strip()
    else:
        answer_text = str(response.content).strip()

    # Extract referenced chunk_ids from draft_answer
    referenced_ids = set(re.findall(r'\[([a-f0-9\-]{8,36})\]', answer_text, re.IGNORECASE))

    # Also check if any known chunk_id from retrieved_chunks appears in the text
    for c_id in chunk_map:
        if c_id in answer_text:
            referenced_ids.add(c_id)

    citations = []
    for c_id in referenced_ids:
        if c_id in chunk_map:
            chunk = chunk_map[c_id]
            citations.append({
                "chunk_id": c_id,
                "company": chunk.get("company"),
                "fiscal_year": chunk.get("fiscal_year"),
                "item_section": chunk.get("item_section"),
                "item_title": chunk.get("item_title"),
                "source_path": chunk.get("source_path"),
            })

    state["draft_answer"] = answer_text
    state["citations"] = citations

    return state
