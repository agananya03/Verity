"""
Draft Generator Node for Verity's Agentic LangGraph Workflow.

Role:
- Generates an initial draft answer to the user query based ONLY on retrieved_chunks.
- Requires the LLM to cite every factual claim with inline [chunk_id] markers matching the provided chunks.
- Instructs the LLM to state "I don't have enough information" if the chunks are insufficient.
- Populates state['draft_answer'] and extracts state['citations'] (list of referenced chunk_ids & metadata).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.agent.llm_client import get_llm_with_fallback
from backend.agent.state import AgentState

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

# Helpful Decline Message
DECLINE_MESSAGE = (
    "I don't have enough information to answer this question based on the retrieved SEC filings. "
    "Verity currently indexes SEC Form 10-K filings for Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), and NVIDIA (NVDA)."
)

# System Prompt for Draft Generation
DRAFT_PROMPT = """You are Verity's financial analyst AI. Your task is to answer the user's financial question accurately and concisely.

STRICT CONSTRAINTS:
1. Answer ONLY using the information provided in the Context Chunks below. Do NOT use outside knowledge or extrapolate.
2. Every factual statement or financial figure MUST be explicitly cited inline with its corresponding numeric chunk marker (e.g., "Microsoft revenue reached $245B in 2024 [1]").
3. If the provided Context Chunks do NOT contain sufficient information to answer the question, state explicitly: "I don't have enough information to answer this question based on the retrieved SEC filings. Verity currently indexes SEC Form 10-K filings for Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), and NVIDIA (NVDA)."
4. Be precise, professional, and directly address the user's question.

Context Chunks:
{context_text}
"""


# Model Configuration & Fallback Chain
# Draft node is the only node where full answer quality and open-ended financial synthesis matter,
# so it retains the stronger primary model with automatic fallback models if quota is exhausted.
MODEL_DRAFT = os.getenv("MODEL_DRAFT", "gemini-3.5-flash")
FALLBACK_DRAFT = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]

# Manifest mapping for SEC filing URLs
_manifest_url_map: Optional[Dict[str, str]] = None


def get_manifest_url_map() -> Dict[str, str]:
    """Builds a lookup mapping from local source_path (and filename) to canonical SEC source_url."""
    global _manifest_url_map
    if _manifest_url_map is not None:
        return _manifest_url_map

    _manifest_url_map = {}

    # 1. Try committed artifacts manifest
    artifacts_manifest = BASE_DIR / "artifacts" / "corpus_manifest.json"
    if artifacts_manifest.exists():
        try:
            with open(artifacts_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
                for filing in data.get("filings", []):
                    url = filing.get("source_url") or filing.get("filing_url")
                    path_str = filing.get("local_path")
                    if url and path_str:
                        _manifest_url_map[path_str] = url
                        _manifest_url_map[Path(path_str).name] = url
                        _manifest_url_map[Path(path_str).as_posix()] = url
        except Exception as e:
            logger.warning(f"Failed to load corpus_manifest.json in draft node: {e}")

    # 2. Try raw manifest fallback
    raw_manifest = BASE_DIR / "data" / "raw" / "manifest.json"
    if raw_manifest.exists():
        try:
            with open(raw_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
                for filing in data:
                    url = filing.get("source_url") or filing.get("filing_url")
                    path_str = filing.get("local_path")
                    if url and path_str:
                        _manifest_url_map[path_str] = url
                        _manifest_url_map[Path(path_str).name] = url
                        _manifest_url_map[Path(path_str).as_posix()] = url
        except Exception as e:
            logger.warning(f"Failed to load raw manifest.json in draft node: {e}")

    return _manifest_url_map


def draft_answer(state: AgentState) -> AgentState:
    """
    Generates a grounded draft answer with concise numeric citations [1] using Gemini.
    Populates state['draft_answer'], state['citations'], and state['is_decline'].
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    api_key = os.getenv("GOOGLE_API_KEY")

    if not chunks:
        state["draft_answer"] = DECLINE_MESSAGE
        state["citations"] = []
        state["is_decline"] = True
        return state

    # Format context chunks with clear numeric indices [Chunk 1], [Chunk 2]...
    context_blocks = []
    chunk_index_map: Dict[int, Dict] = {}
    for idx, c in enumerate(chunks, start=1):
        chunk_index_map[idx] = c
        header = f"[Chunk {idx}] (Company: {c.get('company')}, Year: {c.get('fiscal_year')}, Section: {c.get('item_section')})"
        context_blocks.append(f"{header}\nText: {c.get('text', '')}")

    context_text = "\n\n".join(context_blocks)
    prompt = DRAFT_PROMPT.format(context_text=context_text)

    llm = get_llm_with_fallback(
        primary_model=MODEL_DRAFT,
        fallback_models=FALLBACK_DRAFT,
        google_api_key=api_key or None,
        temperature=0.1,
        request_timeout=30.0,
        max_retries=2,
    )

    critique_feedback = state.get("critique_feedback") or []
    if critique_feedback:
        unsupported_list = "\n".join([f"- {claim}" for claim in critique_feedback])
        user_prompt = (
            f"Question: {query}\n\n"
            f"A reviewer found these claims unsupported:\n{unsupported_list}\n\n"
            f"Rewrite using only the context, remove unsupported claims."
        )
    else:
        user_prompt = f"Question: {query}"

    messages = [
        ("system", prompt),
        ("user", user_prompt),
    ]

    response = llm.invoke(messages)
    if isinstance(response.content, list):
        answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content]).strip()
    else:
        answer_text = str(response.content).strip()

    # Detect if answer is a decline/insufficient information response
    lower_answer = answer_text.lower()
    is_decline = bool(
        "don't have enough information" in lower_answer
        or "do not have enough information" in lower_answer
        or "insufficient information" in lower_answer
        or answer_text.strip() == DECLINE_MESSAGE
    )

    # Extract referenced numeric chunk markers e.g. [1], [2] or [Chunk 1]
    referenced_indices = set()
    for match in re.finditer(r'\[(?:Chunk\s*)?(\d+)\]', answer_text, re.IGNORECASE):
        referenced_indices.add(int(match.group(1)))

    url_map = get_manifest_url_map()
    citations = []
    for idx in sorted(referenced_indices):
        if idx in chunk_index_map:
            chunk = chunk_index_map[idx]
            c_id = chunk.get("id") or chunk.get("chunk_id", "")
            source_path = chunk.get("source_path", "")
            source_url = (
                url_map.get(source_path)
                or url_map.get(Path(source_path).name)
                or url_map.get(Path(source_path).as_posix())
                or ""
            ) if source_path else ""

            citations.append({
                "chunk_id": c_id,
                "citation_marker": f"[{idx}]",
                "company": chunk.get("company"),
                "fiscal_year": chunk.get("fiscal_year"),
                "item_section": chunk.get("item_section"),
                "item_title": chunk.get("item_title"),
                "source_path": source_path,
                "source_url": source_url,
            })

    state["draft_answer"] = answer_text
    state["citations"] = citations
    state["is_decline"] = is_decline

    logger.info(f"Draft Generator Node: is_decline={is_decline}")

    return state
