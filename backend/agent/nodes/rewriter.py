"""
Query Rewriter Node for Verity's Agentic LangGraph Workflow.

Role:
- Formulates multi-turn conversation follow-ups into standalone, context-complete search queries.
- Uses conversation_history (role/content pairs) + current user query.
- Employs get_llm_with_fallback for quota safety.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

from backend.agent.llm_client import get_llm_with_fallback
from backend.agent.state import AgentState

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Verity's conversation-aware query rewriter for financial SEC 10-K RAG.

Your task:
1. If the Follow-up Question relies on context from prior turns (e.g. pronoun references like "they", "it", "that company", "the other one", or incomplete phrasing like "what about revenue?"), REWRITE it into a complete, self-contained standalone search question containing explicit company names, fiscal years, and financial terms.
2. If the Follow-up Question is ALREADY fully self-contained and explicit, output it EXACTLY as written without changing it.
3. Output ONLY the rewritten standalone question. Do NOT include explanations or preambles.
"""

MODEL_REWRITER = os.getenv("REWRITER_MODEL", os.getenv("MODEL_REWRITER", os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite")))


def rewrite_query(state: AgentState) -> AgentState:
    """
    Rewrites user follow-up questions using prior conversation_history.
    Populates state['original_query'], state['rewritten_query'], and updates state['query'].
    """
    raw_query = state.get("query", "").strip()
    history = state.get("conversation_history") or []
    api_key = os.getenv("GOOGLE_API_KEY")

    logger.info(f"Query Rewriter Node: raw_query='{raw_query}', history count={len(history)}")

    state["original_query"] = raw_query

    # If no conversation history, pass query through unchanged
    if not history:
        state["rewritten_query"] = raw_query
        return state

    # Format history turns (last 6 messages max)
    history_lines = []
    for msg in history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    if not history_lines:
        state["rewritten_query"] = raw_query
        return state

    history_text = "\n".join(history_lines)
    user_prompt = f"Conversation History:\n{history_text}\n\nLatest User Follow-up Question: {raw_query}\n\nStandalone Question:"

    try:
        llm = get_llm_with_fallback(
            primary_model=MODEL_REWRITER,
            fallback_models=["gemini-2.5-flash-lite"],
            google_api_key=api_key or None,
            temperature=0.0,
            request_timeout=15.0,
            max_retries=2,
        )

        messages = [
            ("system", SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
        response = llm.invoke(messages)

        if isinstance(response.content, list):
            rewritten = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content]).strip()
        else:
            rewritten = str(response.content).strip()

        # Clean up quotes if wrapped by LLM
        if (rewritten.startswith('"') and rewritten.endswith('"')) or (rewritten.startswith("'") and rewritten.endswith("'")):
            rewritten = rewritten[1:-1].strip()

        if rewritten:
            logger.info(f"Query Rewriter: '{raw_query}' -> '{rewritten}'")
            state["rewritten_query"] = rewritten
            state["query"] = rewritten
        else:
            state["rewritten_query"] = raw_query

    except Exception as e:
        logger.warning(f"Query rewriter failed ({e}). Falling back to raw query.")
        state["rewritten_query"] = raw_query

    return state
