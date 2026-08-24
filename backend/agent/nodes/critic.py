"""
Critic Node for Verity's Agentic LangGraph Workflow.

Role:
- Evaluates the LLM-generated draft_answer against the retrieved_chunks.
- Checks whether every factual statement, figure, and claim is strictly supported by the context.
- Uses Gemini with Pydantic structured output to produce state['critic_verdict'] ({'grounded': bool, 'unsupported_claims': list[str]}).
- Increments state['retry_count'].
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from backend.agent.llm_client import get_llm_with_fallback
from backend.agent.state import AgentState

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

# System Prompt for Critic Evaluation
CRITIC_PROMPT = """You are a strict, objective audit Critic for financial AI responses.

Your task is to review the Draft Answer against the provided Source Context Chunks:
1. Verify if EVERY factual statement, financial number, date, and claim made in the Draft Answer is explicitly backed by the Source Context Chunks.
2. If any claim is missing support, contradictory, unverified, or hallucinated from outside knowledge, mark grounded = false and list the specific unsupported claims.
3. If the Draft Answer correctly states "I don't have enough information", mark grounded = true.

Be extremely rigorous. Financial accuracy requires zero ungrounded assertions.
"""


class CriticVerdictOutput(BaseModel):
    """Pydantic model for structured critic evaluation output."""
    grounded: bool = Field(
        description="True if every claim in the draft answer is fully supported by source context, False otherwise."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of specific sentences, numbers, or claims in the draft answer not supported by source context."
    )


# Model Configuration
# Critic node only performs structured verification (grounded: bool + unsupported_claims list),
# not open-ended generation, so the lighter model is sufficient and preserves free-tier quota for the draft node.
MODEL_CRITIC = os.getenv("MODEL_CRITIC", "gemini-3.5-flash-lite")


def critique(state: AgentState) -> AgentState:
    """
    Evaluates draft_answer groundedness against retrieved_chunks using Gemini structured output.
    Populates state['critic_verdict'] and increments state['retry_count'].
    """
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    api_key = os.getenv("GOOGLE_API_KEY")

    # Increment retry count
    current_retries = state.get("retry_count", 0)
    state["retry_count"] = current_retries + 1

    # Format context for evaluation
    context_text = "\n\n".join(
        [f"[Chunk {c.get('id') or c.get('chunk_id')}]\n{c.get('text', '')}" for c in chunks]
    )

    llm = get_llm_with_fallback(
        primary_model=MODEL_CRITIC,
        fallback_models=[],
        google_api_key=api_key or None,
        temperature=0.0,
        request_timeout=30.0,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(CriticVerdictOutput)

    user_message = f"""Source Context Chunks:
{context_text}

Draft Answer to evaluate:
{draft}
"""

    messages = [
        ("system", CRITIC_PROMPT),
        ("user", user_message),
    ]

    result: CriticVerdictOutput = structured_llm.invoke(messages)

    verdict_dict = {
        "grounded": result.grounded,
        "unsupported_claims": result.unsupported_claims,
    }

    state["critic_verdict"] = verdict_dict

    if not state.get("first_critic_verdict"):
        # First evaluation pass
        state["first_critic_verdict"] = verdict_dict
        state["retry_count"] = 0
    else:
        # Retry evaluation pass
        current_retries = state.get("retry_count", 0)
        state["retry_count"] = current_retries + 1

    return state
