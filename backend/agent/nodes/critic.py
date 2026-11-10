"""
Critic Node for Verity's Agentic LangGraph Workflow.

Role:
- Evaluates the LLM-generated draft_answer against the retrieved_chunks.
- Checks whether every factual statement, figure, and claim is strictly supported by the context.
- Uses Gemini with Pydantic structured output to produce state['critic_verdict'] ({'grounded': bool, 'unsupported_claims': list[str]}).
- Populates state['first_critic_verdict'] on the first pass.
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
CRITIC_PROMPT = """You are a strict, objective factual audit Critic for financial AI responses.

Your task is to review the Draft Answer against the provided Source Context Chunks:

1. CHECK ONLY SUBSTANTIVE FACTUAL ASSERTIONS:
   - Focus exclusively on verifiable factual assertions: numbers, metrics, dates, named facts/entities, and causal or comparative statements.
   - Ignore introductory or framing phrases (e.g., "Based on the filings...", "According to the provided SEC filings...", "In summary...").
   - Ignore markdown headings, bold bullet labels (e.g., "**Net Sales:**", "- **Overview:**"), and citation markers (e.g., "[1]", "[Chunk 2]").

2. GROUNDING & VERBATIM EXTRACTION REQUIREMENTS:
   - Verify whether every factual assertion in the Draft Answer is explicitly backed by the Source Context Chunks.
   - If any factual assertion is missing support, contradictory, unverified, or hallucinated from outside knowledge:
     - Mark grounded = false.
     - unsupported_claims MUST be verbatim sentences or clauses from the answer that contain a specific factual assertion not supported by the context.
   - If all factual assertions are supported by the context (or if the answer states "I don't have enough information"):
     - Mark grounded = true.
     - unsupported_claims MUST be an empty list ([]).

CRITICAL RULES:
- unsupported_claims must be verbatim sentences or clauses extracted directly from the answer that contain an ungrounded factual assertion.
- If everything is supported, unsupported_claims MUST be an empty list ([]).
"""


class CriticVerdictOutput(BaseModel):
    """Pydantic model for structured critic evaluation output."""
    grounded: bool = Field(
        description="True if every factual assertion in the draft answer is fully supported by the source context, False otherwise."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Verbatim sentences or clauses from the answer that contain a specific factual assertion not supported by the context. Must be an empty list if everything is supported."
    )


# Model Configuration
# Critic node only performs structured verification (grounded: bool + unsupported_claims list),
# not open-ended generation, so the lighter model is sufficient and preserves free-tier quota for the draft node.
def get_critic_model() -> str:
    """Returns the critic model configured via CRITIC_MODEL (fallback MODEL_CRITIC)."""
    return os.getenv("CRITIC_MODEL", os.getenv("MODEL_CRITIC", "gemini-3.1-flash-lite"))


MODEL_CRITIC = get_critic_model()


def critique(state: AgentState) -> AgentState:
    """
    Evaluates draft_answer groundedness against retrieved_chunks using Gemini structured output.
    Populates state['critic_verdict'] and state['first_critic_verdict'] on the first pass.
    Does not modify retry_count.
    """
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    api_key = os.getenv("GOOGLE_API_KEY")

    # Format context for evaluation
    context_text = "\n\n".join(
        [f"[Chunk {c.get('id') or c.get('chunk_id')}]\n{c.get('text', '')}" for c in chunks]
    )

    model = get_critic_model()
    llm = get_llm_with_fallback(
        primary_model=model,
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
    from backend.agent.llm_client import get_last_model_used
    state["critic_model"] = get_last_model_used() or model

    unsupported = [c.strip() for c in (result.unsupported_claims or []) if c.strip()]
    grounded = bool(result.grounded and len(unsupported) == 0)
    if grounded:
        unsupported = []

    verdict_dict = {
        "grounded": grounded,
        "unsupported_claims": unsupported,
    }

    state["critic_verdict"] = verdict_dict

    if not state.get("first_critic_verdict"):
        # First evaluation pass
        state["first_critic_verdict"] = verdict_dict

    return state
