"""
Verity Financial RAG FastAPI Backend Server.

Provides REST and Server-Sent Events (SSE) streaming endpoints for:
- Interactive Agentic RAG chat with real-time graph node updates (/chat)
- Operational health checks (/health)
- Evaluation results aggregation (/eval-results)
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Load environment variables via python-dotenv at startup
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request

from backend.agent.graph import MAX_RETRIES, verity_graph
from backend.agent.llm_client import _active_model_cache
from backend.agent.retrieval import get_bm25_chunks_count, is_bm25_loaded, is_qdrant_connected
from backend.agent.state import AgentState


# -----------------------------------------------------------------------------
# Sensitive Data Redaction & Logging Filters (Never leak API keys in logs)
# -----------------------------------------------------------------------------
def redact_sensitive_info(text: str) -> str:
    """Scans text and redacts API keys and sensitive tokens."""
    if not isinstance(text, str) or not text:
        return text
    # Mask exact values of configured sensitive keys
    for key in ("GOOGLE_API_KEY", "QDRANT_API_KEY"):
        val = os.getenv(key)
        if val and len(val) >= 6:
            text = text.replace(val, f"{val[:3]}...[REDACTED]")
    # Mask Google API keys (AIza...)
    text = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "AIza...[REDACTED]", text)
    # Mask key=..., api_key=..., token=... query parameters
    text = re.sub(r"([?&](?:key|api_key|token)=)[^&\s'\"]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    # Mask Authorization headers / bearer tokens
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    return text


class RedactingFilter(logging.Filter):
    """Logging filter that ensures API keys and secrets are never emitted to log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact_sensitive_info(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: redact_sensitive_info(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        redact_sensitive_info(arg) if isinstance(arg, str) else arg
                        for arg in record.args
                    )
        except Exception:
            pass
        return True


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.addFilter(RedactingFilter())
logging.getLogger().addFilter(RedactingFilter())

# Startup system check logging
logger.info("=" * 60)
logger.info("VERITY BACKEND STARTUP SYSTEM STATUS:")
logger.info(f"  - BM25 Index Loaded: {is_bm25_loaded()} ({get_bm25_chunks_count()} chunks)")
logger.info(f"  - Qdrant Vector DB:  {'CONNECTED' if is_qdrant_connected() else 'NOT CONNECTED'}")
logger.info(f"  - Embedding Backend: {os.getenv('EMBEDDING_BACKEND', 'fastembed')}")
logger.info("=" * 60)

# Initialize Rate Limiter (per-IP, 10 requests/minute on /chat)
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI App
app = FastAPI(
    title="Verity Financial RAG API",
    description="Agentic Financial RAG system with self-correcting SEC filing retrieval.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS Middleware
# Credentials are not needed, so allow_credentials=False avoids wildcard origin conflicts
allowed_origin_env = os.getenv("ALLOWED_ORIGIN", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origin_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request Schemas for POST /chat (Strict Validation)
# -----------------------------------------------------------------------------
class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Role of the history message participant ('user' or 'assistant').",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Content of the previous message (max 2000 characters).",
    )


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Financial user query (max 500 characters).",
        json_schema_extra={"example": "What was Microsoft's revenue in fiscal year 2024?"},
    )
    history: List[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=6,
        description="Recent conversation history turns (max 6 messages).",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query string cannot be empty or whitespace only.")
        if len(stripped) > 500:
            raise ValueError("Query string cannot exceed 500 characters.")
        return stripped


# -----------------------------------------------------------------------------
# SSE Event Contract Explanation for Frontend Integration
# -----------------------------------------------------------------------------
"""
SSE EVENT CONTRACT SPECIFICATION:
=================================
The POST /chat endpoint streams Server-Sent Events (SSE) as verity_graph progresses through its nodes.

Frontend Event Listener Types:

1. event: "status"
   Data: {"node": "<node_name>", "state": "running"}
   Description: Emitted as each graph node starts execution (e.g. rewriter, classifier, retriever, draft, critic, revise, finalizer).

2. event: "rewritten_query"
   Data: {"original_query": str, "rewritten_query": str}
   Description: Emitted if the rewriter node reformulates a multi-turn follow-up query.

3. event: "citation"
   Data: [{"chunk_id": str, "company": str, "fiscal_year": str, "item_section": str, ...}, ...]
   Description: Emitted after the draft node completes, containing structured citation metadata for referenced SEC filings.

4. event: "answer"
   Data: {"answer": "<draft_answer_text>"}
   Description: Emitted when the draft node completes with the synthesized draft text.

5. event: "critic_status"
   Data: {"grounded": bool, "unsupported_claims": list[str], "retry_count": int, "is_decline": bool}
   Description: Emitted after the critic node evaluates draft groundedness.

6. event: "done"
   Data: {"final_answer": "<final_verified_answer>", "final_grounded": bool, "unsupported_claims_final": list[str], "status": "completed", "request_id": str}
   Description: Emitted when the graph completes execution and finalizer finishes.

7. event: "error"
   Data: {"error": "<generic_error_message>", "request_id": str}
   Description: Emitted if execution fails or times out (never leaks raw exception details).
"""


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint returning operational status."""
    return {
        "status": "ok",
        "bm25_loaded": is_bm25_loaded(),
        "qdrant_connected": is_qdrant_connected(),
        "chunks": get_bm25_chunks_count(),
    }


@app.get("/eval-results")
async def get_eval_results() -> Dict[str, Any]:
    """
    Reads and returns the JSON contents of backend/eval/results/baseline.json
    and backend/eval/results/improved.json combined.
    """
    eval_dir = BASE_DIR / "eval" / "results"
    baseline_path = eval_dir / "baseline.json"
    improved_path = eval_dir / "improved.json"

    baseline_data: Any = {}
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading baseline.json: {e}")

    improved_data: Any = {}
    if improved_path.exists():
        try:
            with open(improved_path, "r", encoding="utf-8") as f:
                improved_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading improved.json: {e}")

    return {
        "baseline": baseline_data,
        "improved": improved_data,
    }


@app.get("/category-breakdown")
async def get_category_breakdown() -> Dict[str, Any]:
    """
    Reads and returns the contents of backend/eval/results/category_breakdown.json.
    """
    breakdown_path = BASE_DIR / "eval" / "results" / "category_breakdown.json"
    if not breakdown_path.exists():
        raise HTTPException(status_code=404, detail="category_breakdown.json not found")
    try:
        with open(breakdown_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading category_breakdown.json: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corpus-info")
async def get_corpus_info() -> Dict[str, Any]:
    """
    Returns metadata about indexed companies, fiscal years covered per company,
    filing type, and total chunk count.
    Prioritizes committed backend/artifacts/corpus_manifest.json for production reliability,
    falling back to backend/data/raw/manifest.json if present.
    """
    artifacts_manifest = BASE_DIR / "artifacts" / "corpus_manifest.json"
    if artifacts_manifest.exists():
        try:
            with open(artifacts_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "companies": data.get("companies", []),
                    "filing_type": data.get("filing_type", "10-K"),
                    "total_chunks": data.get("total_chunks", 0),
                }
        except Exception as e:
            logger.warning(f"Failed to read corpus_manifest.json artifact: {e}")

    # Fallback to local raw manifest
    manifest_path = BASE_DIR / "data" / "raw" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Corpus manifest not found.")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading manifest.json: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"
    total_chunks = 0
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                total_chunks = len(chunks_data)
        except Exception as e:
            logger.warning(f"Could not load chunks.json for chunk count: {e}")

    company_map: Dict[str, Dict[str, Any]] = {}
    for entry in manifest_data:
        ticker = entry.get("ticker", "UNKNOWN")
        name = entry.get("company", ticker)
        fiscal_year = entry.get("fiscal_year")
        
        if ticker not in company_map:
            company_map[ticker] = {
                "ticker": ticker,
                "name": name,
                "fiscal_years": set(),
            }
        if fiscal_year:
            company_map[ticker]["fiscal_years"].add(fiscal_year)

    companies_list = []
    for ticker, info in company_map.items():
        companies_list.append({
            "ticker": info["ticker"],
            "name": info["name"],
            "fiscal_years": sorted(list(info["fiscal_years"])),
        })

    return {
        "companies": companies_list,
        "filing_type": "10-K",
        "total_chunks": total_chunks,
    }


@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest) -> EventSourceResponse:
    """
    POST /chat endpoint that executes verity_graph in streaming mode and emits SSE events.
    Enforces per-IP rate limiting (10 req/minute), 60s timeout, and structured logging.
    """
    query = chat_request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    history_payload = [{"role": h.role, "content": h.content} for h in chat_request.history]
    request_id = str(uuid.uuid4())

    logger.info(
        "[%s] POST /chat received query='%s', history_count=%d",
        request_id,
        redact_sensitive_info(query),
        len(history_payload),
    )

    async def event_generator():
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        start_time = time.time()

        initial_state: AgentState = {
            "query": query,
            "original_query": query,
            "rewritten_query": query,
            "conversation_history": history_payload,
            "classification": "",
            "extracted_filters": {},
            "retrieved_chunks": [],
            "draft_answer": "",
            "citations": [],
            "critic_verdict": {},
            "critique_feedback": [],
            "retry_count": 0,
            "final_answer": "",
            "final_grounded": False,
            "unsupported_claims_final": [],
            "is_decline": False,
        }

        classification = "unknown"
        retry_count = 0
        final_grounded = False
        unsupported_claims_final: List[str] = []
        is_decline = False
        models_used: set = set()

        # Emit initial status for first node
        yield {
            "event": "status",
            "data": json.dumps({"node": "rewriter", "state": "running"}),
        }

        try:
            # Enforce 60-second execution timeout around graph processing
            async with asyncio.timeout(60.0):
                async for output in verity_graph.astream(initial_state, config=config):
                    for node_name, state_update in output.items():
                        logger.info("[%s] Node execution completed: %s", request_id, node_name)

                        if "classification" in state_update and state_update["classification"]:
                            classification = state_update["classification"]
                        if "retry_count" in state_update:
                            retry_count = state_update["retry_count"]
                        if "is_decline" in state_update:
                            is_decline = state_update["is_decline"]
                        if "final_grounded" in state_update:
                            final_grounded = state_update["final_grounded"]
                        if "unsupported_claims_final" in state_update:
                            unsupported_claims_final = state_update["unsupported_claims_final"]

                        if node_name == "rewriter":
                            orig = state_update.get("original_query", "")
                            rewritten = state_update.get("rewritten_query", "")
                            if rewritten and rewritten.lower() != orig.lower():
                                models_used.add(os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite"))
                                yield {
                                    "event": "rewritten_query",
                                    "data": json.dumps({"original_query": orig, "rewritten_query": rewritten}),
                                }
                            yield {
                                "event": "status",
                                "data": json.dumps({"node": "classifier", "state": "running"}),
                            }

                        elif node_name == "classifier":
                            models_used.add(os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite"))
                            yield {
                                "event": "status",
                                "data": json.dumps({"node": "retriever", "state": "running"}),
                            }

                        elif node_name == "retriever":
                            yield {
                                "event": "status",
                                "data": json.dumps({"node": "draft", "state": "running"}),
                            }

                        elif node_name == "draft":
                            draft_model = _active_model_cache.get(
                                os.getenv("MODEL_DRAFT", "gemini-3.5-flash"),
                                os.getenv("MODEL_DRAFT", "gemini-3.5-flash"),
                            )
                            models_used.add(draft_model)
                            citations = state_update.get("citations", [])
                            draft_ans = state_update.get("draft_answer", "")

                            if citations:
                                yield {
                                    "event": "citation",
                                    "data": json.dumps(citations),
                                }

                            if draft_ans:
                                yield {
                                    "event": "answer",
                                    "data": json.dumps({"answer": draft_ans}),
                                }

                            yield {
                                "event": "status",
                                "data": json.dumps({"node": "critic", "state": "running"}),
                            }

                        elif node_name == "critic":
                            models_used.add(os.getenv("MODEL_CRITIC", "gemini-3.5-flash-lite"))
                            verdict = state_update.get("critic_verdict", {})
                            r_count = state_update.get("retry_count", retry_count)
                            retry_count = r_count

                            critic_payload = {
                                "grounded": verdict.get("grounded", False),
                                "unsupported_claims": verdict.get("unsupported_claims", []),
                                "retry_count": retry_count,
                                "is_decline": is_decline,
                            }

                            yield {
                                "event": "critic_status",
                                "data": json.dumps(critic_payload),
                            }

                            grounded = verdict.get("grounded", False)
                            if not grounded and retry_count < MAX_RETRIES:
                                yield {
                                    "event": "status",
                                    "data": json.dumps({"node": "revise", "state": "running"}),
                                }
                            else:
                                yield {
                                    "event": "status",
                                    "data": json.dumps({"node": "finalizer", "state": "running"}),
                                }

                        elif node_name == "revise":
                            feedback = state_update.get("critique_feedback") or []
                            r_count = state_update.get("retry_count", retry_count)
                            retry_count = r_count
                            if feedback and r_count <= 1:
                                yield {
                                    "event": "status",
                                    "data": json.dumps({"node": "retriever", "state": "running"}),
                                }
                            else:
                                yield {
                                    "event": "status",
                                    "data": json.dumps({"node": "draft", "state": "running"}),
                                }

                        elif node_name == "finalizer":
                            final_answer = state_update.get("final_answer", "")
                            final_grounded = state_update.get("final_grounded", False)
                            unsupported_claims_final = state_update.get("unsupported_claims_final", [])

            # Compute request latency
            total_latency = time.time() - start_time

            # Structured logging per successful request
            logger.info(
                "REQUEST_SUMMARY: %s",
                json.dumps({
                    "request_id": request_id,
                    "classification": classification,
                    "retry_count": retry_count,
                    "final_grounded": final_grounded,
                    "total_latency": round(total_latency, 3),
                    "models_used": sorted(list(models_used)),
                    "status": "completed",
                }),
            )

            # Emit final done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "final_answer": final_answer,
                    "final_grounded": final_grounded,
                    "unsupported_claims_final": unsupported_claims_final,
                    "status": "completed",
                    "request_id": request_id,
                }),
            }

        except TimeoutError:
            total_latency = time.time() - start_time
            logger.error(
                "[%s] Request timed out after 60s during verity_graph execution. Total latency: %.2fs",
                request_id,
                total_latency,
            )
            logger.info(
                "REQUEST_SUMMARY: %s",
                json.dumps({
                    "request_id": request_id,
                    "classification": classification,
                    "retry_count": retry_count,
                    "final_grounded": final_grounded,
                    "total_latency": round(total_latency, 3),
                    "models_used": sorted(list(models_used)),
                    "status": "timeout",
                }),
            )
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "The request timed out while generating a response. Please try again.",
                    "request_id": request_id,
                }),
            }

        except Exception as e:
            total_latency = time.time() - start_time
            logger.error(
                "[%s] Error during verity_graph streaming execution: %s",
                request_id,
                redact_sensitive_info(str(e)),
                exc_info=True,
            )
            logger.info(
                "REQUEST_SUMMARY: %s",
                json.dumps({
                    "request_id": request_id,
                    "classification": classification,
                    "retry_count": retry_count,
                    "final_grounded": final_grounded,
                    "total_latency": round(total_latency, 3),
                    "models_used": sorted(list(models_used)),
                    "status": "error",
                }),
            )
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "An internal error occurred while processing your request. Please try again.",
                    "request_id": request_id,
                }),
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
