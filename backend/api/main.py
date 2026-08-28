"""
Verity Financial RAG FastAPI Backend Server.

Provides REST and Server-Sent Events (SSE) streaming endpoints for:
- Interactive Agentic RAG chat with real-time graph node updates (/chat)
- Operational health checks (/health)
- Evaluation results aggregation (/eval-results)
"""

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

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
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.agent.graph import verity_graph
from backend.agent.state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Verity Financial RAG API",
    description="Agentic Financial RAG system with self-correcting SEC filing retrieval.",
    version="1.0.0",
)

# Configure CORS Middleware
# Allows origins specified in ALLOWED_ORIGIN env var, default: http://localhost:3000
allowed_origin_env = os.getenv("ALLOWED_ORIGIN", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origin_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema for POST /chat
# Request schema for POST /chat
class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        description="Financial user query (e.g. 'What was Microsoft's revenue in 2024?')",
        example="What was Microsoft's revenue in fiscal year 2024?",
    )
    history: List[ChatHistoryMessage] = Field(
        default_factory=list,
        description="Recent conversation history turns for multi-turn query rewriting",
    )


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
   Description: Emitted as each graph node starts execution (e.g. rewriter, classifier, retriever, draft, critic, finalizer).

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
   Data: {"final_answer": "<final_verified_answer>", "status": "completed"}
   Description: Emitted when the graph completes execution and finalizer finishes.
"""


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint returning operational status."""
    return {"status": "ok"}


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
    Reads the ingestion manifest (backend/data/raw/manifest.json) and processed chunks (backend/data/processed/chunks.json),
    returning metadata about indexed companies, fiscal years covered per company, filing type, and total chunk count.
    """
    manifest_path = BASE_DIR / "data" / "raw" / "manifest.json"
    chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Ingestion manifest (manifest.json) not found")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_entries = json.load(f)
    except Exception as e:
        logger.error(f"Error reading manifest.json: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read manifest.json: {str(e)}")

    total_chunks = 0
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                total_chunks = len(chunks_data)
        except Exception as e:
            logger.warning(f"Could not load chunks.json for chunk count: {e}")

    # Group entries by company (preserving company order of appearance)
    company_map: Dict[str, Dict[str, Any]] = {}
    for entry in manifest_entries:
        ticker = entry.get("ticker", "")
        name = entry.get("company", "")
        fiscal_year = str(entry.get("fiscal_year", ""))
        
        if not ticker:
            continue
            
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
async def chat_endpoint(request: ChatRequest) -> EventSourceResponse:
    """
    POST /chat endpoint that executes verity_graph in streaming mode and emits SSE events.
    Accepts JSON body: {"query": "<user_query>", "history": [{"role": "user", "content": "..."}, ...]}
    Returns a Server-Sent Events stream using sse-starlette's EventSourceResponse.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    history_payload = [{"role": h.role, "content": h.content} for h in request.history]

    logger.info(f"POST /chat endpoint received query='{query}', history count={len(history_payload)}, history={history_payload}")

    async def event_generator():
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

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
            "retry_count": 0,
            "final_answer": "",
            "is_decline": False,
        }

        # Emit initial status for first node
        yield {
            "event": "status",
            "data": json.dumps({"node": "rewriter", "state": "running"}),
        }

        final_answer = ""
        is_decline = False

        try:
            async for output in verity_graph.astream(initial_state, config=config):
                for node_name, state_update in output.items():
                    logger.info(f"Node execution completed: {node_name}")

                    if "is_decline" in state_update:
                        is_decline = state_update["is_decline"]

                    if node_name == "rewriter":
                        # Rewriter node finished -> Emit rewritten_query if modified, then signal classifier
                        orig = state_update.get("original_query", "")
                        rewritten = state_update.get("rewritten_query", "")
                        if rewritten and rewritten.lower() != orig.lower():
                            yield {
                                "event": "rewritten_query",
                                "data": json.dumps({"original_query": orig, "rewritten_query": rewritten}),
                            }
                        yield {
                            "event": "status",
                            "data": json.dumps({"node": "classifier", "state": "running"}),
                        }

                    elif node_name == "classifier":
                        # Classifier node finished -> Next node is retriever
                        yield {
                            "event": "status",
                            "data": json.dumps({"node": "retriever", "state": "running"}),
                        }

                    elif node_name == "retriever":
                        # Retriever node finished -> Next node is draft
                        yield {
                            "event": "status",
                            "data": json.dumps({"node": "draft", "state": "running"}),
                        }

                    elif node_name == "draft":
                        # Draft node finished -> Emit citations and answer, then signal critic is starting
                        citations = state_update.get("citations", [])
                        draft_ans = state_update.get("draft_answer", "")

                        if citations:
                            yield {
                                "event": "citation",
                                "data": json.dumps(citations),
                            }

                        if draft_ans:
                            # Honest trade-off: Emitting complete draft as single 'answer' event
                            yield {
                                "event": "answer",
                                "data": json.dumps({"answer": draft_ans}),
                            }

                        yield {
                            "event": "status",
                            "data": json.dumps({"node": "critic", "state": "running"}),
                        }

                    elif node_name == "critic":
                        # Critic node finished -> Emit critic_status
                        verdict = state_update.get("critic_verdict", {})
                        retry_count = state_update.get("retry_count", 0)

                        critic_payload = {
                            "grounded": verdict.get("grounded", False),
                            "unsupported_claims": verdict.get("unsupported_claims", []),
                            "retry_count": retry_count,
                            "is_decline": is_decline,
                        }

                        logger.info(f"Emitting critic_status SSE event payload: {critic_payload}")

                        yield {
                            "event": "critic_status",
                            "data": json.dumps(critic_payload),
                        }

                        # Check if routing back to retriever for self-correction or to finalizer
                        grounded = verdict.get("grounded", False)
                        if not grounded and retry_count < 2:
                            yield {
                                "event": "status",
                                "data": json.dumps({"node": "retriever", "state": "running"}),
                            }
                        else:
                            yield {
                                "event": "status",
                                "data": json.dumps({"node": "finalizer", "state": "running"}),
                            }

                    elif node_name == "finalizer":
                        final_answer = state_update.get("final_answer", "")

            # Emit final done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "final_answer": final_answer,
                    "status": "completed",
                }),
            }

        except Exception as e:
            logger.error(f"Error during verity_graph streaming execution: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
