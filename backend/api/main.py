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
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        description="Financial user query (e.g. 'What was Microsoft's revenue in 2024?')",
        example="What was Microsoft's revenue in fiscal year 2024?",
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
   Description: Emitted as each graph node starts execution (e.g. classifier, retriever, draft, critic, finalizer).

2. event: "citation"
   Data: [{"chunk_id": str, "company": str, "fiscal_year": str, "item_section": str, ...}, ...]
   Description: Emitted after the draft node completes, containing structured citation metadata for referenced SEC filings.

3. event: "answer"
   Data: {"answer": "<draft_answer_text>"}
   Description: Emitted when the draft node completes with the synthesized draft text.

   * NOTE ON TOKEN STREAMING TRADEOFF *
   verity_graph executes structured LangGraph state operations where LLM responses are validated
   against retrieved SEC context and audited by a self-correcting Critic node before final output.
   Streaming token-by-token mid-node would bypass graph verification and groundedness validation.
   Therefore, we emit the complete draft answer as a single "answer" event rather than faking token streaming.

4. event: "critic_status"
   Data: {"grounded": bool, "unsupported_claims": list[str], "retry_count": int}
   Description: Emitted after the critic node evaluates draft groundedness. If grounded is False and retry_count < 2,
   the graph triggers a self-correction loop back to the retriever node.

5. event: "done"
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


@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> EventSourceResponse:
    """
    POST /chat endpoint that executes verity_graph in streaming mode and emits SSE events.
    Accepts JSON body: {"query": "<user_query>"}
    Returns a Server-Sent Events stream using sse-starlette's EventSourceResponse.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    async def event_generator():
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: AgentState = {
            "query": query,
            "classification": "",
            "extracted_filters": {},
            "retrieved_chunks": [],
            "draft_answer": "",
            "citations": [],
            "critic_verdict": {},
            "retry_count": 0,
            "final_answer": "",
        }

        # Emit initial status for first node
        yield {
            "event": "status",
            "data": json.dumps({"node": "classifier", "state": "running"}),
        }

        final_answer = ""

        try:
            async for output in verity_graph.astream(initial_state, config=config):
                for node_name, state_update in output.items():
                    logger.info(f"Node execution completed: {node_name}")

                    if node_name == "classifier":
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
                        }

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
