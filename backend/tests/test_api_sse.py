"""
Unit tests for FastAPI SSE streaming endpoint (/chat) in backend/api/main.py.

Tests:
- Uses FastAPI TestClient with a mocked graph astream.
- Asserts strict SSE event order (status -> citation -> answer -> critic_status -> done).
- Asserts that the 'done' event contains final_answer, final_grounded, unsupported_claims_final,
  status, and request_id fields.
- Verifies input validation rejects queries exceeding 500 characters or invalid history roles.
"""

import json
import pytest
from starlette.testclient import TestClient
from backend.api.main import app


def parse_sse_events(raw_sse_text: str):
    """Parses raw text/event-stream response body into a list of (event_type, json_data) tuples."""
    events = []
    lines = raw_sse_text.strip().split("\n")
    current_event = "message"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.replace("event:", "").strip()
        elif line.startswith("data:"):
            data_str = line.replace("data:", "").strip()
            try:
                data_json = json.loads(data_str)
                events.append((current_event, data_json))
            except Exception:
                events.append((current_event, data_str))
    return events


def test_sse_endpoint_streaming_order(monkeypatch):
    """
    Mocks verity_graph.astream to simulate standard node execution and verifies
    that SSE events are emitted in the expected order with the required fields.
    """
    mock_citations = [
        {
            "chunk_id": "chunk_1",
            "company": "Apple",
            "fiscal_year": "2024",
            "item_section": "Item 7",
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/sample.htm",
        }
    ]

    async def mock_astream(initial_state, config=None):
        # 1. Rewriter
        yield {"rewriter": {"original_query": initial_state["query"], "rewritten_query": initial_state["query"]}}
        # 2. Classifier
        yield {"classifier": {"classification": "simple_lookup", "extracted_filters": {"company": "Apple"}}}
        # 3. Retriever
        yield {"retriever": {"retrieved_chunks": mock_citations}}
        # 4. Draft
        yield {"draft": {"draft_answer": "Apple revenue was $391B in 2024. [1]", "citations": mock_citations}}
        # 5. Critic
        yield {"critic": {"critic_verdict": {"grounded": True, "unsupported_claims": []}, "retry_count": 0}}
        # 6. Finalizer
        yield {
            "finalizer": {
                "final_answer": "Apple revenue was $391B in 2024. [1]",
                "final_grounded": True,
                "unsupported_claims_final": [],
            }
        }

    from unittest.mock import MagicMock
    mock_graph = MagicMock()
    mock_graph.astream = mock_astream
    monkeypatch.setattr("backend.api.main.verity_graph", mock_graph)

    client = TestClient(app)
    raw_lines = []
    with client.stream(
        "POST",
        "/chat",
        json={
            "query": "What was Apple's revenue in 2024?",
            "history": [],
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        got_done_event = False
        for line in response.iter_lines():
            raw_lines.append(line)
            if line.startswith("event: done"):
                got_done_event = True
            elif got_done_event and line.startswith("data:"):
                break

    events = parse_sse_events("\n".join(raw_lines))
    event_names = [e[0] for e in events]

    # Verify event types present
    assert "status" in event_names
    assert "citation" in event_names
    assert "answer" in event_names
    assert "critic_status" in event_names
    assert "done" in event_names

    # Verify order of events
    citation_idx = event_names.index("citation")
    answer_idx = event_names.index("answer")
    critic_idx = event_names.index("critic_status")
    done_idx = event_names.index("done")
    assert citation_idx < answer_idx < critic_idx < done_idx

    # Verify 'done' payload fields
    done_event = [e[1] for e in events if e[0] == "done"][0]
    assert "final_answer" in done_event
    assert done_event["final_answer"] == "Apple revenue was $391B in 2024. [1]"
    assert "final_grounded" in done_event
    assert done_event["final_grounded"] is True
    assert "unsupported_claims_final" in done_event
    assert done_event["unsupported_claims_final"] == []
    assert "status" in done_event
    assert done_event["status"] == "completed"
    assert "request_id" in done_event
    assert len(done_event["request_id"]) > 0


def test_sse_endpoint_validation_errors():
    """
    Asserts that FastAPI validates ChatRequest:
    - Empty query -> 422 Unprocessable Entity
    - Query > 500 chars -> 422
    - Invalid role (e.g. 'system') -> 422
    - History > 6 messages -> 422
    """
    client = TestClient(app)

    # Empty query
    r1 = client.post("/chat", json={"query": "", "history": []})
    assert r1.status_code == 422

    # Query too long
    r2 = client.post("/chat", json={"query": "a" * 501, "history": []})
    assert r2.status_code == 422

    # Invalid role
    r3 = client.post(
        "/chat",
        json={
            "query": "Valid query",
            "history": [{"role": "system", "content": "You are a bot"}],
        },
    )
    assert r3.status_code == 422

    # History too long (> 6)
    r4 = client.post(
        "/chat",
        json={
            "query": "Valid query",
            "history": [{"role": "user", "content": f"msg {i}"} for i in range(7)],
        },
    )
    assert r4.status_code == 422
