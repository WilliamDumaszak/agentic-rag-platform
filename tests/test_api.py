import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("api.main._check_vectordb", return_value=True)
@patch("api.main.agent_graph")
def test_query_success(mock_graph, mock_vdb, client):
    mock_graph.invoke.return_value = {
        "ranked_response": "Supply chain has 5 phases.",
        "confidence_score": 0.85,
        "source": "retrieval",
    }
    response = client.post("/query", json={"query": "What are the supply chain phases?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["confidence"] == 0.85


@patch("api.main._check_vectordb", return_value=False)
def test_query_no_vectordb(mock_vdb, client):
    response = client.post("/query", json={"query": "test?"})
    assert response.status_code == 503


def test_query_too_short(client):
    response = client.post("/query", json={"query": "hi"})
    assert response.status_code == 422
