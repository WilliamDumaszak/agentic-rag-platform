from typing import Annotated
from pydantic import BaseModel
import operator


class AgentState(BaseModel):
    """Shared state passed through all nodes in the LangGraph pipeline."""

    query: str
    next_step: str = ""
    retrieved_docs: list = []
    possible_responses: list = []
    similarity_scores: list = []
    ranked_response: str = ""
    confidence_score: float = 0.0
    source: str = ""          # "retrieval" | "web" | "direct"
