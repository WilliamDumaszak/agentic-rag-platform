from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, example="What is supply chain?")


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    source: str


class IngestResponse(BaseModel):
    chunks_stored: int
    message: str


class EvalResponse(BaseModel):
    hit_rate: float | None
    mrr: float | None
    n_queries: int | None


class RagasSample(BaseModel):
    question: str = Field(..., min_length=3)
    answer: str
    contexts: list[str] = Field(..., min_length=1)
    ground_truth: str | None = None


class RagasEvalResponse(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    n_samples: int
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    vectordb_ready: bool


# ── HITL schemas ──────────────────────────────────────────────────────────────

class HITLQueueItem(BaseModel):
    id: int
    query: str
    answer: str
    confidence: float
    source: str
    prompt_hash: str
    status: str
    reviewed_by: str | None = None
    review_note: str | None = None
    created_at: str
    reviewed_at: str | None = None


class HITLReviewRequest(BaseModel):
    reviewed_by: str = Field(default="", description="Reviewer identifier (email, username, etc.)")
    review_note: str = Field(default="", description="Optional free-text note explaining the decision.")


class IngestDocumentResponse(BaseModel):
    filename: str
    sections_extracted: int
    chunks_stored: int
    provider: str
    message: str
