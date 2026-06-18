"""
FastAPI application for the Agentic RAG Platform.

Endpoints:
  POST /query       — run a query through the LangGraph agent
  POST /ingest      — load documents into ChromaDB
  GET  /health      — liveness check
  GET  /evaluate    — run hit_rate + MRR against ground truth
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import agent_graph
from agent.state import AgentState
from api.schemas import (
    EvalResponse,
    HealthResponse,
    HITLQueueItem,
    HITLReviewRequest,
    IngestDocumentResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RagasEvalResponse,
    RagasSample,
)
from evaluation.metrics import evaluate_retriever
from hitl.queue import approve as hitl_approve, enqueue as hitl_enqueue
from hitl.queue import get_pending as hitl_get_pending, reject as hitl_reject
from hitl.queue import setup_table as hitl_setup_table
from rag.ingestion import ingest_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def _check_vectordb() -> bool:
    import os
    vdb_dir = CONFIG["vectordb"]["persist_directory"]
    return os.path.isdir(vdb_dir) and len(os.listdir(vdb_dir)) > 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic RAG Platform...")
    hitl_setup_table()
    yield


app = FastAPI(
    title="Agentic RAG Platform",
    description=(
        "LangGraph-powered agentic RAG with ChromaDB, Ollama, "
        "DuckDuckGo web search fallback, and multi-response ranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(status="ok", vectordb_ready=_check_vectordb())


@app.post("/ingest", response_model=IngestResponse, tags=["data"])
def ingest():
    """Trigger document ingestion into ChromaDB."""
    try:
        n_chunks = ingest_documents()
        return IngestResponse(
            chunks_stored=n_chunks,
            message=f"Successfully stored {n_chunks} chunks.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse, tags=["inference"])
def query(request: QueryRequest):
    """Run a query through the agentic RAG pipeline."""
    # guardrails: validate input before any processing
    from guardrails.validator import validate_input, validate_output, sanitize_output
    check = validate_input(request.query, CONFIG.get("guardrails", {}).get("max_input_length", 1000))
    if not check.valid:
        raise HTTPException(status_code=422, detail=check.reason)

    if not _check_vectordb():
        raise HTTPException(
            status_code=503,
            detail="Vector database is empty. Call POST /ingest first.",
        )
    try:
        initial_state = AgentState(query=request.query)
        result = agent_graph.invoke(initial_state)
        answer = result.get("ranked_response", "No answer generated.")
        confidence = float(result.get("confidence_score", 0.0))
        source = result.get("source", "unknown")

        # guardrails: sanitize output PII
        if CONFIG.get("guardrails", {}).get("block_pii_in_output", True):
            answer = sanitize_output(answer)

        # HITL: route low-confidence responses to the review queue
        routed_to_review = False
        hitl_id = None
        confidence_threshold = float(
            CONFIG.get("agent", {}).get("confidence_threshold", 0.3)
        )
        if confidence < confidence_threshold:
            from prompts.manager import get_prompt_hash
            prompt_hash = get_prompt_hash(version="v2")
            hitl_id = hitl_enqueue(
                query=request.query,
                answer=answer,
                confidence=confidence,
                source=source,
                prompt_hash=prompt_hash,
            )
            routed_to_review = True
            logger.info(
                f"Low-confidence response (score={confidence:.3f} < {confidence_threshold}) "
                f"routed to HITL queue, id={hitl_id}"
            )

        response = QueryResponse(
            query=request.query,
            answer=answer,
            confidence=confidence,
            source=source,
        )

        if routed_to_review:
            # Attach review metadata as extra fields via JSON response
            data = response.model_dump()
            data["routed_to_review"] = True
            data["review_id"] = hitl_id
            data["review_reason"] = (
                f"Confidence {confidence:.3f} is below threshold {confidence_threshold}"
            )
            return JSONResponse(content=data)

        return response

    except Exception as exc:
        logger.error(f"Query failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/document", response_model=IngestDocumentResponse, tags=["data"])
def ingest_document(
    file: "UploadFile",  # noqa: F821  (resolved via import below)
):
    """
    Upload a PDF (or text) file, extract text via Document Intelligence, chunk and index.

    Dual-provider: 'local' uses PyMuPDF; 'azure' uses Azure Document Intelligence.
    Provider is set in config document_intelligence.provider.
    """
    import shutil
    import tempfile
    from fastapi import UploadFile
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from rag.ingestion import build_embedding_model
    from ingestion.document_intelligence import extract_document

    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        docs = extract_document(tmp_path)
        provider = CONFIG.get("document_intelligence", {}).get("provider", "local")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["ingestion"]["chunk_size"],
            chunk_overlap=CONFIG["ingestion"]["chunk_overlap"],
        )
        chunks = splitter.split_documents(docs)

        embedding_model = build_embedding_model()
        Chroma.from_documents(
            chunks,
            embedding_model,
            persist_directory=CONFIG["vectordb"]["persist_directory"],
            collection_name=CONFIG["vectordb"]["collection_name"],
        )

        return IngestDocumentResponse(
            filename=file.filename or "unknown",
            sections_extracted=len(docs),
            chunks_stored=len(chunks),
            provider=provider,
            message=f"Extracted {len(docs)} sections, stored {len(chunks)} chunks.",
        )
    except Exception as exc:
        logger.error(f"Document ingest failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── HITL review endpoints ──────────────────────────────────────────────────────

@app.get("/review/pending", response_model=list[HITLQueueItem], tags=["review"])
def review_pending():
    """
    List all responses pending human review (low-confidence answers).

    Returns items ordered oldest-first so reviewers handle the backlog in FIFO order.
    After reviewing, call POST /review/{id}/approve or POST /review/{id}/reject.
    """
    return hitl_get_pending()


@app.post("/review/{item_id}/approve", tags=["review"])
def review_approve(item_id: int, body: HITLReviewRequest = HITLReviewRequest()):
    """
    Approve a low-confidence response. Marks the item as reviewed and approved.
    The original answer is considered acceptable for delivery.
    """
    success = hitl_approve(
        item_id,
        reviewed_by=body.reviewed_by,
        review_note=body.review_note,
    )
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Item {item_id} not found or already reviewed.",
        )
    return {"item_id": item_id, "status": "approved"}


@app.post("/review/{item_id}/reject", tags=["review"])
def review_reject(item_id: int, body: HITLReviewRequest = HITLReviewRequest()):
    """
    Reject a low-confidence response. Marks the item as rejected.
    Use this when the LLM answer is incorrect or harmful and should not be served.
    """
    success = hitl_reject(
        item_id,
        reviewed_by=body.reviewed_by,
        review_note=body.review_note,
    )
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Item {item_id} not found or already reviewed.",
        )
    return {"item_id": item_id, "status": "rejected"}


@app.post("/query/stream", tags=["inference"])
def query_stream(request: QueryRequest):
    """
    Streaming RAG using Server-Sent Events (SSE).

    Pipeline: validate → retrieve → cross-encoder rerank → stream LLM tokens.
    Unlike /query (multi-response ranking), this endpoint streams a single response
    token-by-token for real-time UX. Compatible with Ollama, Azure OpenAI, and vLLM.

    Client usage:
        const es = new EventSource('/query/stream', {method: 'POST', body: JSON.stringify({query: '...'})});
        es.onmessage = (e) => { const d = JSON.parse(e.data); process(d.token); if (d.done) es.close(); }
    """
    from guardrails.validator import validate_input
    check = validate_input(request.query, CONFIG.get("guardrails", {}).get("max_input_length", 1000))
    if not check.valid:
        raise HTTPException(status_code=422, detail=check.reason)
    if not _check_vectordb():
        raise HTTPException(status_code=503, detail="Vector database is empty.")

    def _generate():
        import json
        import requests as req

        # 1. retrieve + rerank
        try:
            from agent.nodes import _build_retriever
            from rag.hybrid_retriever import hybrid_retrieve
            from reranking.cross_encoder import rerank

            docs = hybrid_retrieve(request.query)
            if not docs:
                retriever = _build_retriever()
                docs = retriever.invoke(request.query)

            docs = rerank(request.query, docs)
            context = "\n".join(d.page_content for d in docs)
            sources = list({d.metadata.get("source", "unknown") for d in docs})
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Retrieval failed: {exc}', 'done': True})}\n\n"
            return

        prompt = (
            "Answer the question based on the context below.\n"
            f"Context: {context}\n"
            f"Question: {request.query}\nAnswer:"
        )

        # 2. stream from LLM provider
        provider = CONFIG["llm"]["provider"]
        try:
            if provider == "ollama":
                payload = {
                    "model": CONFIG["llm"]["model"],
                    "prompt": prompt,
                    "stream": True,
                }
                with req.post(
                    f"{CONFIG['llm']['ollama_base_url']}/api/generate",
                    json=payload, stream=True, timeout=120,
                ) as resp:
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line.decode())
                            token = data.get("response", "")
                            done = data.get("done", False)
                            yield f"data: {json.dumps({'token': token, 'done': done, 'sources': sources if done else None})}\n\n"
                            if done:
                                break

            else:
                # azure_openai or vllm — both expose OpenAI-compatible streaming
                from openai import AzureOpenAI, OpenAI
                if provider == "azure_openai":
                    client = AzureOpenAI(
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                        api_version=CONFIG["llm"]["azure_openai_api_version"],
                    )
                    model_id = CONFIG["llm"]["azure_openai_deployment"]
                else:
                    # openai-compatible: works for vLLM, Together, Groq, etc.
                    client = OpenAI(
                        api_key=os.getenv("OPENAI_API_KEY", "no-key"),
                        base_url=CONFIG["llm"].get("vllm_base_url", "http://vllm:8001/v1"),
                    )
                    model_id = CONFIG["llm"].get("vllm_model", CONFIG["llm"]["model"])

                stream = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    max_tokens=CONFIG["llm"]["max_tokens"],
                    temperature=CONFIG["llm"]["temperature"],
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    finish = chunk.choices[0].finish_reason
                    done = finish is not None
                    yield f"data: {json.dumps({'token': delta, 'done': done, 'sources': sources if done else None})}\n\n"

        except Exception as exc:
            logger.error(f"LLM streaming failed: {exc}", exc_info=True)
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/evaluate", response_model=EvalResponse, tags=["evaluation"])
def evaluate():
    """Run hit_rate and MRR evaluation against the ground truth dataset."""
    if not _check_vectordb():
        raise HTTPException(status_code=503, detail="Vector database is empty.")
    try:
        from langchain_chroma import Chroma
        from rag.ingestion import build_embedding_model

        db = Chroma(
            persist_directory=CONFIG["vectordb"]["persist_directory"],
            collection_name=CONFIG["vectordb"]["collection_name"],
            embedding_function=build_embedding_model(),
        )
        retriever = db.as_retriever()
        metrics = evaluate_retriever(retriever, CONFIG["evaluation"]["ground_truth_path"])
        return EvalResponse(**metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate/ragas", response_model=RagasEvalResponse, tags=["evaluation"])
def evaluate_ragas(samples: list[RagasSample]):
    """
    Run RAGAS evaluation (faithfulness, answer_relevancy, context_precision).

    Send a list of {question, answer, contexts, ground_truth?} objects.
    Typically used after calling /query to collect question+answer+contexts.
    """
    if not samples:
        raise HTTPException(status_code=422, detail="Provide at least one sample.")
    try:
        from evaluation.ragas_eval import evaluate_with_ragas
        result = evaluate_with_ragas([s.model_dump() for s in samples])
        return RagasEvalResponse(**result)
    except Exception as exc:
        logger.error(f"RAGAS eval failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
