"""
Agent node functions — each receives an AgentState and returns an updated AgentState.

Flow:
  decision_node
    ├─ "retrieve"    → retrieve_node → generate_node → rank_node
    ├─ "web_search"  → web_search_node
    └─ "generate"    → generate_node → rank_node
"""

import logging
import os

import numpy as np
import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from agent.state import AgentState

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_llm():
    provider = CONFIG["llm"]["provider"]

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", CONFIG["llm"].get("azure_openai_endpoint", "")),
            azure_deployment=CONFIG["llm"]["azure_openai_deployment"],
            api_version=CONFIG["llm"]["azure_openai_api_version"],
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"],
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=CONFIG["llm"]["model"],
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_tokens"],
        )

    # default: ollama
    from langchain_community.llms import Ollama
    return Ollama(
        model=CONFIG["llm"]["model"],
        base_url=CONFIG["llm"]["ollama_base_url"],
        temperature=CONFIG["llm"]["temperature"],
    )


def _build_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=CONFIG["embeddings"]["model"],
        model_kwargs={"device": CONFIG["embeddings"]["device"]},
        encode_kwargs={"normalize_embeddings": CONFIG["embeddings"]["normalize"]},
    )


def _build_retriever():
    provider = CONFIG["vectordb"].get("provider", "chroma")
    embeddings = _build_embeddings()

    if provider == "azure_search":
        from langchain_community.vectorstores import AzureSearch
        return AzureSearch(
            azure_search_endpoint=os.getenv(
                "AZURE_SEARCH_ENDPOINT", CONFIG["vectordb"].get("azure_search_endpoint", "")
            ),
            azure_search_key=os.getenv("AZURE_SEARCH_API_KEY", ""),
            index_name=CONFIG["vectordb"]["azure_search_index"],
            embedding_function=embeddings.embed_query,
        ).as_retriever()

    # default: chroma
    from langchain_chroma import Chroma
    db = Chroma(
        persist_directory=CONFIG["vectordb"]["persist_directory"],
        collection_name=CONFIG["vectordb"]["collection_name"],
        embedding_function=embeddings,
    )
    return db.as_retriever()


# ── decision ──────────────────────────────────────────────────────────────────

def decision_node(state: AgentState) -> AgentState:
    """Decide whether to retrieve from vectordb, search the web, or generate directly."""
    query_lower = state.query.lower()
    web_keywords = CONFIG["agent"]["web_search_keywords"]

    if any(kw in query_lower for kw in web_keywords):
        next_step = "web_search"
    elif "?" in state.query or len(state.query.split()) > 4:
        next_step = "retrieve"
    else:
        next_step = "generate"

    logger.info(f"Decision: {next_step} for query='{state.query}'")
    return state.model_copy(update={"next_step": next_step})


# ── retrieve ──────────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> AgentState:
    retriever = _build_retriever()
    docs = retriever.invoke(state.query)
    logger.info(f"Retrieved {len(docs)} documents.")

    # cross-encoder reranking: re-scores (query, doc) pairs for higher precision
    try:
        from reranking.cross_encoder import rerank
        docs = rerank(state.query, docs)
        logger.info(f"After reranking: {len(docs)} documents retained.")
    except Exception as exc:
        logger.warning(f"Reranking skipped: {exc}")

    return state.model_copy(update={"retrieved_docs": docs, "source": "retrieval"})


# ── web search ────────────────────────────────────────────────────────────────

def web_search_node(state: AgentState) -> AgentState:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)
    try:
        result = search.run(state.query)
        logger.info("Web search completed.")
    except Exception as exc:
        logger.warning(f"Web search failed: {exc}. Falling back to retrieval.")
        return retrieve_node(state.model_copy(update={"next_step": "retrieve"}))

    return state.model_copy(
        update={
            "ranked_response": result,
            "confidence_score": 1.0,
            "source": "web",
        }
    )


# ── generate multiple responses ───────────────────────────────────────────────

def generate_node(state: AgentState) -> AgentState:
    llm = _build_llm()
    parser = StrOutputParser()

    context = "\n".join(
        doc.page_content for doc in state.retrieved_docs
    ) if state.retrieved_docs else ""

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Answer the question based on the context below.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    chain = prompt | llm | parser

    n = CONFIG["agent"]["n_responses"]
    responses = []
    for _ in range(n):
        try:
            resp = chain.invoke({"context": context, "question": state.query})
            responses.append(resp)
        except Exception as exc:
            logger.warning(f"Generation attempt failed: {exc}")

    logger.info(f"Generated {len(responses)} response variants.")
    return state.model_copy(update={"possible_responses": responses})


# ── rank responses ────────────────────────────────────────────────────────────

def rank_node(state: AgentState) -> AgentState:
    if not state.possible_responses:
        return state.model_copy(
            update={"ranked_response": "No response generated.", "confidence_score": 0.0}
        )

    embeddings = _build_embeddings()

    # encode query + retrieved context as reference
    reference_text = state.query
    if state.retrieved_docs:
        reference_text += " " + " ".join(d.page_content for d in state.retrieved_docs[:3])

    try:
        ref_emb = embeddings.embed_documents([reference_text])
        resp_embs = embeddings.embed_documents(state.possible_responses)
        scores = cosine_similarity(ref_emb, resp_embs)[0].tolist()
    except Exception as exc:
        logger.warning(f"Similarity scoring failed: {exc}. Using first response.")
        scores = [1.0] + [0.0] * (len(state.possible_responses) - 1)

    best_idx = int(np.argmax(scores))
    best_response = state.possible_responses[best_idx]
    confidence = float(scores[best_idx])

    logger.info(f"Best response idx={best_idx}, confidence={confidence:.3f}")
    return state.model_copy(
        update={
            "similarity_scores": scores,
            "ranked_response": best_response,
            "confidence_score": confidence,
            "source": state.source or "retrieval",
        }
    )
