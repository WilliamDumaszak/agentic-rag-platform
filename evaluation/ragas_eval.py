"""
RAGAS-based LLM evaluation.

Metrics:
  faithfulness       — is the answer grounded in the retrieved context?
  answer_relevancy   — is the answer relevant to the question?
  context_precision  — is the retrieved context precise for the question?

Works with both local Ollama and Azure OpenAI (reads config.yaml to decide).
Requires: ragas>=0.2.0
"""

import logging
import os

import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def _build_ragas_llm():
    """Return a RAGAS-compatible LLM wrapper, respecting config provider."""
    from ragas.llms import LangchainLLMWrapper

    provider = CONFIG["llm"]["provider"]

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", CONFIG["llm"].get("azure_openai_endpoint", "")),
            azure_deployment=CONFIG["llm"]["azure_openai_deployment"],
            api_version=CONFIG["llm"]["azure_openai_api_version"],
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            temperature=0,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=CONFIG["llm"]["model"], temperature=0)
    else:
        # local Ollama
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=CONFIG["llm"]["model"],
            base_url=CONFIG["llm"]["ollama_base_url"],
            temperature=0,
        )

    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Return a RAGAS-compatible embeddings wrapper using HuggingFace (always local)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = HuggingFaceEmbeddings(
        model_name=CONFIG["embeddings"]["model"],
        model_kwargs={"device": CONFIG["embeddings"]["device"]},
        encode_kwargs={"normalize_embeddings": CONFIG["embeddings"]["normalize"]},
    )
    return LangchainEmbeddingsWrapper(emb)


def evaluate_with_ragas(samples: list[dict]) -> dict:
    """
    Run RAGAS evaluation on a list of samples.

    Each sample must have:
      question  (str)
      answer    (str)
      contexts  (list[str])  — retrieved passages used to generate the answer
      ground_truth (str, optional)

    Returns dict: {faithfulness, answer_relevancy, context_precision, n_samples}
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
    context_precision = ContextPrecision(llm=ragas_llm)

    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("ground_truth", s["answer"]),
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    try:
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        scores = result.to_pandas()
        return {
            "faithfulness": round(float(scores["faithfulness"].mean()), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"].mean()), 4),
            "context_precision": round(float(scores["context_precision"].mean()), 4),
            "n_samples": len(samples),
        }
    except Exception as exc:
        logger.error(f"RAGAS evaluation failed: {exc}")
        return {"error": str(exc), "n_samples": len(samples)}
