"""
Azure AI Search helper — drop-in alternative to ChromaDB for cloud deployments.

Provides ingest and retrieval functions using Azure Cognitive Search REST API.
Used by agent/nodes.py when vectordb.provider == "azure_search" in config.yaml.
"""

import logging
import os

import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def _get_search_client():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", CONFIG["vectordb"].get("azure_search_endpoint", ""))
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    index_name = CONFIG["vectordb"]["azure_search_index"]

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )


def _get_index_client():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        SearchableField,
    )

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", CONFIG["vectordb"].get("azure_search_endpoint", ""))
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    return SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))


def ensure_index() -> None:
    """Create the Azure AI Search index if it does not exist."""
    from azure.search.documents.indexes.models import (
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        SearchableField,
    )

    index_client = _get_index_client()
    index_name = CONFIG["vectordb"]["azure_search_index"]

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
    ]
    index = SearchIndex(name=index_name, fields=fields)
    index_client.create_or_update_index(index)
    logger.info(f"Azure AI Search index '{index_name}' ready.")


def ingest_documents(documents: list[dict]) -> int:
    """
    Upload documents to Azure AI Search.
    Each doc must have: id (str), content (str), source (str).
    Returns number of documents indexed.
    """
    client = _get_search_client()
    batch = [{"@search.action": "upload", **doc} for doc in documents]
    result = client.upload_documents(documents=batch)
    succeeded = sum(1 for r in result if r.succeeded)
    logger.info(f"Indexed {succeeded}/{len(batch)} documents to Azure AI Search.")
    return succeeded


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Full-text BM25 search. Returns list of source dicts."""
    client = _get_search_client()
    k = top_k or CONFIG["vectordb"].get("top_k", 5)
    results = client.search(search_text=query, top=k)
    return [dict(r) for r in results]
