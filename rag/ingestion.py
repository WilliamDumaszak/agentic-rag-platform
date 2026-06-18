"""
Document ingestion pipeline.

Loads JSON/text documents from the documents/ directory,
splits them into chunks, embeds and stores in ChromaDB.
"""

import logging
import os

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

JQ_SCHEMA = 'to_entries | map(.key + ": " + .value) | join("\\n")'


def build_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=CONFIG["embeddings"]["model"],
        model_kwargs={"device": CONFIG["embeddings"]["device"]},
        encode_kwargs={"normalize_embeddings": CONFIG["embeddings"]["normalize"]},
    )


def ingest_documents(documents_dir: str | None = None) -> int:
    """
    Load documents, split into chunks and store in ChromaDB.
    Returns the number of chunks stored.
    """
    docs_dir = documents_dir or CONFIG["ingestion"]["documents_dir"]

    loader = DirectoryLoader(
        docs_dir,
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": JQ_SCHEMA},
    )
    documents = loader.load()

    if not documents:
        logger.warning(f"No documents found in {docs_dir}")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["ingestion"]["chunk_size"],
        chunk_overlap=CONFIG["ingestion"]["chunk_overlap"],
    )
    chunks = splitter.split_documents(documents)

    embedding_model = build_embedding_model()

    Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=CONFIG["vectordb"]["persist_directory"],
        collection_name=CONFIG["vectordb"]["collection_name"],
    )

    logger.info(f"Ingested {len(chunks)} chunks from {len(documents)} documents.")
    return len(chunks)
