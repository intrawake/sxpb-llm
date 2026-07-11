"""Generic chunking, embedding, indexing, and search tools."""

from sxpb_llm.rag.chroma import ChromaClient, ChromaError
from sxpb_llm.rag.chunk import (
    Chunk,
    chunk_file,
    chunk_markdown,
    chunk_plain,
    chunk_sxpb,
    chunk_yaml_list,
)
from sxpb_llm.rag.embed import EmbeddingClient, EmbeddingError
from sxpb_llm.rag.index import Indexer, IndexReport, PidLock, SourceSpec
from sxpb_llm.rag.manage import collection_info, copy_collection
from sxpb_llm.rag.search import SearchResult, search

__all__ = [
    "ChromaClient",
    "ChromaError",
    "Chunk",
    "EmbeddingClient",
    "EmbeddingError",
    "Indexer",
    "IndexReport",
    "PidLock",
    "SearchResult",
    "SourceSpec",
    "chunk_file",
    "chunk_markdown",
    "chunk_plain",
    "chunk_sxpb",
    "chunk_yaml_list",
    "collection_info",
    "copy_collection",
    "search",
]
