"""Semantic search over a ChromaDB collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sxpb_llm.rag.chroma import ChromaClient
from sxpb_llm.rag.embed import EmbeddingClient


@dataclass(frozen=True)
class SearchResult:
    id: str
    content: str
    metadata: dict[str, Any]
    distance: float

    @property
    def score(self) -> float:
        return 1.0 - self.distance


def search(
    query: str,
    *,
    chroma: ChromaClient,
    embedder: EmbeddingClient,
    collection: str,
    top_k: int = 5,
    where: dict | None = None,
    unique_files: bool = False,
) -> list[SearchResult]:
    """Embed a query and return nearest documents in score order."""
    collection_id = chroma.collection_id(collection)
    payload: dict[str, Any] = {
        "query_embeddings": [embedder.embed([query])[0]],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        payload["where"] = where
    response = chroma.query(collection_id, payload)
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    results = [
        SearchResult(doc_id, document, metadata, float(distance))
        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]
    if not unique_files:
        return results
    best_by_file: dict[str, SearchResult] = {}
    for result in results:
        filepath = str(result.metadata.get("filepath", result.id))
        previous = best_by_file.get(filepath)
        if previous is None or result.score > previous.score:
            best_by_file[filepath] = result
    return sorted(best_by_file.values(), key=lambda item: item.score, reverse=True)
