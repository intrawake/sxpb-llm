"""Collection inspection and management helpers."""

from __future__ import annotations

from sxpb_llm.rag.chroma import ChromaClient, ChromaError


def collection_info(chroma: ChromaClient, name: str) -> dict:
    collection = chroma.get_collection(name)
    if not collection:
        raise ChromaError(f"collection {name!r} does not exist")
    return {**collection, "count": chroma.count(str(collection["id"]))}


def copy_collection(
    chroma: ChromaClient, source: str, destination: str, *, batch_size: int = 100
) -> int:
    """Copy IDs, documents, embeddings, and metadata into a new collection."""
    source_id = chroma.collection_id(source)
    if chroma.get_collection(destination):
        raise ChromaError(f"destination collection {destination!r} already exists")
    destination_id = chroma.ensure_collection(destination)
    copied = 0
    for page in chroma.iter_get(
        source_id,
        batch_size=batch_size,
        include=["documents", "embeddings", "metadatas"],
    ):
        ids = page.get("ids") or []
        chroma.upsert(
            destination_id,
            {
                "ids": ids,
                "documents": page.get("documents") or [],
                "embeddings": page.get("embeddings") or [],
                "metadatas": page.get("metadatas") or [],
            },
        )
        copied += len(ids)
    if chroma.count(destination_id) != copied:
        raise ChromaError("destination count does not match copied count")
    return copied
