"""Incremental, crash-safe file indexing."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sxpb_llm.rag.chroma import ChromaClient, ChromaError
from sxpb_llm.rag.chunk import Chunk, chunk_file
from sxpb_llm.rag.embed import EmbeddingClient

MetadataFactory = Callable[[Path], Mapping[str, str | int | float | bool]]
ChunkFile = Callable[[str | Path, int], list[Chunk]]


@dataclass(frozen=True)
class SourceSpec:
    """A root and file patterns sharing chunking and metadata policy."""

    root: Path
    patterns: tuple[str, ...]
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    metadata_factory: MetadataFactory | None = None

    def files(self) -> list[Path]:
        found: set[Path] = set()
        if not self.root.exists():
            return []
        for pattern in self.patterns:
            found.update(path for path in self.root.glob(pattern) if path.is_file())
        return list(found)


@dataclass(frozen=True)
class IndexReport:
    candidates: int
    indexed_files: int
    indexed_chunks: int
    removed_chunks: int


@dataclass(frozen=True)
class _PreparedFile:
    path: Path
    source: SourceSpec
    filepath: str
    source_hash: str
    index_hash: str
    mtime: str
    chunks: list[Chunk]
    existing_vectors: dict[str, list[float]]


class PidLock:
    """Simple process lock that recovers from stale PID files."""

    def __init__(self, path: Path):
        self.path = path
        self.owned = False

    def __enter__(self) -> "PidLock":
        try:
            old_pid = self.path.read_text().strip()
        except FileNotFoundError:
            old_pid = ""
        if old_pid and Path(f"/proc/{old_pid}").exists():
            raise RuntimeError(f"indexer already running as PID {old_pid}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        self.owned = True
        return self

    def __exit__(self, *_args: object) -> None:
        if not self.owned:
            return
        try:
            if self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except FileNotFoundError:
            pass


class Indexer:
    def __init__(
        self,
        *,
        chroma: ChromaClient,
        embedder: EmbeddingClient,
        collection: str,
        path_base: Path,
        max_chunk_size: int = 3000,
        write_batch_size: int = 100,
        lock_file: Path | None = None,
        chunk_file_fn: ChunkFile = chunk_file,
    ):
        self.chroma = chroma
        self.embedder = embedder
        self.collection = collection
        self.path_base = path_base
        self.max_chunk_size = max_chunk_size
        self.write_batch_size = write_batch_size
        self.lock_file = lock_file or Path(f"/tmp/{collection}_indexer.lock")
        self.chunk_file = chunk_file_fn

    def _filepath(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.path_base))
        except ValueError:
            return str(path.resolve())

    def _entries(
        self, collection_id: str, filepath: str
    ) -> tuple[list[str], list[dict], list[str], list[list[float]]]:
        result = self.chroma.get(
            collection_id,
            where={"filepath": {"$eq": filepath}},
            include=["metadatas", "documents", "embeddings"],
        )
        return (
            list(result.get("ids") or []),
            list(result.get("metadatas") or []),
            list(result.get("documents") or []),
            list(result.get("embeddings") or []),
        )

    def _delete_ids(self, collection_id: str, ids: Sequence[str]) -> int:
        for start in range(0, len(ids), self.write_batch_size):
            self.chroma.delete(
                collection_id, {"ids": list(ids[start : start + self.write_batch_size])}
            )
        return len(ids)

    def _prepare_file(
        self,
        path: Path,
        source: SourceSpec,
        collection_id: str,
        *,
        force: bool,
    ) -> tuple[_PreparedFile | None, int]:
        filepath = self._filepath(path)
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        chunks = self.chunk_file(path, self.max_chunk_size)
        if not chunks:
            old_ids, _old_metadata, _old_documents, _old_vectors = self._entries(
                collection_id, filepath
            )
            return None, self._delete_ids(collection_id, old_ids)

        # Include chunk output in the indexed version. A chunker or size-policy
        # change must rebuild unchanged source files rather than silently reuse
        # stale chunks.
        index_digest = hashlib.sha256()
        index_digest.update(f"{source_hash}:{self.max_chunk_size}".encode())
        for chunk in chunks:
            index_digest.update(b"\0")
            index_digest.update(chunk.section.encode())
            index_digest.update(b"\0")
            index_digest.update(chunk.content.encode())
        index_hash = index_digest.hexdigest()

        old_ids, old_metadata, old_documents, old_vectors = self._entries(
            collection_id, filepath
        )
        current_ids = [
            doc_id
            for doc_id, metadata in zip(old_ids, old_metadata)
            if metadata.get("index_hash") == index_hash
        ]
        current_id_set = set(current_ids)
        stale_ids = [doc_id for doc_id in old_ids if doc_id not in current_id_set]
        expected_count = next(
            (
                int(metadata["chunk_count"])
                for metadata in old_metadata
                if metadata.get("index_hash") == index_hash
                and metadata.get("chunk_count") is not None
            ),
            None,
        )
        if (
            not force
            and expected_count is not None
            and len(current_ids) == expected_count
        ):
            return None, self._delete_ids(collection_id, stale_ids)
        return (
            _PreparedFile(
                path,
                source,
                filepath,
                source_hash,
                index_hash,
                str(path.stat().st_mtime),
                chunks,
                {
                    document: vector
                    for document, vector in zip(old_documents, old_vectors)
                    if document and vector
                },
            ),
            0,
        )

    def _write_file(
        self,
        prepared: _PreparedFile,
        vectors: list[list[float]],
        collection_id: str,
    ) -> tuple[int, int]:
        if len(vectors) != len(prepared.chunks):
            raise RuntimeError("embedding count does not match chunk count")
        # Sources can change or disappear during a long embedding batch. Do not
        # publish stale chunks; a later incremental run will pick up the new path.
        try:
            current_hash = hashlib.sha256(prepared.path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return 0, 0
        if current_hash != prepared.source_hash:
            return 0, 0
        version = prepared.index_hash[:16]
        ids = [
            f"{prepared.filepath}::{version}::{i}" for i in range(len(prepared.chunks))
        ]
        static_metadata = dict(prepared.source.metadata)
        if prepared.source.metadata_factory:
            static_metadata.update(prepared.source.metadata_factory(prepared.path))
        metadatas = [
            {
                **static_metadata,
                "filepath": prepared.filepath,
                "section": chunk.section[:200],
                "chunk_index": i,
                "chunk_count": len(prepared.chunks),
                "mtime": prepared.mtime,
                "source_hash": prepared.source_hash,
                "index_hash": prepared.index_hash,
            }
            for i, chunk in enumerate(prepared.chunks)
        ]
        for start in range(0, len(ids), self.write_batch_size):
            end = start + self.write_batch_size
            self.chroma.upsert(
                collection_id,
                {
                    "ids": ids[start:end],
                    "embeddings": vectors[start:end],
                    "documents": [
                        chunk.content for chunk in prepared.chunks[start:end]
                    ],
                    "metadatas": metadatas[start:end],
                },
            )

        written_ids, written_metadata, _written_documents, _written_vectors = (
            self._entries(collection_id, prepared.filepath)
        )
        confirmed = {
            doc_id
            for doc_id, metadata in zip(written_ids, written_metadata)
            if metadata.get("index_hash") == prepared.index_hash
        }
        if not set(ids).issubset(confirmed):
            raise ChromaError(
                f"could not verify all chunks written for {prepared.filepath}"
            )
        desired_ids = set(ids)
        obsolete = [doc_id for doc_id in written_ids if doc_id not in desired_ids]
        return len(ids), self._delete_ids(collection_id, obsolete)

    def _clean_orphans(self, collection_id: str, candidate_filepaths: set[str]) -> int:
        orphan_ids: list[str] = []
        for page in self.chroma.iter_get(collection_id, include=["metadatas"]):
            for doc_id, metadata in zip(
                page.get("ids") or [], page.get("metadatas") or []
            ):
                filepath = metadata.get("filepath")
                if filepath and filepath not in candidate_filepaths:
                    orphan_ids.append(doc_id)
        return self._delete_ids(collection_id, orphan_ids)

    def run(
        self,
        sources: Sequence[SourceSpec],
        *,
        force: bool = False,
        clean: bool = False,
        randomize: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> IndexReport:
        """Index all sources, optionally removing files absent from the source set."""
        with PidLock(self.lock_file):
            collection_id = self.chroma.ensure_collection(self.collection)
            by_path: dict[Path, SourceSpec] = {}
            for source in sources:
                for path in source.files():
                    by_path[path] = source
            candidates = list(by_path)
            if randomize:
                random.shuffle(candidates)
            prepared_files: list[_PreparedFile] = []
            removed_chunks = 0
            for path in candidates:
                if progress:
                    progress(self._filepath(path))
                prepared, removed = self._prepare_file(
                    path, by_path[path], collection_id, force=force
                )
                if prepared:
                    prepared_files.append(prepared)
                removed_chunks += removed

            # Batch across file boundaries, but checkpoint each embedding batch
            # into ChromaDB. A late failure then loses at most one batch rather
            # than all work from a long reindex.
            target_size = max(1, int(getattr(self.embedder, "batch_size", 10)))
            groups: list[list[_PreparedFile]] = []
            group: list[_PreparedFile] = []
            group_size = 0
            for prepared in prepared_files:
                chunk_count = len(prepared.chunks)
                if group and group_size + chunk_count > target_size:
                    groups.append(group)
                    group = []
                    group_size = 0
                group.append(prepared)
                group_size += chunk_count
            if group:
                groups.append(group)

            indexed_files = 0
            indexed_chunks = 0
            for group in groups:
                texts = [
                    chunk.content for prepared in group for chunk in prepared.chunks
                ]
                vectors_by_content: dict[str, list[float]] = {}
                for prepared in group:
                    vectors_by_content.update(prepared.existing_vectors)
                missing_texts = list(
                    dict.fromkeys(
                        text for text in texts if text not in vectors_by_content
                    )
                )
                if missing_texts:
                    new_vectors = self.embedder.embed(missing_texts)
                    if len(new_vectors) != len(missing_texts):
                        raise RuntimeError("embedding count does not match chunk count")
                    vectors_by_content.update(zip(missing_texts, new_vectors))
                vectors = [vectors_by_content[text] for text in texts]
                vector_offset = 0
                for prepared in group:
                    vector_end = vector_offset + len(prepared.chunks)
                    count, removed = self._write_file(
                        prepared, vectors[vector_offset:vector_end], collection_id
                    )
                    if count:
                        indexed_files += 1
                        indexed_chunks += count
                    removed_chunks += removed
                    vector_offset = vector_end
            if clean:
                removed_chunks += self._clean_orphans(
                    collection_id, {self._filepath(path) for path in candidates}
                )
            return IndexReport(
                candidates=len(candidates),
                indexed_files=indexed_files,
                indexed_chunks=indexed_chunks,
                removed_chunks=removed_chunks,
            )
