from pathlib import Path
from typing import Any, cast

import pytest

from sxpb_llm.rag.chunk import Chunk
from sxpb_llm.rag.index import Indexer, SourceSpec


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(i), 1.0] for i, _text in enumerate(texts)]


class FakeChroma:
    def __init__(self):
        self.items: dict[str, tuple[str, list[float], dict]] = {}
        self.fail_verification = False

    def ensure_collection(self, _name: str) -> str:
        return "collection-id"

    def get(self, _collection_id: str, **payload):
        filepath = payload["where"]["filepath"]["$eq"]
        matching = [
            (doc_id, item)
            for doc_id, item in self.items.items()
            if item[2].get("filepath") == filepath
        ]
        if self.fail_verification and matching:
            matching = matching[:-1]
        return {
            "ids": [doc_id for doc_id, _item in matching],
            "documents": [item[0] for _doc_id, item in matching],
            "embeddings": [item[1] for _doc_id, item in matching],
            "metadatas": [item[2] for _doc_id, item in matching],
        }

    def upsert(self, _collection_id: str, payload: dict) -> None:
        for doc_id, document, embedding, metadata in zip(
            payload["ids"],
            payload["documents"],
            payload["embeddings"],
            payload["metadatas"],
        ):
            self.items[doc_id] = (document, embedding, metadata)

    def delete(self, _collection_id: str, payload: dict) -> None:
        for doc_id in payload["ids"]:
            self.items.pop(doc_id, None)

    def iter_get(self, _collection_id: str, **_payload):
        yield {
            "ids": list(self.items),
            "metadatas": [item[2] for item in self.items.values()],
        }


def _indexer(
    tmp_path: Path,
    chroma: FakeChroma,
    max_chunk_size: int = 3000,
    embedder: FakeEmbedder | None = None,
) -> Indexer:
    def chunks(path: str | Path, _size: int) -> list[Chunk]:
        return [Chunk("one", Path(path).read_text()), Chunk("two", "second")]

    return Indexer(
        chroma=cast(Any, chroma),
        embedder=cast(Any, embedder or FakeEmbedder()),
        collection="test",
        path_base=tmp_path,
        lock_file=tmp_path / "index.lock",
        chunk_file_fn=chunks,
        max_chunk_size=max_chunk_size,
    )


def test_indexer_uses_versioned_ids_and_incremental_run_is_noop(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    source_file = source_dir / "a.md"
    source_file.write_text("first")
    chroma = FakeChroma()
    indexer = _indexer(tmp_path, chroma)
    sources = [SourceSpec(source_dir, ("*.md",))]

    first = indexer.run(sources, randomize=False)
    assert first.indexed_chunks == 2
    first_ids = set(chroma.items)
    assert all(doc_id.startswith("docs/a.md::") for doc_id in first_ids)

    second = indexer.run(sources, randomize=False)
    assert second.indexed_chunks == 0
    assert set(chroma.items) == first_ids

    source_file.write_text("changed")
    third = indexer.run(sources, randomize=False)
    assert third.indexed_chunks == 2
    assert not first_ids.intersection(chroma.items)
    assert len(chroma.items) == 2


def test_existing_vectors_are_reused_when_chunk_content_is_unchanged(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "a.md").write_text("first")
    chroma = FakeChroma()
    chroma.items = {
        "legacy-0": ("first", [1.0, 2.0], {"filepath": "docs/a.md"}),
        "legacy-1": ("second", [3.0, 4.0], {"filepath": "docs/a.md"}),
    }
    embedder = FakeEmbedder()

    report = _indexer(tmp_path, chroma, embedder=embedder).run(
        [SourceSpec(source_dir, ("*.md",))], randomize=False
    )

    assert report.indexed_chunks == 2
    assert embedder.calls == []
    assert not {"legacy-0", "legacy-1"}.intersection(chroma.items)


def test_changed_files_are_embedded_and_deduplicated_across_files(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "a.md").write_text("first")
    (source_dir / "b.md").write_text("second")
    chroma = FakeChroma()
    embedder = FakeEmbedder()

    _indexer(tmp_path, chroma, embedder=embedder).run(
        [SourceSpec(source_dir, ("*.md",))], randomize=False
    )

    assert len(embedder.calls) == 1
    assert set(embedder.calls[0]) == {"first", "second"}


def test_completed_embedding_batches_are_checkpointed_before_later_failure(
    tmp_path: Path,
):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "a.md").write_text("first")
    (source_dir / "b.md").write_text("second")
    chroma = FakeChroma()

    class FailingEmbedder(FakeEmbedder):
        batch_size = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            if self.calls:
                raise RuntimeError("later batch failed")
            return super().embed(texts)

    with pytest.raises(RuntimeError, match="later batch"):
        _indexer(tmp_path, chroma, embedder=FailingEmbedder()).run(
            [SourceSpec(source_dir, ("*.md",))], randomize=False
        )

    assert len(chroma.items) == 2


def test_file_deleted_during_embedding_is_skipped_safely(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    source_file = source_dir / "a.md"
    source_file.write_text("first")
    chroma = FakeChroma()

    class DeletingEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = super().embed(texts)
            source_file.unlink()
            return vectors

    report = _indexer(tmp_path, chroma, embedder=DeletingEmbedder()).run(
        [SourceSpec(source_dir, ("*.md",))], randomize=False
    )

    assert report.indexed_files == 0
    assert not chroma.items


def test_chunk_policy_change_reindexes_unchanged_source(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "a.md").write_text("first")
    chroma = FakeChroma()
    sources = [SourceSpec(source_dir, ("*.md",))]
    _indexer(tmp_path, chroma).run(sources, randomize=False)
    old_ids = set(chroma.items)

    report = _indexer(tmp_path, chroma, max_chunk_size=2000).run(
        sources, randomize=False
    )

    assert report.indexed_chunks == 2
    assert not old_ids.intersection(chroma.items)


def test_old_version_is_not_deleted_when_new_write_cannot_be_verified(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    source_file = source_dir / "a.md"
    source_file.write_text("first")
    chroma = FakeChroma()
    indexer = _indexer(tmp_path, chroma)
    sources = [SourceSpec(source_dir, ("*.md",))]
    indexer.run(sources, randomize=False)
    old_ids = set(chroma.items)

    source_file.write_text("changed")
    chroma.fail_verification = True
    with pytest.raises(RuntimeError, match="verify"):
        indexer.run(sources, randomize=False)

    assert old_ids.issubset(chroma.items)


def test_clean_removes_documents_outside_current_source_set(tmp_path: Path):
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    source_file = source_dir / "a.md"
    source_file.write_text("first")
    chroma = FakeChroma()
    indexer = _indexer(tmp_path, chroma)
    sources = [SourceSpec(source_dir, ("*.md",))]
    indexer.run(sources, randomize=False)
    source_file.unlink()

    report = indexer.run(sources, clean=True, randomize=False)
    assert report.removed_chunks == 2
    assert not chroma.items
