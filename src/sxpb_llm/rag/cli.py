"""Command-line interface for the generic retrieval toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sxpb_llm.rag.chroma import ChromaClient, ChromaError
from sxpb_llm.rag.embed import EmbeddingClient
from sxpb_llm.rag.index import Indexer, SourceSpec
from sxpb_llm.rag.manage import collection_info, copy_collection
from sxpb_llm.rag.search import search


def _pairs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        result[key] = item
    return result


def _chroma(args: argparse.Namespace) -> ChromaClient:
    return ChromaClient(args.chroma_url, args.tenant, args.database)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sxpb-rag")
    parser.add_argument("--chroma-url", required=True, help="ChromaDB API v2 URL")
    parser.add_argument("--tenant", default="default_tenant")
    parser.add_argument("--database", default="default_database")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="incrementally index files")
    index.add_argument("collection")
    index.add_argument("paths", nargs="+", type=Path)
    index.add_argument("--glob", action="append", default=["**/*.md", "**/*.sxpb"])
    index.add_argument("--metadata", action="append", default=[])
    index.add_argument("--path-base", type=Path, default=Path.cwd())
    index.add_argument("--embed-url", required=True)
    index.add_argument("--embed-model", required=True)
    index.add_argument("--embed-batch-size", type=int, default=10)
    index.add_argument("--max-chunk-size", type=int, default=3000)
    index.add_argument("--force", action="store_true")
    index.add_argument("--clean", action="store_true")
    index.add_argument("--quiet", action="store_true")

    query = sub.add_parser("search", help="semantic search")
    query.add_argument("collection")
    query.add_argument("query", nargs="+")
    query.add_argument("--embed-url", required=True)
    query.add_argument("--embed-model", required=True)
    query.add_argument("-n", "--top-k", type=int, default=5)
    query.add_argument("--where", action="append", default=[])
    query.add_argument("--files", action="store_true")
    query.add_argument("--json", action="store_true")

    sub.add_parser("list", help="list collections")
    info = sub.add_parser("info", help="show collection metadata and count")
    info.add_argument("collection")
    copy = sub.add_parser("copy", help="copy a collection")
    copy.add_argument("source")
    copy.add_argument("destination")
    wipe = sub.add_parser("wipe", help="delete a collection")
    wipe.add_argument("collection")
    wipe.add_argument("--confirm", help="must exactly match the collection name")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        chroma = _chroma(args)
        if args.command == "list":
            for collection in chroma.list_collections():
                print(f"{collection.get('name')}\t{collection.get('id')}")
            return 0
        if args.command == "info":
            print(json.dumps(collection_info(chroma, args.collection), indent=2))
            return 0
        if args.command == "copy":
            count = copy_collection(chroma, args.source, args.destination)
            print(f"Copied {count} documents from {args.source} to {args.destination}.")
            return 0
        if args.command == "wipe":
            if args.confirm != args.collection:
                parser.error("--confirm must exactly match the collection name")
            chroma.delete_collection(args.collection)
            print(f"Deleted collection {args.collection}.")
            return 0

        embedder = EmbeddingClient(
            args.embed_url,
            args.embed_model,
            batch_size=getattr(args, "embed_batch_size", 10),
        )
        if args.command == "index":
            metadata = _pairs(args.metadata)
            sources = [
                SourceSpec(path, tuple(args.glob), metadata=metadata)
                for path in args.paths
            ]
            indexer = Indexer(
                chroma=chroma,
                embedder=embedder,
                collection=args.collection,
                path_base=args.path_base,
                max_chunk_size=args.max_chunk_size,
            )
            report = indexer.run(
                sources,
                force=args.force,
                clean=args.clean,
                progress=None if args.quiet else lambda path: print(f"Checking {path}"),
            )
            if not args.quiet:
                print(
                    f"Indexed {report.indexed_files} files / "
                    f"{report.indexed_chunks} chunks; removed "
                    f"{report.removed_chunks} stale chunks."
                )
            return 0
        if args.command == "search":
            where = _pairs(args.where)
            results = search(
                " ".join(args.query),
                chroma=chroma,
                embedder=embedder,
                collection=args.collection,
                top_k=args.top_k,
                where=where or None,
                unique_files=args.files,
            )
            if args.json:
                print(
                    json.dumps(
                        [
                            {
                                "id": item.id,
                                "score": item.score,
                                "content": item.content,
                                "metadata": item.metadata,
                            }
                            for item in results
                        ],
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                for item in results:
                    print(f"{item.score:.4f}  {item.metadata.get('filepath', item.id)}")
                    print(f"  {item.metadata.get('section', '')}")
                    if not args.files:
                        print(item.content[:500])
                        print("-" * 60)
            return 0
    except (ChromaError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
