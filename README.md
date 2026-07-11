# sxpb-llm

Shared OpenAI-compatible LLM calls, model definitions, SxPB extraction, and
retrieval utilities.

## Retrieval toolkit

`sxpb_llm.rag` provides:

- Markdown, SxPB, YAML-list, and plain-text chunkers
- Batched OpenAI-compatible embeddings
- A small ChromaDB v2 client
- Incremental file indexing with PID locking and orphan cleanup
- Semantic search and collection inspection/copy/removal

The package contains no deployment endpoints or source paths. Applications supply
them through the Python API or `sxpb-rag` CLI.

```bash
sxpb-rag \
  --chroma-url http://localhost:8000/api/v2 \
  index docs \
  ./reference \
  --path-base "$PWD" \
  --embed-url http://localhost:11434/v1/embeddings \
  --embed-model example-embedding-model \
  --clean

sxpb-rag \
  --chroma-url http://localhost:8000/api/v2 \
  search docs "deployment guide" \
  --embed-url http://localhost:11434/v1/embeddings \
  --embed-model example-embedding-model
```

Index IDs include a hash of the source and resulting chunk stream, so source or
chunk-policy changes create a new version. A changed version is upserted and
verified before old IDs are deleted, so a failed replacement leaves the prior
version searchable. `wipe` additionally requires `--confirm COLLECTION`.

Run the test and quality suite with:

```bash
pdm run tidytest
```
