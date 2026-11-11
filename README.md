# sxpb-llm

Shared OpenAI-compatible LLM calls, model definitions, SxPB extraction, and
retrieval utilities.

## Markdown code blocks

`parse_code_blocks(text)` extracts language, operation (`>` / `<`) and filename
fields from fenced blocks, with optional SxPB parsing. Content and original
`opening` / `closing` fence lines are preserved exactly, including line endings.
Unfenced operation-looking lines remain prose. The parser never adds or trims
whitespace; reconstruction is exact:

```python
blocks = parse_code_blocks(text)
assert "".join((b.opening or "") + b.content + (b.closing or "") for b in blocks) == text
```

Markdown recognition still accepts indentation, tilde fences and longer closers.
Check `error` and original boundaries if an application needs stricter syntax.
Parsing an operation or filename does not grant read/write/execution authority.
Optional `parse_sxpb` decoding leaves the raw content intact. Program bodies
should not be parsed as SxPB.

`format_code_block(content, *, language="", operation=None, filepath=None)`
returns a Markdown string with matching backtick fences longer than any run
inside its content (and at least three backticks). Header fields follow the
same convention as `parse_code_blocks`; supply a language when including an
operation or filepath. Fields are joined with spaces, rejecting CR, LF and
backticks that would break the opening fence. No class instance is needed.

```python
from sxpb_llm import format_code_block

view = format_code_block("print(1)", language="py", operation="<", filepath="a.py")
```

Content spaces and line endings are preserved, adding an LF only when nonempty
content lacks a final LF. Empty content stays empty; the closing fence has no
trailing newline. This formats a new block, not the original spelling of parsed
boundaries; use `opening + content + closing` to reconstruct an existing block.
Applications own transport framing: a protocol that removes one framing LF
should explicitly format `content + "\n"` instead. Formatting grants no authority.

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
