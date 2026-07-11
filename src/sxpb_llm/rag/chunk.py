"""Pure text chunkers for retrieval indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Chunk:
    """A piece of source text and its human-readable section label."""

    section: str
    content: str


Chunker = Callable[[str, str, int], list[Chunk]]


def _pack_parts(parts: list[str], section: str, max_chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{current}\n\n{part}" if current else part
        if current and len(candidate) > max_chunk_size:
            chunks.append(Chunk(section, current))
            current = part
        else:
            current = candidate
    if current:
        chunks.append(Chunk(section, current))
    return chunks


def chunk_plain(
    text: str, filepath: str = "document", max_chunk_size: int = 3000
) -> list[Chunk]:
    """Split text on paragraph boundaries, preserving oversized paragraphs."""
    stripped = text.strip()
    if not stripped:
        return []
    section = Path(filepath).name
    return _pack_parts(re.split(r"\n\s*\n", stripped), section, max_chunk_size)


def chunk_markdown(
    text: str, filepath: str = "document.md", max_chunk_size: int = 3000
) -> list[Chunk]:
    """Split Markdown at level-two headings, then oversized sections by paragraph."""
    chunks: list[Chunk] = []
    for raw_section in re.split(r"^(?=## )", text, flags=re.MULTILINE):
        section = raw_section.strip()
        if not section:
            continue
        first_line = section.split("\n", 1)[0]
        label = first_line if first_line.startswith("##") else Path(filepath).name
        if len(section) <= max_chunk_size:
            chunks.append(Chunk(label, section))
        else:
            chunks.extend(_pack_parts(section.split("\n\n"), label, max_chunk_size))
    return chunks or chunk_plain(text, filepath, max_chunk_size)


def chunk_yaml_list(
    text: str, filepath: str = "document.yaml", max_chunk_size: int = 3000
) -> list[Chunk]:
    """Chunk a top-level YAML sequence by item while retaining original text.

    This intentionally recognizes only unindented ``- `` item boundaries. Other
    YAML shapes fall back to plain text and do not require a YAML parser.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("- ") or line == "-"]
    if not starts:
        return chunk_plain(text, filepath, max_chunk_size)
    prefix = "\n".join(lines[: starts[0]]).strip()
    items: list[str] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        item = "\n".join(lines[start:end]).strip()
        if item:
            items.append(item)
    if prefix and items:
        items[0] = f"{prefix}\n{items[0]}"
    return _pack_parts(items, Path(filepath).name, max_chunk_size)


def _sxpb_form_spans(text: str, target_depth: int = 0) -> list[tuple[int, int]]:
    """Locate balanced forms while respecting strings and semicolon comments."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    in_comment = False
    for i, char in enumerate(text):
        if in_comment:
            if char == "\n":
                in_comment = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == ";":
            in_comment = True
        elif char == '"':
            in_string = True
        elif char == "(":
            if depth == target_depth:
                start = i
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return []
            if depth == target_depth and start is not None:
                spans.append((start, i + 1))
                start = None
    if depth != 0 or in_string:
        return []
    return spans


def _sxpb_forms(text: str, target_depth: int = 0) -> list[str]:
    """Split at balanced forms without dropping comments or surrounding text."""
    spans = _sxpb_form_spans(text, target_depth)
    if not spans:
        return []
    forms: list[str] = []
    cursor = 0
    for _start, end in spans:
        forms.append(text[cursor:end].strip())
        cursor = end
    trailing = text[cursor:].strip()
    if trailing:
        forms[-1] = f"{forms[-1]}\n{trailing}"
    return forms


def _sxpb_label(form: str, fallback: str) -> str:
    without_comments = re.sub(r"(?m)^\s*;.*$", "", form).lstrip()
    match = re.match(r"\(\s*([^\s()]+)", without_comments)
    return match.group(1).strip('"') if match else fallback


def chunk_sxpb(
    text: str, filepath: str = "document.sxpb", max_chunk_size: int = 3000
) -> list[Chunk]:
    """Chunk SxPB on balanced top-level forms, then direct child forms.

    It is a lexical structural chunker rather than a schema-specific one. Large
    single nests are split at direct child boundaries; malformed data falls back
    to paragraph chunking.
    """
    fallback = Path(filepath).name
    forms = _sxpb_forms(text)
    if not forms:
        return chunk_plain(text, filepath, max_chunk_size)

    chunks: list[Chunk] = []
    for form in forms:
        label = _sxpb_label(form, fallback)
        if len(form) <= max_chunk_size:
            chunks.append(Chunk(label, form.strip()))
            continue
        children = _sxpb_forms(form, target_depth=1)
        # A form with no useful children cannot be structurally divided.
        if len(children) < 2:
            chunks.extend(chunk_plain(form, filepath, max_chunk_size))
            continue
        for packed in _pack_parts(children, label, max_chunk_size):
            chunks.append(packed)
    return chunks


def chunk_file(path: str | Path, max_chunk_size: int = 3000) -> list[Chunk]:
    """Read and chunk a supported file, falling back to plain text."""
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    by_suffix: dict[str, Chunker] = {
        ".md": chunk_markdown,
        ".markdown": chunk_markdown,
        ".sxpb": chunk_sxpb,
        ".yaml": chunk_yaml_list,
        ".yml": chunk_yaml_list,
    }
    chunker = by_suffix.get(source.suffix.lower(), chunk_plain)
    return chunker(text, str(source), max_chunk_size)
