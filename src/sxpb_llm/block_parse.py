"""General markdown code block parser.

Extracts fenced code blocks (``` or ~~~) from markdown text,
parsing language, operation arrow, and filepath from the fence info string.
Prose between code blocks is also captured as CodeBlock instances.

SxPB content can optionally be parsed into the returned CodeBlock objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sxpb


class CodeBlockError(str, Enum):
    """Error classifications for malformed code blocks."""

    UNMATCHED_FENCE = "unmatched_fence"
    """Opening fence has no matching closing fence."""

    NOT_FENCED = "not_fenced"
    """Block is prose / non-code text between fenced blocks."""


# Characters recognized as operation arrows in fence info strings.
# "<" = read, ">" = write. Extend as new operations emerge.
_OPERATION_CHARS = frozenset({"<", ">"})


@dataclass
class CodeBlock:
    """A code block (or prose) extracted from markdown text.

    Attributes:
        language: The language tag, e.g. ``"sxpb"``, ``"python"``, ``"json"``.
            Empty string ``""`` for prose / non-code text between fenced blocks.
        operation: The operation arrow if present, e.g. ``"<"``, ``">"``,
            or ``None``.
        filepath: The file path if present, e.g. ``"/dev/stdout"``,
            ``"output.txt"``, or ``None``.
        content: The raw content of the block.  For fenced code blocks this
            is the text between the fences (trailing whitespace stripped).
            For prose blocks this is the literal text between fences,
            preserved exactly including leading/trailing whitespace.
        error: An error classification if the block was malformed, or
            ``None`` if the block was parsed cleanly.  See :class:`CodeBlockError`.
            Callers can use this to reject blocks that don't meet their
            strictness requirements.
        sxpb: Parsed SxPB content when ``parse_code_blocks(..., parse_sxpb=True)``
            is used and this block is parseable as SxPB; otherwise ``None``.
        sxpb_error: The exception raised while parsing SxPB content, or ``None``.
    """

    language: str
    operation: str | None
    filepath: str | None
    content: str
    error: CodeBlockError | None = None
    sxpb: Any = None
    sxpb_error: Exception | None = None


def parse_code_blocks(
    text: str,
    *,
    parse_sxpb: bool = False,
    precise: bool = False,
) -> list[CodeBlock]:
    """Parse markdown text into a list of :class:`CodeBlock` instances.

    Fenced code blocks (`` ``` `` or ``~~~``) are detected and their info
    string is parsed for language, optional operation arrow, and optional
    filepath.  Text that appears between fenced blocks is captured as prose
    ``CodeBlock``\\s with ``language=""``.

    The content of fenced code blocks has trailing whitespace stripped.
    Prose block content is preserved exactly as it appears in the input.

    Args:
        text: The markdown text to parse.
        parse_sxpb: If true, parse SxPB-looking blocks and store the result on
            ``CodeBlock.sxpb``.  Parsing is attempted for ``sxpb`` fenced blocks,
            untagged fenced blocks, and bare ``>file`` / ``<file`` blocks.
        precise: Passed through to :func:`sxpb.loads` when ``parse_sxpb`` is true.

    Returns:
        A list of ``CodeBlock`` instances in document order.
    """
    if not text:
        return []

    blocks: list[CodeBlock] = []
    lines = text.splitlines(keepends=True)
    i = 0
    prose_start = 0

    # Opening fence: optional indent, backticks/tildes, optional info string.
    _fence_re = re.compile(r"^[ \t]*(```+|~~~+)[ \t]*(\S*)[ \t]*(.*?)[ \t]*$")

    while i < len(lines):
        m = _fence_re.match(lines[i])
        if not m:
            i += 1
            continue

        # ── flush accumulated prose before this fence ──────────────
        if prose_start < i:
            prose = "".join(lines[prose_start:i])
            blocks.append(
                CodeBlock(
                    language="",
                    operation=None,
                    filepath=None,
                    content=prose,
                    error=CodeBlockError.NOT_FENCED,
                )
            )

        fence_str = m.group(1)  # e.g. "```" or "~~~~"
        fence_char = fence_str[0]  # '`' or '~'
        fence_len = len(fence_str)
        lang_token = m.group(2)  # e.g. "sxpb", "python", ""
        rest = m.group(3).strip()  # e.g. "> /dev/stdout"

        # ── parse info string: language, operation, filepath ──────
        language = lang_token
        operation: str | None = None
        filepath: str | None = None

        if rest:
            # Operation can be space-separated (``> /dev/stdout``) or
            # attached to the filepath (``>assistant_name.sxpb``).
            if rest[0] in _OPERATION_CHARS:
                operation = rest[0]
                filepath = rest[1:].strip() or None
            else:
                filepath = rest

        # ── find closing fence ─────────────────────────────────────
        i += 1
        content_start = i
        close_pattern = re.compile(
            r"^[ \t]*" + re.escape(fence_char) + "{" + str(fence_len) + r",}[ \t]*$"
        )

        while i < len(lines):
            if close_pattern.match(lines[i]):
                break
            i += 1

        if i < len(lines):
            # closing fence found
            content = "".join(lines[content_start:i])
            i += 1  # skip closing fence line
            error = None
        else:
            # no closing fence — take rest as content (lenient)
            content = "".join(lines[content_start:])
            error = CodeBlockError.UNMATCHED_FENCE

        blocks.append(
            CodeBlock(
                language=language,
                operation=operation,
                filepath=filepath,
                content=content.rstrip(),
                error=error,
            )
        )
        prose_start = i

    # ── flush remaining prose after last fence ─────────────────────
    if prose_start < len(lines):
        prose = "".join(lines[prose_start:])
        blocks.append(
            CodeBlock(
                language="",
                operation=None,
                filepath=None,
                content=prose,
                error=CodeBlockError.NOT_FENCED,
            )
        )

    blocks = _expand_bare_blocks(blocks)
    if parse_sxpb:
        _parse_sxpb_blocks(blocks, precise=precise)
    return blocks


_BARE_OP_RE = re.compile(
    r"(?:^|\n)[ \t]*([<>])\s*(\S+)[ \t]*\n",
    re.MULTILINE,
)
_BLANK_LINE_RE = re.compile(r"\n\n")


def _expand_bare_blocks(blocks: list[CodeBlock]) -> list[CodeBlock]:
    """Split NOT_FENCED prose blocks at bare ``>filepath`` boundaries.

    Within prose, a line like ``>filename.sxpb`` (or ``<filename.sxpb``)
    on its own is treated as a bare operation block.  The content following
    it (until the next bare operation or the end of the prose) becomes the
    block's content.
    """
    result: list[CodeBlock] = []

    for block in blocks:
        if block.error is not CodeBlockError.NOT_FENCED:
            result.append(block)
            continue

        prose = block.content
        last_end = 0
        for m in _BARE_OP_RE.finditer(prose):
            # Flush prose before this match.  The match includes a leading
            # newline (or start-of-string anchor), so we extend the prefix
            # to keep that newline in the prose.
            prefix_end = m.start()
            if prefix_end > 0:
                prefix_end += 1  # preserve the leading \n consumed by the match
            prefix = prose[last_end:prefix_end]
            # A single "\n" between consecutive bare blocks is
            # structural, not meaningful prose.
            if prefix and prefix != "\n":
                result.append(
                    CodeBlock(
                        language="",
                        operation=None,
                        filepath=None,
                        content=prefix,
                        error=CodeBlockError.NOT_FENCED,
                    )
                )

            operation = m.group(1)
            filepath = m.group(2)
            content_start = m.end()

            # Find where this bare block ends: at the next bare op line,
            # a blank line, or the end of the prose.  The blank-line
            # search starts one character before content_start because
            # the op line's trailing ``\n`` and a following ``\n`` form
            # the ``\n\n`` pair that straddles the boundary.
            next_match = _BARE_OP_RE.search(prose, content_start)
            blank_match = _BLANK_LINE_RE.search(prose, max(0, content_start - 1))
            candidates = []
            if next_match:
                candidates.append(next_match.start())
            if blank_match and blank_match.start() + 1 >= content_start:
                candidates.append(blank_match.start() + 1)  # after the first \n
            content_end = min(candidates) if candidates else len(prose)

            content = prose[content_start:content_end].rstrip()

            result.append(
                CodeBlock(
                    language="",
                    operation=operation,
                    filepath=filepath,
                    content=content,
                    error=None,
                )
            )
            last_end = content_end

        # Flush any remaining prose (skip lone "\n" separators).
        suffix = prose[last_end:]
        if suffix and suffix != "\n":
            result.append(
                CodeBlock(
                    language="",
                    operation=None,
                    filepath=None,
                    content=suffix,
                    error=CodeBlockError.NOT_FENCED,
                )
            )

    return result


def _parse_sxpb_blocks(blocks: list[CodeBlock], *, precise: bool) -> None:
    """Populate ``block.sxpb`` for blocks that are intended to contain SxPB."""
    for block in blocks:
        if not _should_parse_as_sxpb(block):
            continue

        try:
            block.sxpb = sxpb.loads(block.content, precise=precise)
            block.sxpb_error = None
        except Exception as e:
            block.sxpb = None
            block.sxpb_error = e


def _should_parse_as_sxpb(block: CodeBlock) -> bool:
    """Return whether optional SxPB parsing should be attempted for a block."""
    if block.error is CodeBlockError.NOT_FENCED:
        return False
    if block.language == "sxpb":
        return True
    # Untagged fenced code blocks and bare operation blocks both have an empty
    # language and no NOT_FENCED prose error.
    return block.language == "" and block.error is None
