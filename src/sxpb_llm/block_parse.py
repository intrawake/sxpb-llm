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
        content: Exact text between the fences, including trailing whitespace
            and the final line ending. For prose, the exact text between blocks.
        error: An error classification if the block was malformed, or
            ``None`` if the block was parsed cleanly.  See :class:`CodeBlockError`.
            Callers can use this to reject blocks that don't meet their
            strictness requirements.
        sxpb: Parsed SxPB content when ``parse_code_blocks(..., parse_sxpb=True)``
            is used and this block is parseable as SxPB; otherwise ``None``.
        sxpb_error: The exception raised while parsing SxPB content, or ``None``.
        opening: Exact opening fence line, including its line ending;
            ``None`` for prose.
        closing: Exact closing fence line, including its line ending;
            ``None`` for prose or an unmatched opening fence.
    """

    language: str
    operation: str | None
    filepath: str | None
    content: str
    error: CodeBlockError | None = None
    sxpb: Any = None
    sxpb_error: Exception | None = None
    opening: str | None = None
    closing: str | None = None


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

    Content and original fence lines are preserved exactly. Only LF delimits
    physical lines (CRLF is recognized); bare CR and Unicode separators remain
    data. Unfenced operation-looking lines remain prose. No newline is added or
    removed. Concatenating each block's ``(opening or '') + content +
    (closing or '')`` reconstructs the input.

    Args:
        text: The markdown text to parse.
        parse_sxpb: If true, parse SxPB-looking blocks and store the result on
            ``CodeBlock.sxpb``. Parsing is attempted for ``sxpb`` and untagged
            fenced blocks.
        precise: Passed through to :func:`sxpb.loads` when ``parse_sxpb`` is true.

    Returns:
        A list of ``CodeBlock`` instances in document order.
    """
    if not text:
        return []

    blocks: list[CodeBlock] = []
    # Unlike splitlines(), bare CR and Unicode separators remain data.
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])

    def match_line(line: str) -> str:
        if line.endswith("\n"):
            return line[:-2] if line.endswith("\r\n") else line[:-1]
        return line

    i = 0
    prose_start = 0

    # Opening fence: optional indent, backticks/tildes, optional info string.
    _fence_re = re.compile(r"^[ \t]*(```+|~~~+)[ \t]*(\S*)[ \t]*(.*?)[ \t]*$")

    while i < len(lines):
        m = _fence_re.match(match_line(lines[i]))
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

        opening = lines[i]
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
            if close_pattern.match(match_line(lines[i])):
                break
            i += 1

        if i < len(lines):
            # closing fence found
            content = "".join(lines[content_start:i])
            closing = lines[i]
            i += 1  # skip closing fence line
            error = None
        else:
            # no closing fence — take rest as content (lenient)
            content = "".join(lines[content_start:])
            error = CodeBlockError.UNMATCHED_FENCE
            closing = None

        blocks.append(
            CodeBlock(
                language=language,
                operation=operation,
                filepath=filepath,
                content=content,
                error=error,
                opening=opening,
                closing=closing,
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

    if parse_sxpb:
        _parse_sxpb_blocks(blocks, precise=precise)
    return blocks


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
    # Untagged fenced code blocks have an empty language and no prose error.
    return block.language == "" and block.error is None
