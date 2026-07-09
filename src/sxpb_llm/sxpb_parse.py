"""Extract SxPB content from LLM responses.

Merges the two complementary approaches from sxpb-game and subllminal:

1. **Structured block parsing**: Uses :func:`sxpb_llm.parse_code_blocks`
   with ``parse_sxpb=True`` to find parsed SxPB blocks.
2. **Line-growth fallback** (subllminal): Scans for bare ``(answer ...)``
   lines and grows them until a valid SxPB parse is found.
"""

from typing import Any, cast

import sxpb

from sxpb_llm.block_parse import CodeBlockError, parse_code_blocks


def get_sxpb_from_markdown(text: str) -> dict | None:
    """Extract content from a markdown code block and parse it as SxPB.

    Kept for backward compatibility.  Prefer :func:`parse_code_blocks` with
    ``parse_sxpb=True`` for new code that already needs block metadata.

    Returns:
        The parsed SxPB dict, or ``None`` if no block was found or parsing
        failed/non-dict content was found.
    """
    if not text:
        return None

    blocks = parse_code_blocks(text, parse_sxpb=True)

    # Prefer explicitly tagged SxPB blocks.
    for block in blocks:
        if block.language == "sxpb" and isinstance(block.sxpb, dict):
            return cast(dict, block.sxpb)

    # Compatibility fallback: accept untagged fenced / bare parseable blocks.
    for block in blocks:
        if block.error is not CodeBlockError.NOT_FENCED and isinstance(
            block.sxpb, dict
        ):
            return cast(dict, block.sxpb)

    return None


def parse_sxpb_answer(response: str) -> str | None:
    """Parse an ``(answer "...")`` value from an LLM response.

    Strategy (tries in order, returns first success):

    1. **Structured block** (sxpb-game format):
       Searches for ``sxpb > /dev/stdout`` blocks (handles both
       space-separated and attached ``>/dev/stdout``) and looks for an
       ``(answer ...)`` record inside.  The last parseable matching block wins.

    2. **Line-growth** (subllminal format):
       Finds lines starting with ``(answer`` and grows them until a valid
       SxPB parse yields an ``answer`` key.

    Returns:
        The answer string, or ``None`` if nothing could be parsed.
    """
    if not response:
        return None

    # ── Strategy 1: sxpb > /dev/stdout blocks ─────────────────────────
    blocks = parse_code_blocks(response, parse_sxpb=True)
    for block in reversed(blocks):
        if (
            block.language == "sxpb"
            and block.operation == ">"
            and block.filepath == "/dev/stdout"
            and isinstance(block.sxpb, dict)
            and "answer" in block.sxpb
        ):
            answer = block.sxpb["answer"]
            if answer and str(answer).strip():
                return str(answer).strip()

    # ── Strategy 2: bare (answer ...) line-growth ──────────────────────
    lines = response.splitlines()
    candidate_indices = [
        i
        for i, line in enumerate(lines)
        if line.startswith("(answer ") or line.startswith('(answer"')
    ]

    for start_idx in reversed(candidate_indices):
        for end_idx in range(start_idx, len(lines)):
            if not lines[end_idx].strip().endswith(")"):
                continue

            candidate_text = "\n".join(lines[start_idx : end_idx + 1])
            try:
                parsed: Any = sxpb.loads(candidate_text)
                if isinstance(parsed, dict) and "answer" in parsed:
                    content = parsed["answer"]
                    if content and str(content).strip():
                        return str(content).strip()
                break  # parsed, but not an answer dict — stop growing
            except Exception:
                continue  # keep growing

    return None
