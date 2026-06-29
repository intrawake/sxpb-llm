"""Extract SxPB content from LLM responses.

Merges the two complementary approaches from sxpb-game and subllminal:

1. **Fenced block** (sxpb-game): Uses :func:`sxpb_llm.parse_code_blocks`
   to find ```sxpb > /dev/stdout``` blocks — the exact format sxpb-game's
   harness instructs LLMs to use.
2. **Line-growth fallback** (subllminal): Scans for bare ``(answer ...)``
   lines and grows them until a valid SxPB parse is found.
"""

from typing import Any, cast

import sxpb

from sxpb_llm.block_parse import parse_code_blocks


def get_sxpb_from_markdown(text: str) -> dict | None:
    """Extract content from a ```sxpb ... ``` code block and parse it.

    If no ``sxpb``-tagged block is found, falls back to the first untagged
    ``` block.

    Returns:
        The parsed SxPB dict, or ``None`` if no block was found or parsing
        failed.
    """
    if not text:
        return None

    blocks = parse_code_blocks(text)

    # Prefer sxpb-tagged blocks, then fall back to any code block.
    for block in blocks:
        if block.language == "sxpb":
            try:
                result: Any = sxpb.loads(block.content)
                if isinstance(result, dict):
                    return cast(dict, result)
            except Exception:
                pass

    for block in blocks:
        if block.language:
            try:
                result: Any = sxpb.loads(block.content)
                if isinstance(result, dict):
                    return cast(dict, result)
            except Exception:
                pass

    return None


def parse_sxpb_answer(response: str) -> str | None:
    """Parse an ``(answer "...")`` value from an LLM response.

    Strategy (tries in order, returns first success):

    1. **Fenced block** (sxpb-game format):
       Searches for ```sxpb > /dev/stdout ... ``` blocks (handles both
       space-separated and attached ``>/dev/stdout``) and looks for an
       ``(answer ...)`` record inside.  The *last* matching block wins.

    2. **Line-growth** (subllminal format):
       Finds lines starting with ``(answer`` and grows them until a valid
       SxPB parse yields an ``answer`` key.

    Returns:
        The answer string, or ``None`` if nothing could be parsed.
    """
    if not response:
        return None

    # ── Strategy 1: fenced sxpb > /dev/stdout blocks ───────────────────
    blocks = parse_code_blocks(response)
    # Last matching block wins (same as old matches[-1] behavior).
    for block in reversed(blocks):
        if (
            block.language == "sxpb"
            and block.operation == ">"
            and block.filepath == "/dev/stdout"
        ):
            try:
                parsed_block = sxpb.loads(block.content)
                if isinstance(parsed_block, dict) and "answer" in parsed_block:
                    answer = parsed_block["answer"]
                    if answer and str(answer).strip():
                        return str(answer).strip()
            except Exception:
                pass
            break  # Only the last sxpb > /dev/stdout block is considered

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
                parsed = sxpb.loads(candidate_text)
                if isinstance(parsed, dict) and "answer" in parsed:
                    content = parsed["answer"]
                    if content and str(content).strip():
                        return str(content).strip()
                break  # parsed, but not an answer dict — stop growing
            except Exception:
                continue  # keep growing

    return None
