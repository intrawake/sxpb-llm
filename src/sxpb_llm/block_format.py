"""Markdown code-block formatting, independent of application protocols."""

import re


def format_code_block(
    content: str,
    *,
    language: str = "",
    operation: str | None = None,
    filepath: str | None = None,
) -> str:
    """Return a Markdown fenced-block string without trimming its content.

    Header fields use the same language, operation and filepath convention as
    ``parse_code_blocks``. Supply a language when using operation or filepath.
    Fields are joined with spaces; no filesystem or permission checks are made.
    CR, LF and backticks are rejected so fields cannot break the opening fence.

    Fences are at least three backticks and longer than any run in the content.
    Nonempty content lacking a final LF receives one before the closing fence;
    empty content stays empty. The closing fence has no trailing newline.
    Transport-specific framing belongs to callers: a protocol that removes
    one framing LF should pass the content followed by one LF explicitly.
    """
    info = " ".join(part for part in (language, operation, filepath) if part)
    if any(c in info for c in "\r\n`"):
        raise ValueError("header fields must not contain CR, LF or backticks")
    fence = "`" * max(
        3, 1 + max((len(run) for run in re.findall(r"`+", content)), default=0)
    )
    if content and not content.endswith("\n"):
        content += "\n"
    return f"{fence}{info}\n{content}{fence}"
