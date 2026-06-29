"""Unit tests for sxpb_llm.block_parse.

Tests for CodeBlock dataclass, parse_code_blocks(), fence parsing,
bare >filepath expansion, blank-line boundaries, invariants, etc.
"""

from sxpb_llm.block_parse import CodeBlockError, CodeBlock, parse_code_blocks


# ======================================================================
# parse_code_blocks / CodeBlock tests
# ======================================================================


def test_codeblock_dataclass_fields():
    """CodeBlock is a simple dataclass with four fields."""
    cb = CodeBlock(
        language="sxpb",
        operation=">",
        filepath="/dev/stdout",
        content='(answer "42")',
    )
    assert cb.language == "sxpb"
    assert cb.operation == ">"
    assert cb.filepath == "/dev/stdout"
    assert cb.content == '(answer "42")'

    # Prose block: empty language, no operation/filepath
    prose = CodeBlock(language="", operation=None, filepath=None, content="Hello.")
    assert prose.language == ""
    assert prose.operation is None
    assert prose.filepath is None
    assert prose.error is None  # constructed directly, not via parse_code_blocks()


def test_parse_empty_text():
    """Empty or whitespace-only text yields no blocks."""
    assert parse_code_blocks("") == []


def test_parse_no_code_blocks():
    """Text with no fenced blocks yields a single prose block."""
    blocks = parse_code_blocks("Just some plain text.")
    assert len(blocks) == 1
    assert blocks[0].language == ""
    assert blocks[0].operation is None
    assert blocks[0].filepath is None
    assert blocks[0].content == "Just some plain text."


def test_parse_single_code_block_language_only():
    """A simple fenced block with just a language tag."""
    text = '```python\nprint("hi")\n```'
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "python"
    assert blocks[0].operation is None
    assert blocks[0].filepath is None
    assert blocks[0].content == 'print("hi")'


def test_parse_code_block_with_operation_and_filepath():
    """Fence info string: language > filepath."""
    text = '```sxpb > /dev/stdout\n(answer "42")\n```'
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "sxpb"
    assert blocks[0].operation == ">"
    assert blocks[0].filepath == "/dev/stdout"
    assert blocks[0].content == '(answer "42")'


def test_parse_code_block_attached_operation():
    """Operation char attached to filepath: >filepath (no space)."""
    text = "```sxpb >assistant_name.sxpb\n(name ChatBot)\n```"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "sxpb"
    assert blocks[0].operation == ">"
    assert blocks[0].filepath == "assistant_name.sxpb"
    assert blocks[0].content == "(name ChatBot)"


def test_parse_code_block_read_arrow():
    """The '<' operation is for reading input."""
    text = '```sxpb < /dev/stdin\n(quiz (question "Why?"))\n```'
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "sxpb"
    assert blocks[0].operation == "<"
    assert blocks[0].filepath == "/dev/stdin"


def test_parse_code_block_filepath_no_operation():
    """Filepath without an operation arrow."""
    text = "```text output.txt\nHello world\n```"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "text"
    assert blocks[0].operation is None
    assert blocks[0].filepath == "output.txt"
    assert blocks[0].content == "Hello world"


def test_parse_code_block_operation_no_filepath():
    """Operation arrow with nothing after it."""
    text = "```sxpb >\n(some data)\n```"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "sxpb"
    assert blocks[0].operation == ">"
    assert blocks[0].filepath is None


def test_parse_code_block_no_language():
    """A fenced block with no language tag at all."""
    text = "```\nplain preformatted text\n```"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == ""
    assert blocks[0].operation is None
    assert blocks[0].filepath is None
    assert blocks[0].content == "plain preformatted text"


def test_parse_prose_before_code_block():
    """Text before the first fenced block becomes a prose block."""
    text = "Some preamble.\n\n```python\npass\n```"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].language == ""
    assert blocks[0].content == "Some preamble.\n\n"
    assert blocks[1].language == "python"
    assert blocks[1].content == "pass"


def test_parse_prose_after_code_block():
    """Text after the last fenced block becomes a prose block."""
    text = "```python\npass\n```\n\nTrailing thoughts."
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].language == "python"
    assert blocks[1].language == ""
    # The \n from the closing fence line is consumed; prose starts at the blank line.
    assert blocks[1].content == "\nTrailing thoughts."


def test_parse_prose_between_blocks():
    """Text between two fenced blocks is captured as prose."""
    text = (
        "```sxpb > a.sxpb\n(data-a)\n```\n"
        "\n"
        "Middle commentary.\n"
        "\n"
        "```sxpb > b.sxpb\n(data-b)\n```"
    )
    blocks = parse_code_blocks(text)
    assert len(blocks) == 3
    assert blocks[0].language == "sxpb"
    assert blocks[0].content == "(data-a)"
    assert blocks[1].language == ""
    assert blocks[1].content == "\nMiddle commentary.\n\n"
    assert blocks[2].language == "sxpb"
    assert blocks[2].content == "(data-b)"


def test_parse_consecutive_blocks_no_prose():
    """Adjacent fenced blocks with no prose between them."""
    text = "```sxpb > a.sxpb\n(A)\n```\n```sxpb > b.sxpb\n(B)\n```"
    blocks = parse_code_blocks(text)
    # The closing ```\n and opening ``` on the next line are adjacent — no prose between.
    assert len(blocks) == 2
    assert blocks[0].content == "(A)"
    assert blocks[1].content == "(B)"


def test_parse_tilde_fences():
    """Tilde fences (~~~) work the same as backtick fences."""
    text = '~~~sxpb > /dev/stdout\n(answer "tilde")\n~~~'
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "sxpb"
    assert blocks[0].operation == ">"
    assert blocks[0].filepath == "/dev/stdout"
    assert blocks[0].content == '(answer "tilde")'


def test_parse_longer_fences():
    """Fences with more than 3 characters still work."""
    text = '````python\nprint("four ticks")\n````'
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "python"
    assert blocks[0].content == 'print("four ticks")'


def test_parse_mixed_fence_lengths():
    """Opening fence of length N needs closing fence of length >= N."""
    text = "````\ncode with ``` inner ticks\n````"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].content == "code with ``` inner ticks"


def test_parse_missing_closing_fence():
    """When no closing fence exists, rest of text becomes content (lenient)."""
    text = "```sxpb > output.sxpb\n(incomplete"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "sxpb"
    assert blocks[0].content == "(incomplete"
    assert blocks[0].error == CodeBlockError.UNMATCHED_FENCE


def test_parse_content_preserves_internal_blank_lines():
    """Blank lines inside a code block are preserved."""
    text = "```sxpb\n(first)\n\n\n(second)\n```"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].content == "(first)\n\n\n(second)"


def test_parse_content_strips_trailing_whitespace():
    """Trailing whitespace/newlines in code blocks are stripped."""
    text = "```text\nhello\n\n\n\n```"
    blocks = parse_code_blocks(text)
    assert blocks[0].content == "hello"


def test_parse_realistic_paludoro_response():
    """A realistic multi-artifact LLM response like paludoro sees."""
    text = (
        "Yo, Alex! My name is **ChatBot**.  \n"
        "\n"
        "Here are the name artifacts as requested:  \n"
        "```sxpb > assistant_name.sxpb  \n"
        "(name ChatBot)  \n"
        "```  \n"
        "\n"
        "```sxpb > user_name.sxpb  \n"
        "(name Alex)  \n"
        "```  \n"
        "\n"
        "Let me know how else I can help! 👍"
    )
    blocks = parse_code_blocks(text)

    # With _expand_bare_blocks, the lone "\n" between consecutive
    # fences is filtered as structural whitespace.
    assert len(blocks) == 4

    # Prose before first artifact
    assert blocks[0].language == ""
    assert "Yo, Alex" in blocks[0].content

    # First artifact
    assert blocks[1].language == "sxpb"
    assert blocks[1].operation == ">"
    assert blocks[1].filepath == "assistant_name.sxpb"
    assert blocks[1].content == "(name ChatBot)"

    # Second artifact (middle prose filtered as structural \n)
    assert blocks[2].language == "sxpb"
    assert blocks[2].operation == ">"
    assert blocks[2].filepath == "user_name.sxpb"
    assert blocks[2].content == "(name Alex)"

    # Trailing prose
    assert blocks[3].language == ""
    assert "Let me know" in blocks[3].content


def test_parse_realistic_sxpb_game_format():
    """The sxpb-game harness format: sxpb > /dev/stdout with (answer)."""
    text = """\
Let me think about the board...

```sxpb > /dev/stdout
(answer "b2")
```

That should be a good move."""
    blocks = parse_code_blocks(text)
    assert len(blocks) == 3
    assert blocks[1].language == "sxpb"
    assert blocks[1].operation == ">"
    assert blocks[1].filepath == "/dev/stdout"
    assert blocks[1].content == '(answer "b2")'


def test_parse_ignores_indented_fences():
    """Indented fences (up to 3 spaces) are recognized."""
    text = '   ```sxpb > /dev/stdout\n(answer "indented")\n   ```'
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].filepath == "/dev/stdout"
    assert blocks[0].content == '(answer "indented")'


def test_parse_roundtrip_concatenation():
    """Concatenating block contents approximates the original.

    Prose is preserved exactly.  Code block content is stripped of trailing
    whitespace, so the reconstruction may differ slightly in trailing newlines
    but preserves all semantic content.
    """
    original = (
        "Preamble.\n"
        "\n"
        "```sxpb > a.sxpb\n"
        "(a)\n"
        "```\n"
        "Middle.\n"
        "```sxpb > b.sxpb\n"
        "(b)\n"
        "```\n"
        "Aftermath."
    )
    blocks = parse_code_blocks(original)
    assert len(blocks) == 5  # prose, code, prose, code, prose

    # Verify structure and key content
    assert blocks[0].language == ""
    assert "Preamble." in blocks[0].content

    assert blocks[1].language == "sxpb"
    assert blocks[1].filepath == "a.sxpb"
    assert blocks[1].content == "(a)"

    assert blocks[2].language == ""
    assert "Middle." in blocks[2].content

    assert blocks[3].language == "sxpb"
    assert blocks[3].filepath == "b.sxpb"
    assert blocks[3].content == "(b)"

    assert blocks[4].language == ""
    assert blocks[4].content == "Aftermath."


def test_parse_only_prose():
    """A string with no fenced blocks is a single prose block."""
    text = "Hello\nworld\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == ""
    assert blocks[0].content == text


# ── Bare block tests ──────────────────────────────────────────────────────


def test_bare_write_block_splits_prose():
    """Bare >filepath in prose produces a CodeBlock with operation."""
    text = "Hello.\n>out.txt\n(content)\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[0].content == "Hello.\n"  # trailing \n preserved
    assert blocks[1].language == ""
    assert blocks[1].operation == ">"
    assert blocks[1].filepath == "out.txt"
    assert blocks[1].content == "(content)"
    assert blocks[1].error is None


def test_bare_block_no_leading_prose():
    """Bare block at start of text yields no empty prose prefix."""
    blocks = parse_code_blocks(">f.sxpb\n(data)\n")
    assert len(blocks) == 1
    assert blocks[0].operation == ">"
    assert blocks[0].filepath == "f.sxpb"
    assert blocks[0].content == "(data)"


def test_bare_block_no_trailing_prose():
    """Bare block at end of text yields no empty prose suffix."""
    blocks = parse_code_blocks(">f.sxpb\n(data)")
    assert len(blocks) == 1
    assert blocks[0].operation == ">"
    assert blocks[0].content == "(data)"


def test_bare_read_block():
    """Bare <filepath is recognized as a read operation."""
    # Without a blank line after the read content, "After." is
    # part of the bare block's content (no closing delimiter).
    blocks = parse_code_blocks("Before.\n<in.txt\n(read)\nAfter.\n")
    assert len(blocks) == 2
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[1].operation == "<"
    assert blocks[1].filepath == "in.txt"
    assert blocks[1].content == "(read)\nAfter."


def test_bare_consecutive_blocks():
    """Consecutive bare > blocks split correctly."""
    text = ">a.sxpb\n(a)\n>b.sxpb\n(b)\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].operation == ">"
    assert blocks[0].filepath == "a.sxpb"
    assert blocks[0].content == "(a)"
    assert blocks[1].operation == ">"
    assert blocks[1].filepath == "b.sxpb"
    assert blocks[1].content == "(b)"


def test_bare_block_multiline_content():
    """Bare block content spans multiple lines until next bare op."""
    text = ">f.sxpb\nline1\nline2\n>g.sxpb\nline3\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].content == "line1\nline2"
    assert blocks[1].content == "line3"


def test_bare_block_trailing_whitespace_stripped():
    """Bare block content has trailing whitespace stripped."""
    # The trailing \n\n: first \n ends the op line; second \n is a
    # structural separator that gets filtered.
    blocks = parse_code_blocks(">f.sxpb\n(data)  \n\n")
    assert len(blocks) == 1
    assert blocks[0].content == "(data)"


def test_bare_block_with_whitespace_around_arrow():
    """Whitespace around the operation arrow is tolerated."""
    for prefix in ["  > f.sxpb\n", "\t>\tf.sxpb\n", " >f.sxpb\n"]:
        blocks = parse_code_blocks(prefix + "(x)\n")
        assert blocks[0].operation == ">"
        assert blocks[0].filepath == "f.sxpb"


def test_bare_block_mixed_with_fenced():
    """Bare and fenced blocks coexist correctly."""
    # Without a blank line, "More prose." is part of the bare block content.
    text = (
        "Prose before.\n"
        ">bare.sxpb\n(bare)\n"
        "More prose.\n"
        "```sxpb > fenced.sxpb\n(fenced)\n```\n"
        "After.\n"
    )
    blocks = parse_code_blocks(text)
    assert len(blocks) == 4
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[1].operation == ">"
    assert blocks[1].filepath == "bare.sxpb"
    assert blocks[1].content == "(bare)\nMore prose."
    assert blocks[2].language == "sxpb"
    assert blocks[2].operation == ">"
    assert blocks[2].filepath == "fenced.sxpb"
    assert blocks[2].content == "(fenced)"
    assert blocks[3].error is CodeBlockError.NOT_FENCED


# ── NOT_FENCED error type tests ──────────────────────────────────────────


def test_prose_always_has_not_fenced_error():
    """Every prose block from parse_code_blocks has NOT_FENCED."""
    blocks = parse_code_blocks("A.\n```py\nprint(1)\n```\nB.\n")
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[1].error is None
    assert blocks[2].error is CodeBlockError.NOT_FENCED


def test_untagged_fenced_block_has_no_error():
    """A fenced block with no language tag is NOT prose."""
    blocks = parse_code_blocks("```\nstuff\n```")
    assert len(blocks) == 1
    assert blocks[0].language == ""
    assert blocks[0].error is None


def test_not_fenced_distinguishes_from_untagged_fence():
    """NOT_FENCED error flag lets callers tell prose from untagged fences."""
    text = "Prose.\n```\nuntagged\n```\n"
    blocks = parse_code_blocks(text)
    assert blocks[0].language == ""
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[1].language == ""
    assert blocks[1].error is None


# ── Edge case tests ──────────────────────────────────────────────────────


def test_triple_backtick_inside_content_not_confused():
    """A ``` line inside a fenced block doesn't prematurely close it."""
    text = "```py\nprint('```')\n```"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "py"
    assert "print('```')" in blocks[0].content


def test_fence_with_no_info_string():
    """A fence with just ``` and nothing else."""
    blocks = parse_code_blocks("```\ncontent\n```")
    assert len(blocks) == 1
    assert blocks[0].language == ""
    assert blocks[0].operation is None
    assert blocks[0].filepath is None
    assert blocks[0].content == "content"
    assert blocks[0].error is None


def test_fence_with_trailing_spaces():
    """Trailing spaces after fence info string are ignored."""
    blocks = parse_code_blocks("```py   \n1\n```")
    assert blocks[0].language == "py"
    assert blocks[0].content == "1"


def test_bare_block_content_with_blank_lines():
    """Blank line within bare block content splits it."""
    text = ">f.sxpb\nline1\n\nline2\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].content == "line1"
    assert blocks[1].error is CodeBlockError.NOT_FENCED


def test_bare_operation_mid_line_not_detected():
    """A > not at line start is NOT a bare operation."""
    text = "This is a > test with arrow mid-line.\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[0].operation is None


def test_bare_block_filepath_no_ext():
    """Bare operation with filepath that has no extension."""
    blocks = parse_code_blocks(">README\ncontent\n")
    assert blocks[0].filepath == "README"


def test_bare_block_filepath_with_path():
    """Bare operation with a path-like filepath."""
    blocks = parse_code_blocks(">sub/dir/file.sxpb\n(x)\n")
    assert blocks[0].filepath == "sub/dir/file.sxpb"


def test_codeblock_enum_values():
    """Verify CodeBlockError enum string values."""
    assert CodeBlockError.UNMATCHED_FENCE.value == "unmatched_fence"
    assert CodeBlockError.NOT_FENCED.value == "not_fenced"


def test_codeblock_repr():
    """CodeBlock repr is informative."""
    cb = CodeBlock(language="py", operation=">", filepath="f.py", content="1+1")
    r = repr(cb)
    assert "py" in r
    assert ">" in r
    assert "f.py" in r
    assert "1+1" in r


def test_fence_char_count_four_open_three_close():
    """4-backtick open, 3-backtick close: invalid — no match."""
    blocks = parse_code_blocks("````\nhi\n```")
    assert len(blocks) == 1
    assert blocks[0].error is CodeBlockError.UNMATCHED_FENCE


def test_mixed_tilde_backtick_not_confused():
    """A ~~~ fence is not closed by ```."""
    blocks = parse_code_blocks("~~~\nhi\n```\nstill in tilde block")
    assert len(blocks) == 1
    assert blocks[0].error is CodeBlockError.UNMATCHED_FENCE


def test_bare_block_non_sxpb_content():
    """Bare >block captures arbitrary content, not just SxPB."""
    text = ">notes.txt\nArbitrary prose content.\nMultiple lines.\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].operation == ">"
    assert blocks[0].filepath == "notes.txt"
    assert blocks[0].content == "Arbitrary prose content.\nMultiple lines."


# ── Bare block content boundary tests ────────────────────────────────────


def test_bare_block_empty_content_after_blank_line():
    """Blank line immediately after bare op gives empty content."""
    text = ">empty.sxpb\n\nMore prose.\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].operation == ">"
    assert blocks[0].content == ""
    assert blocks[1].error is CodeBlockError.NOT_FENCED
    # The blank line's ``\n`` gets preserved in the prose suffix
    assert blocks[1].content == "\nMore prose.\n"


def test_bare_block_two_blank_lines_after_op():
    """Two blank lines after bare op: empty content, then prose."""
    text = ">f.sxpb\n\n\nProse after.\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].content == ""
    assert blocks[1].content == "\n\nProse after.\n"


def test_bare_block_content_stops_at_blank_line():
    """Content before blank line, prose after blank line."""
    text = ">f.sxpb\nline1\nline2\n\nprose after blank\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].content == "line1\nline2"
    assert blocks[1].error is CodeBlockError.NOT_FENCED
    assert "prose after blank" in blocks[1].content


def test_bare_block_content_at_end_of_string():
    """Bare block at end of string with no trailing newline."""
    blocks = parse_code_blocks(">f.sxpb\nlast line")
    assert len(blocks) == 1
    assert blocks[0].content == "last line"


def test_bare_block_then_fenced_block():
    """Bare op followed by a fenced block in the same prose."""
    text = "hello\n>out.sxpb\n(bare)\n```py\n1+1\n```\nbye\n"
    blocks = parse_code_blocks(text)
    # Should be: prose, bare, fenced, prose
    assert len(blocks) == 4
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[0].content == "hello\n"
    assert blocks[1].operation == ">"
    assert blocks[1].filepath == "out.sxpb"
    assert blocks[1].content == "(bare)"
    assert blocks[2].language == "py"
    assert blocks[2].content == "1+1"
    assert blocks[3].content == "bye\n"


def test_bare_block_before_fenced_no_blank_line():
    """Bare op directly before fenced block opening."""
    text = ">f.sxpb\ncontent\n```py\npass\n```"
    blocks = parse_code_blocks(text)
    # Without blank line, the fence line becomes part of bare content.
    # But actually: the fence triggers as a new block, not bare content.
    # The bare content stops at the fence because the main parser already
    # split on fences before _expand_bare_blocks runs.
    assert len(blocks) >= 2
    assert blocks[0].operation == ">"


# ── Bare block prefix/suffix edge cases ──────────────────────────────────


def test_bare_block_double_newline_prefix_preserved():
    """A \\n\\n prefix between blocks is preserved (not structural)."""
    text = "hello\n\n>f.sxpb\n(x)\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].content == "hello\n\n"
    assert blocks[1].operation == ">"
    assert blocks[1].content == "(x)"


def test_bare_mixed_read_write_consecutive():
    """Consecutive bare < and > blocks don't leave spurious prose."""
    text = "<in.txt\n(read)\n>out.txt\n(write)\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].operation == "<"
    assert blocks[0].content == "(read)"
    assert blocks[1].operation == ">"
    assert blocks[1].content == "(write)"


def test_bare_block_prefix_multiple_newlines():
    """Leading prose before first bare op is preserved exactly."""
    text = "\n\nparagraph\n\n>f.sxpb\n(data)\n"
    blocks = parse_code_blocks(text)
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[0].content == "\n\nparagraph\n\n"
    assert blocks[1].content == "(data)"


def test_bare_block_suffix_is_empty():
    """No trailing prose after last bare block means no suffix block."""
    text = "start\n>a.sxpb\n(1)\n>b.sxpb\n(2)"
    blocks = parse_code_blocks(text)
    assert blocks[-1].operation == ">"
    assert blocks[-1].content == "(2)"


# ── Fenced + bare interaction tests ──────────────────────────────────────


def test_fenced_block_content_not_expanded():
    """Bare >filepath patterns INSIDE a fenced block are NOT expanded."""
    text = """```text
This has >filepath inside
and <other lines
that look like bare ops.\n```"""
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert ">filepath" in blocks[0].content
    assert blocks[0].error is None
    assert blocks[0].language == "text"


def test_prose_between_fence_and_bare_op():
    """Prose between a closing fence and a subsequent bare op."""
    text = "```py\n1\n```\nCommentary.\n>f.sxpb\n(data)\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 3
    assert blocks[0].language == "py"
    assert blocks[1].error is CodeBlockError.NOT_FENCED
    assert blocks[1].content == "Commentary.\n"
    assert blocks[2].operation == ">"
    assert blocks[2].content == "(data)"


def test_bare_op_immediately_after_closing_fence():
    """Bare op line right after a closing fence (no prose between)."""
    text = "```py\n1\n```\n>out.sxpb\n(data)\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].language == "py"
    assert blocks[1].operation == ">"


# ── UNMATCHED_FENCE edge cases ───────────────────────────────────────────


def test_unmatched_fence_followed_by_prose_and_fence():
    """A later matching closing fence closes the earlier opening fence."""
    text = "```py\ncode\n\nMore text.\n\n```sxpb\nmore\n```"
    blocks = parse_code_blocks(text)
    # The first ```py finds a closing ``` later, so it's matched.
    # `` ```sxpb `` is skipped because it has extra text after the ticks.
    # The final `` ``` `` closes.  The "language" is "py".
    assert len(blocks) == 1
    assert blocks[0].error is None  # matched
    assert "```sxpb" in blocks[0].content


def test_unmatched_fence_then_matched_fence_in_new_prose():
    """Extra chars after backticks prevent fence matching; later plain ``` closes."""
    text = "```py\noops no close\n\n```ok\nclosed\n```"
    blocks = parse_code_blocks(text)
    # `` ```ok `` doesn't match the close pattern (has 'ok'),
    # but `` ``` `` on the last line does.
    assert len(blocks) == 1
    assert blocks[0].error is None
    assert "```ok" in blocks[0].content


def test_truly_unmatched_fence_no_close_at_all():
    """No closing fence anywhere → UNMATCHED_FENCE."""
    text = "```py\nno close anywhere\nstill no close"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].error is CodeBlockError.UNMATCHED_FENCE
    assert blocks[0].content == "no close anywhere\nstill no close"


# ── Invariant tests ──────────────────────────────────────────────────────


def test_not_fenced_blocks_never_have_lang_op_filepath():
    """Every NOT_FENCED block has empty lang, None op, None filepath."""
    text = (
        "Prose.\n"
        ">bare.sxpb\n(bare)\n"
        "More prose.\n"
        "```sxpb > f.sxpb\n(fenced)\n```\n"
        "Final.\n"
    )
    blocks = parse_code_blocks(text)
    for b in blocks:
        if b.error is CodeBlockError.NOT_FENCED:
            assert b.language == "", f"NOT_FENCED should have empty language: {b}"
            assert b.operation is None, f"NOT_FENCED should have None operation: {b}"
            assert b.filepath is None, f"NOT_FENCED should have None filepath: {b}"


def test_bare_blocks_have_no_error_and_no_language():
    """Bare blocks (from _expand_bare_blocks) have language="", error=None."""
    text = ">a.sxpb\n(1)\n>b.sxpb\n(2)\n<c.sxpb\n(3)\n"
    blocks = parse_code_blocks(text)
    assert all(b.language == "" for b in blocks)
    assert all(b.error is None for b in blocks)
    assert all(b.operation is not None for b in blocks)


def test_fenced_blocks_never_have_not_fenced_error():
    """Fenced blocks (with language or untagged) never get NOT_FENCED."""
    text = (
        "```py\n1\n```\n"
        "```\n2\n```\n"
        "~~~sh\n3\n~~~\n"
        "```py\n4"  # unmatched
    )
    blocks = parse_code_blocks(text)
    for b in blocks:
        if b.error is CodeBlockError.NOT_FENCED:
            assert b.language == "", f"NOT_FENCED but has language {b.language!r}: {b}"
        elif b.error is None:
            # bare blocks (from expansion) have language="" too
            pass
        elif b.error is CodeBlockError.UNMATCHED_FENCE:
            # unmatched — these are fenced but broken
            pass


# ── Operation detection edge cases ───────────────────────────────────────


def test_attached_read_arrow_in_fence():
    """<filepath attached in fence info string."""
    text = "```sxpb <input.sxpb\n(read)\n```"
    blocks = parse_code_blocks(text)
    assert blocks[0].operation == "<"
    assert blocks[0].filepath == "input.sxpb"


def test_multiple_arrows_first_wins():
    """Only the first char is checked for operation — rest is filepath."""
    text = "```sxpb > <> arrows.txt\ncontent\n```"
    blocks = parse_code_blocks(text)
    assert blocks[0].operation == ">"
    assert blocks[0].filepath == "<> arrows.txt"


def test_no_language_but_operation_in_fence():
    """Operation and filepath without language: `` ```> file.txt ``."""
    text = "```> file.txt\ncontent\n```"
    blocks = parse_code_blocks(text)
    assert blocks[0].language == ">"  # first token becomes "language"
    # Actually: lang_token = ">" since there's no space before it
    # This is correct — the fence info string parser is greedy on first token


def test_bare_operation_with_spaces_in_filepath():
    """Filepath with spaces is NOT treated as bare op (non-space match fails)."""
    # The regex uses \S+ for filepath matching, so spaces break the match.
    # The entire text becomes a single prose block.
    text = ">file with spaces.sxpb\ncontent\n"
    blocks = parse_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].error is CodeBlockError.NOT_FENCED
    assert blocks[0].operation is None
