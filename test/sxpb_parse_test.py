"""Tests for SxPB parsing layered onto markdown CodeBlock parsing."""

from sxpb_llm.block_parse import parse_code_blocks
from sxpb_llm.sxpb_parse import get_sxpb_from_markdown, parse_sxpb_answer


def test_parse_code_blocks_can_parse_sxpb_metadata_blocks():
    """Optional SxPB parsing populates CodeBlock.sxpb without a second parser."""
    text = """\
Preamble prose.

```sxpb < /dev/stdin
(question "move?")
```

```sxpb > /dev/stdout
(answer "b2")
```

>artifact.sxpb
(title "Report")
"""
    blocks = parse_code_blocks(text, parse_sxpb=True)
    parsed = [block for block in blocks if block.sxpb is not None]

    assert [block.operation for block in parsed] == ["<", ">", ">"]
    assert [block.filepath for block in parsed] == [
        "/dev/stdin",
        "/dev/stdout",
        "artifact.sxpb",
    ]
    assert parsed[0].sxpb == {"question": "move?"}
    assert parsed[1].sxpb == {"answer": "b2"}
    assert parsed[2].sxpb == {"title": "Report"}


def test_parse_code_blocks_sxpb_parsing_is_opt_in():
    """Default block parsing remains structural only."""
    text = """```sxpb > /dev/stdout
(answer "b2")
```"""
    assert parse_code_blocks(text)[0].sxpb is None
    assert parse_code_blocks(text, parse_sxpb=True)[0].sxpb == {"answer": "b2"}


def test_parse_code_blocks_records_sxpb_parse_errors():
    """Invalid SxPB blocks keep their text and expose the parse failure."""
    text = """```sxpb > bad.sxpb
not sxpb at all ) )
```"""
    block = parse_code_blocks(text, parse_sxpb=True)[0]
    assert block.sxpb is None
    assert block.sxpb_error is not None
    assert block.content == "not sxpb at all ) )"


def test_parse_code_blocks_does_not_parse_other_languages_as_sxpb():
    """A random Python block containing parens is not treated as SxPB."""
    text = """```python
(answer "not really")
```"""
    block = parse_code_blocks(text, parse_sxpb=True)[0]
    assert block.language == "python"
    assert block.sxpb is None
    assert block.sxpb_error is None


def test_get_sxpb_from_markdown_compatibility_wrapper():
    """Legacy helper returns only dict content from the first parseable block."""
    tagged = """\
```sxpb > /dev/stdout
(candidates
 (candidate (name "Test") (texture "Smooth"))
)
```
"""
    assert get_sxpb_from_markdown(tagged) == {
        "candidates": {"candidate": {"name": "Test", "texture": "Smooth"}}
    }

    untagged = """\
```
(answer "fallback")
```
"""
    assert get_sxpb_from_markdown(untagged) == {"answer": "fallback"}


def test_parse_sxpb_answer_uses_last_stdout_block_for_compatibility():
    """The old parse_sxpb_answer behavior used the last stdout match."""
    response = """\
```sxpb > /dev/stdout
(answer "first")
```

```sxpb > /dev/stdout
(answer "last")
```
"""
    assert parse_sxpb_answer(response) == "last"
