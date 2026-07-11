from sxpb_llm.rag.chunk import (
    chunk_markdown,
    chunk_sxpb,
    chunk_yaml_list,
)


def test_markdown_splits_level_two_sections_and_large_paragraph_groups():
    text = "Intro\n\n## One\n\naaaa\n\nbbbb\n\n## Two\n\nend"
    chunks = chunk_markdown(text, "guide.md", max_chunk_size=17)
    assert [(chunk.section, chunk.content) for chunk in chunks] == [
        ("guide.md", "Intro"),
        ("## One", "## One\n\naaaa"),
        ("## One", "bbbb"),
        ("## Two", "## Two\n\nend"),
    ]


def test_yaml_list_preserves_top_level_items_and_nested_lines():
    text = '- "first"\n  - "nested"\n- "second"\n'
    chunks = chunk_yaml_list(text, "memory.yaml", max_chunk_size=20)
    assert [chunk.content for chunk in chunks] == [
        '- "first"\n  - "nested"',
        '- "second"',
    ]


def test_sxpb_uses_balanced_forms_and_ignores_parentheses_in_strings_comments():
    text = '(())\n\n(item (text "not ) structure"))  ; ignored ( comment\n\n(next ok)'
    chunks = chunk_sxpb(text, "data.sxpb", max_chunk_size=100)
    assert [chunk.content for chunk in chunks] == [
        "(())",
        '(item (text "not ) structure"))',
        "; ignored ( comment\n\n(next ok)",
    ]


def test_large_sxpb_nest_splits_at_child_form_boundaries():
    text = "(root (first aaaaaa) (second bbbbbb) (third cccccc))"
    chunks = chunk_sxpb(text, "data.sxpb", max_chunk_size=25)
    assert [chunk.section for chunk in chunks] == ["root", "root", "root"]
    assert [chunk.content for chunk in chunks] == [
        "(root (first aaaaaa)",
        "(second bbbbbb)",
        "(third cccccc)\n)",
    ]


def test_malformed_sxpb_falls_back_without_losing_text():
    text = "(broken\n\nsecond paragraph"
    chunks = chunk_sxpb(text, "bad.sxpb", max_chunk_size=50)
    assert "\n\n".join(chunk.content for chunk in chunks) == text
