from exo.api.adapters.chat_completions import (
    openai_compatible_content,
    trim_system_prompt,
)


def test_openai_compatible_content_prefers_reasoning_when_content_empty() -> None:
    assert openai_compatible_content("", "完整回答") == "完整回答"


def test_openai_compatible_content_prefers_reasoning_for_sentinel_content() -> None:
    assert openai_compatible_content("*", "完整回答") == "完整回答"


def test_openai_compatible_content_prefers_reasoning_for_quote_placeholder() -> None:
    assert openai_compatible_content(' "', "完整回答") == "完整回答"


def test_openai_compatible_content_prefers_reasoning_for_multi_symbol_placeholder() -> None:
    assert openai_compatible_content("...", "完整回答") == "完整回答"


def test_openai_compatible_content_keeps_meaningful_content() -> None:
    assert openai_compatible_content("OK", "完整回答") == "OK"


def test_trim_system_prompt_applies_openclaw_defaults() -> None:
    long_skill_desc = "A" * 120
    prompt = (
        "You are a personal assistant running inside OpenClaw.\n"
        "## OpenClaw CLI Quick Reference\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>demo</name>\n"
        f"    <description>{long_skill_desc}</description>\n"
        "    <location>/tmp/demo/SKILL.md</location>\n"
        "  </skill>\n"
        "</available_skills>\n"
        + ("B" * 3200)
    )

    trimmed, trim_policy = trim_system_prompt(prompt)

    assert trim_policy == "openclaw-default"
    assert trimmed.endswith(" [trimmed]")
    assert len(trimmed) == 3010
    assert long_skill_desc not in trimmed


def test_trim_system_prompt_keeps_non_openclaw_prompt_untouched() -> None:
    prompt = "You are a helpful assistant.\n" + ("C" * 3200)

    trimmed, trim_policy = trim_system_prompt(prompt)

    assert trim_policy == "explicit-env"
    assert trimmed == prompt


def test_trim_system_prompt_compacts_each_skill_entry() -> None:
    prompt = """You are a personal assistant running inside OpenClaw.
<available_skills>
  <skill>
    <name>demo-one</name>
    <description>First line of a very long description that should be compacted.
Second line should also disappear after compaction.</description>
    <location>/tmp/demo-one/SKILL.md</location>
  </skill>
  <skill>
    <name>demo-two</name>
    <description>Short description</description>
    <location>/tmp/demo-two/SKILL.md</location>
  </skill>
</available_skills>
"""

    trimmed, trim_policy = trim_system_prompt(prompt)

    assert trim_policy == "openclaw-default"
    assert "Second line should also disappear" not in trimmed
    assert "<name>demo-one</name>" in trimmed
    assert "<name>demo-two</name>" in trimmed
    assert "<location>/tmp/demo-one/SKILL.md</location>" in trimmed
    assert "<location>/tmp/demo-two/SKILL.md</location>" in trimmed
    assert (
        "<description>First line of a very long desc…</description>" in trimmed
    )
    assert "<description>Short description</description>" in trimmed


def test_trim_system_prompt_applies_to_developer_role_content() -> None:
    """trim_system_prompt works identically regardless of whether the caller
    originally received 'system' or 'developer' role – the function itself is
    role-agnostic; this test merely confirms the detection logic fires for
    content that uses OpenClaw markers."""
    long_skill_desc = "A" * 120
    prompt = (
        "You are a personal assistant running inside OpenClaw.\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>test-skill</name>\n"
        f"    <description>{long_skill_desc}</description>\n"
        "    <location>/tmp/test/SKILL.md</location>\n"
        "  </skill>\n"
        "</available_skills>\n"
        + ("X" * 3200)
    )

    trimmed, trim_policy = trim_system_prompt(prompt)

    assert trim_policy == "openclaw-default"
    assert trimmed.endswith(" [trimmed]")
    assert long_skill_desc not in trimmed
