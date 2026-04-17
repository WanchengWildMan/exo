from exo.api.adapters.chat_completions import openai_compatible_content


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
