from exo.shared.types.text_generation import resolve_max_output_tokens


def test_none_max_output_tokens_uses_default_without_extra_reserve() -> None:
    assert (
        resolve_max_output_tokens(
            max_output_tokens=None,
            enable_thinking=True,
            default_max_tokens=1024,
            thinking_content_token_reserve=512,
        )
        == 1024
    )


def test_non_thinking_requests_keep_requested_budget() -> None:
    assert (
        resolve_max_output_tokens(
            max_output_tokens=256,
            enable_thinking=False,
            default_max_tokens=1024,
            thinking_content_token_reserve=512,
        )
        == 256
    )


def test_thinking_requests_gain_extra_internal_budget() -> None:
    assert (
        resolve_max_output_tokens(
            max_output_tokens=256,
            enable_thinking=True,
            default_max_tokens=1024,
            thinking_content_token_reserve=512,
        )
        == 768
    )


def test_thinking_requests_respect_global_cap() -> None:
    assert (
        resolve_max_output_tokens(
            max_output_tokens=900,
            enable_thinking=True,
            default_max_tokens=1024,
            thinking_content_token_reserve=512,
        )
        == 1024
    )
