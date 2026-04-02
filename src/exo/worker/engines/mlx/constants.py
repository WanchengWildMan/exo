import os


def _optional_int_env(name: str, default: int | None) -> int | None:
	value = os.getenv(name)
	if value is None:
		return default
	normalized = value.strip().lower()
	if normalized in ("", "none", "null"):
		return None
	return int(normalized)


def _int_env(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	return int(value.strip())


# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = _optional_int_env("EXO_KV_GROUP_SIZE", 32)
KV_BITS: int | None = _optional_int_env("EXO_KV_BITS", None)
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = _int_env("EXO_MAX_TOKENS", 32168)
MAX_KV_SIZE: int | None = _optional_int_env("EXO_MAX_KV_SIZE", 3200)
KEEP_KV_SIZE: int | None = _optional_int_env("EXO_KEEP_KV_SIZE", 1600)
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = _int_env("EXO_CACHE_GROUP_SIZE", 64)
KV_CACHE_BITS: int | None = _optional_int_env("EXO_KV_CACHE_BITS", None)

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
