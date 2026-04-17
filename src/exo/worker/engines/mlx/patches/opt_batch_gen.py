import os
import time
from typing import Any, cast

import mlx.core as mx
from loguru import logger
from mlx_lm.generate import BatchGenerator, generation_stream

_PRECOMPUTE_TOP_K = 20

# ── Decode-phase memory management (aligned with oMLX) ──────────────────────
# How often (in decode steps) to force mx.synchronize() + mx.clear_cache().
# On 16GB devices running 9B models, intermediate tensors from decode steps
# accumulate rapidly.  512 is far too infrequent — 8 tokens can already OOM.
# oMLX does sync+clear much more aggressively.  Default: every 1 step.
_DECODE_SYNC_INTERVAL: int = int(os.getenv("EXO_DECODE_SYNC_INTERVAL", "1"))

# Decode memory safety valve: if Metal memory exceeds this limit (in GB) at
# the START of a decode step (before the forward pass), force-finish all
# entries with finish_reason="length" instead of running the forward pass
# and risking OOM.  Default 0 = auto (total_ram − 1.5 GB).
_DECODE_MAX_METAL_GB: float = float(os.getenv("EXO_DECODE_MAX_METAL_GB", "0"))

_decode_memory_limit_bytes: int | None = None


def _get_decode_memory_limit_bytes() -> int:
    """Return the decode-phase Metal memory limit in bytes (cached)."""
    global _decode_memory_limit_bytes
    if _decode_memory_limit_bytes is not None:
        return _decode_memory_limit_bytes
    if _DECODE_MAX_METAL_GB > 0:
        _decode_memory_limit_bytes = int(_DECODE_MAX_METAL_GB * 1024**3)
    else:
        import psutil
        total_gb = psutil.virtual_memory().total / (1024**3)
        _decode_memory_limit_bytes = int((total_gb - 1.5) * 1024**3)
    return _decode_memory_limit_bytes

_original_public_next = BatchGenerator.next

_pending_topk_idx: mx.array | None = None
_pending_topk_val: mx.array | None = None
_pending_selected_lps: mx.array | None = None


def _fast_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    tic = time.perf_counter()
    batch = self.active_batch
    assert batch is not None
    batch_size = len(batch)

    prev_tokens = batch.y
    prev_logprobs = batch.logprobs

    global _pending_topk_idx, _pending_topk_val, _pending_selected_lps

    # ── Memory safety valve (aligned with oMLX ProcessMemoryEnforcer) ────
    # Check BEFORE the forward pass.  If Metal memory already exceeds the
    # limit after the previous step's sync+clear, skip the forward pass and
    # force-finish all entries with finish_reason="length" to prevent OOM.
    active_bytes = mx.get_active_memory()
    limit_bytes = _get_decode_memory_limit_bytes()
    if active_bytes > limit_bytes:
        logger.warning(
            f"Decode memory safety valve: {active_bytes / 1e9:.2f}GB > "
            f"{limit_bytes / 1e9:.2f}GB limit. "
            f"Force-finishing {batch_size} entries to prevent OOM."
        )
        prev_token_list = cast(list[int], prev_tokens.tolist())
        responses: list[Any] = []
        for e in range(batch_size):
            cache = batch.extract_cache(e)
            responses.append(
                self.Response(
                    batch.uids[e], prev_token_list[e], prev_logprobs[e], "length", cache
                )
            )
        self.active_batch = None
        _pending_topk_idx = None
        _pending_topk_val = None
        _pending_selected_lps = None
        self._next_count += 1
        self._stats.generation_time += time.perf_counter() - tic
        self._stats.generation_tokens += len(responses)
        return responses

    has_processors = any(p for ps in batch.logits_processors for p in ps)
    if has_processors:
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate([toks, prev_tokens[i : i + 1]])

    logits = self.model(prev_tokens[:, None], cache=batch.cache)
    logits = logits[:, -1, :]

    if has_processors:
        processed_logits: list[mx.array] = []
        for e in range(batch_size):
            sample_logits: mx.array = logits[e : e + 1]
            for processor in batch.logits_processors[e]:
                sample_logits = processor(batch.tokens[e], sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if (
        batch_size == 1
        or any(batch.samplers)
        and all(s is batch.samplers[0] for s in batch.samplers)
    ):
        sampler = batch.samplers[0] or self.sampler
        batch.y = sampler(logprobs)
    elif any(batch.samplers):
        all_samples: list[mx.array] = []
        for e in range(batch_size):
            s = batch.samplers[e] or self.sampler
            all_samples.append(s(logprobs[e : e + 1]))
        batch.y = mx.concatenate(all_samples, axis=0)
    else:
        batch.y = self.sampler(logprobs)
    batch.logprobs = list(logprobs)

    emit_topk_indices: list[list[int]] = (
        cast(list[list[int]], _pending_topk_idx.tolist())
        if _pending_topk_idx is not None
        else []
    )
    emit_topk_values: list[list[float]] = (
        cast(list[list[float]], _pending_topk_val.tolist())
        if _pending_topk_val is not None
        else []
    )
    emit_selected_lps: list[float] = (
        cast(list[float], _pending_selected_lps.tolist())
        if _pending_selected_lps is not None
        else []
    )

    needs_topk: bool = getattr(self, "_needs_topk", False)
    if needs_topk:
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        _pending_topk_idx = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        _pending_topk_val = mx.take_along_axis(logprobs, _pending_topk_idx, axis=1)
        sort_order = mx.argsort(-_pending_topk_val, axis=1)
        _pending_topk_idx = mx.take_along_axis(_pending_topk_idx, sort_order, axis=1)
        _pending_topk_val = mx.take_along_axis(_pending_topk_val, sort_order, axis=1)
        _pending_selected_lps = logprobs[mx.arange(batch_size), batch.y]
        mx.async_eval(
            batch.y,
            *batch.logprobs,
            *batch.tokens,
            _pending_topk_idx,
            _pending_topk_val,
            _pending_selected_lps,
        )
    else:
        _pending_topk_idx = None
        _pending_topk_val = None
        _pending_selected_lps = None
        mx.async_eval(batch.y, *batch.logprobs, *batch.tokens)

    prev_token_list: list[int] = cast(list[int], prev_tokens.tolist())

    toc = time.perf_counter()
    self._stats.generation_time += toc - tic

    keep_idx: list[int] = []
    end_idx: list[int] = []
    responses: list[Any] = []
    stop_tokens = self.stop_tokens

    for e in range(batch_size):
        t = prev_token_list[e]
        uid = batch.uids[e]
        num_tok = batch.num_tokens[e] + 1
        batch.num_tokens[e] = num_tok

        if t in stop_tokens:
            finish_reason = "stop"
            end_idx.append(e)
        elif num_tok >= batch.max_tokens[e]:
            finish_reason = "length"
            end_idx.append(e)
        else:
            finish_reason = None
            keep_idx.append(e)

        cache = None
        if finish_reason is not None:
            cache = batch.extract_cache(e)
        response = self.Response(uid, t, prev_logprobs[e], finish_reason, cache)
        if emit_topk_indices and e < len(emit_topk_indices):
            response._topk_indices = emit_topk_indices[e]  # pyright: ignore[reportAttributeAccessIssue]
            response._topk_values = emit_topk_values[e]  # pyright: ignore[reportAttributeAccessIssue]
            response._selected_logprob = emit_selected_lps[e]  # pyright: ignore[reportAttributeAccessIssue]
        responses.append(response)

    if end_idx:
        if keep_idx:
            batch.filter(keep_idx)
            if (
                _pending_topk_idx is not None
                and _pending_topk_val is not None
                and _pending_selected_lps is not None
            ):
                ki = mx.array(keep_idx)
                _pending_topk_idx = _pending_topk_idx[ki]
                _pending_topk_val = _pending_topk_val[ki]
                _pending_selected_lps = _pending_selected_lps[ki]
        else:
            self.active_batch = None
            _pending_topk_idx = None
            _pending_topk_val = None
            _pending_selected_lps = None

    self._next_count += 1
    if _DECODE_SYNC_INTERVAL > 0 and self._next_count % _DECODE_SYNC_INTERVAL == 0:
        mx.synchronize()
        mx.clear_cache()
    self._stats.generation_tokens += len(responses)
    return responses


def _patched_public_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    batch = self.active_batch
    # Only do decode with fast_next
    if batch is not None and not self.unprocessed_prompts:
        with mx.stream(generation_stream):
            return _fast_next(self)
    return _original_public_next(self)


def apply_batch_gen_patch() -> None:
    BatchGenerator.next = _patched_public_next
