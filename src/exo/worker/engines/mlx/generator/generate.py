import contextlib
import functools
import math
import os
import time
from copy import deepcopy
from typing import Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models.cache import ArraysCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
    resolve_max_output_tokens,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    flush_prefill_sends,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KEEP_KV_SIZE,
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_KV_SIZE,
    MAX_TOKENS,
    THINKING_CONTENT_TOKEN_RESERVE,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_barrier,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    get_inner_model,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5


@contextlib.contextmanager
def patch_embed_tokens(
    model: Model, embeddings: mx.array, start_offset: int = 0, token_count: int = 0
) -> Generator[None]:
    inner = get_inner_model(model)  # type: ignore
    original_embed = inner.embed_tokens  # type: ignore
    end_offset = start_offset + token_count
    offset = [start_offset]

    def _inject(input_ids: mx.array) -> mx.array:
        start = offset[0]
        if start >= end_offset:
            return original_embed(input_ids)  # type: ignore
        chunk_len = input_ids.shape[-1]
        end = min(start + chunk_len, end_offset)
        offset[0] = end
        if end - start < chunk_len:
            return original_embed(input_ids)  # type: ignore
        return embeddings[:, start:end, :]

    for attr in dir(original_embed):  # type: ignore
        if not attr.startswith("_") and not hasattr(_inject, attr):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(_inject, attr, getattr(original_embed, attr))  # type: ignore

    inner.embed_tokens = _inject
    try:
        yield
    finally:
        inner.embed_tokens = original_embed


class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


class PrefillOOM(BaseException):
    """Raised when memory usage exceeds threshold during prefill."""


_PREFILL_MEMORY_CHECK_INTERVAL: int = int(os.getenv("EXO_PREFILL_MEMORY_CHECK_INTERVAL", "8"))
# Maximum Metal GPU memory (in GB) above which prefill is aborted.
# Uses mx.get_active_memory() which tracks actual Metal allocations — more accurate
# than psutil.virtual_memory() on Apple Silicon (which includes OS caches, etc.).
# Default: total system RAM minus 1.5GB reserve (auto-calculated at first check).
_PREFILL_MAX_METAL_GB: float = float(os.getenv("EXO_PREFILL_MAX_METAL_GB", "0"))
# Fallback: minimum available system RAM (in GB) below which prefill is aborted.
# Only used when mx.get_active_memory() is unavailable.
_PREFILL_MIN_AVAILABLE_GB: float = float(os.getenv("EXO_PREFILL_MIN_AVAILABLE_GB", "0.5"))
# Interval (in steps) between forced mx.eval() calls during prefill to flush the
# MLX computation graph and free intermediate activation memory.  Without this the
# graph accumulates over all chunks and causes OOM long before the KV cache itself
# becomes a problem. Default: every 8 steps.
_PREFILL_EVAL_INTERVAL: int = int(os.getenv("EXO_PREFILL_EVAL_INTERVAL", "1"))


def _get_metal_memory_limit_gb() -> float:
    """Get Metal memory limit in GB (auto-calculated if not set)."""
    if _PREFILL_MAX_METAL_GB > 0:
        return _PREFILL_MAX_METAL_GB
    # Auto: total system RAM minus 1.5GB reserve for OS + other apps
    import psutil
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    return total_gb - 1.5


def _check_memory_during_prefill(step: int, processed: int, total: int) -> None:
    """Log memory usage during prefill and abort if Metal memory exceeds limit.

    Uses mx.get_active_memory() which tracks actual Metal GPU allocations on
    Apple Silicon — more accurate than psutil.virtual_memory() which includes
    OS file caches and other non-Metal memory. (Learned from oMLX #429.)
    """
    if step % _PREFILL_MEMORY_CHECK_INTERVAL != 0:
        return
    active_bytes = mx.get_active_memory()
    active_gb = active_bytes / (1024 ** 3)
    limit_gb = _get_metal_memory_limit_gb()
    logger.info(
        f"[Prefill memory] step={step} tokens={processed}/{total} "
        f"metal_active={active_gb:.2f}GB metal_limit={limit_gb:.2f}GB"
    )
    if active_gb > limit_gb:
        logger.error(
            f"Metal active memory {active_gb:.2f}GB exceeds limit {limit_gb:.2f}GB "
            f"during prefill at token {processed}/{total}. Aborting to prevent OOM."
        )
        raise PrefillOOM(
            f"Metal memory {active_gb:.2f}GB exceeds {limit_gb:.2f}GB "
            f"at prefill token {processed}/{total}"
        )


def _maybe_eval_cache(cache: KVCacheType, step: int) -> None:
    """Periodically force-evaluate the KV cache to flush the MLX computation graph.

    Without this, MLX accumulates the entire prefill as a lazy graph (~56 MB of
    attention activations per step × N steps) which dwarfs the actual KV cache
    memory and causes OOM.

    After evaluating, clears MLX's internal memory cache to free transient
    intermediate tensors (attention matrices, MLP activations, etc.) that became
    "dead" after eval.  Without this, dead tensors from previous chunks accumulate
    on Metal and consume memory that's no longer needed.  (Key insight from oMLX:
    _sync_and_clear_cache() after each prefill chunk.)

    Called every chunk (eval_interval=1 enforced) to aggressively reclaim memory.
    On 16GB devices, even 2 chunks of accumulated transients (~100-400 MB) can
    push Metal over the OOM threshold.
    """
    if step > 0 and step % _PREFILL_EVAL_INTERVAL == 0:
        mx.eval([c.state for c in cache])
        # Free dead intermediate tensors from previous chunks.
        # synchronize() is required before clear_cache() to prevent
        # M4 kernel panics (oMLX #300, #435).
        mx.synchronize()
        before = mx.get_active_memory()
        mx.clear_cache()
        after = mx.get_active_memory()
        freed_mb = (before - after) / 1048576
        if freed_mb > 1:
            logger.debug(f"[Prefill clear] step={step} freed={freed_mb:.0f}MB")


class PrefillSyncTimeout(BaseException):
    """Raised when a distributed sync during prefill exceeds the timeout."""


_PREFILL_SYNC_TIMEOUT: float = float(os.getenv("EXO_PREFILL_SYNC_TIMEOUT", "60"))


def _run_prefill_sync(
    callback: Callable[[], None] | None,
    *,
    rank: int,
    phase: str,
) -> None:
    if callback is None:
        return

    logger.info(f"[R{rank}] Waiting for distributed prefill sync: {phase}")
    started_at = time.perf_counter()

    if _PREFILL_SYNC_TIMEOUT <= 0:
        callback()
    else:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(callback)
            try:
                future.result(timeout=_PREFILL_SYNC_TIMEOUT)
            except FuturesTimeoutError:
                elapsed_s = time.perf_counter() - started_at
                msg = (
                    f"[R{rank}] Distributed prefill sync timed out after {elapsed_s:.1f}s "
                    f"during {phase}. The other rank likely crashed or lost connection."
                )
                logger.error(msg)
                raise PrefillSyncTimeout(msg) from None

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        f"[R{rank}] Distributed prefill sync complete: {phase} ({elapsed_ms:.1f}ms)"
    )


def _has_pipeline_communication_layer(model: Model):
    for layer in model.layers:
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            return True
    return False


def pipeline_parallel_prefill(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
    distributed_prompt_progress_callback: Callable[[], None] | None,
    group: mx.distributed.Group,
) -> None:
    """Prefill the KV cache for pipeline parallel with overlapping stages.

    Each rank processes the full prompt through its real cache, offset by leading
    and trailing dummy iterations.

    Total iterations per rank = N_real_chunks + world_size - 1:
      - rank r leading dummies  (skip_pipeline_io, throwaway cache)
      - N_real_chunks real      (pipeline IO active, real cache)
      - (world_size-1-r) trailing dummies (skip_pipeline_io, throwaway cache)

    e.g.
    Timeline (2 ranks, 3 chunks of 10240 tokens @ step=4096):
        iter 0: R0 real[0:4096]     R1 dummy
        iter 1: R0 real[4096:8192]  R1 real[0:4096]
        iter 2: R0 real[8192:10240] R1 real[4096:8192]
        iter 3: R0 dummy            R1 real[8192:10240]

    This function is designed to match mlx_lm's stream_generate exactly in terms of
    side effects (given the same prefill step size)
    """
    prefill_step_size = prefill_step_size // min(4, group.size())

    quantize_cache_fn: Callable[..., None] = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    _prompt_cache: KVCacheType = prompt_cache
    rank = group.rank()
    world_size = group.size()

    # Build list of real prompt chunk sizes
    total = len(prompt)
    real_chunk_sizes: list[int] = []
    remaining = total - 1
    while remaining:
        n = min(prefill_step_size, remaining)
        real_chunk_sizes.append(n)
        remaining -= n
    n_real = len(real_chunk_sizes)

    # Each rank does: [rank leading dummies] [N real chunks] [world_size-1-rank trailing dummies]
    n_leading = rank
    n_trailing = world_size - 1 - rank
    n_total = n_leading + n_real + n_trailing

    t_start = time.perf_counter()
    processed = 0
    logger.info(
        f"[R{rank}] Pipeline prefill: {n_real} real + {n_leading} leading + {n_trailing} trailing = {n_total} iterations"
    )
    clear_prefill_sends()

    # Initial callback matching generate_step
    prompt_progress_callback(0, total)

    try:
        with mx.stream(generation_stream):
            for leading_index in range(n_leading):
                _run_prefill_sync(
                    distributed_prompt_progress_callback,
                    rank=rank,
                    phase=f"leading_dummy {leading_index + 1}/{n_leading}",
                )

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                model(
                    prompt[processed : processed + chunk_size][None],
                    cache=_prompt_cache,
                )
                quantize_cache_fn(_prompt_cache)
                processed += chunk_size

                _run_prefill_sync(
                    distributed_prompt_progress_callback,
                    rank=rank,
                    phase=f"real_chunk {i + 1}/{n_real}",
                )

                flush_prefill_sends()

                # Periodically force-evaluate the KV cache to flush the lazy
                # computation graph.  Without this, MLX accumulates all prefill
                # steps as a single graph whose intermediate activations dwarf
                # the actual KV cache and cause OOM on memory-constrained nodes.
                _maybe_eval_cache(_prompt_cache, i)

                prompt_progress_callback(processed, total)
                _check_memory_during_prefill(i, processed, total)

            for trailing_index in range(n_trailing):
                _run_prefill_sync(
                    distributed_prompt_progress_callback,
                    rank=rank,
                    phase=f"trailing_dummy {trailing_index + 1}/{n_trailing}",
                )

    finally:
        clear_prefill_sends()

    # Post-loop: process remaining 1 token + add +1 entry to match stream_generate.
    for _ in range(2):
        with mx.stream(generation_stream):
            model(prompt[-1:][None], cache=_prompt_cache)
            quantize_cache_fn(_prompt_cache)
        flush_prefill_sends()

    assert _prompt_cache is not None
    with mx.stream(generation_stream):
        mx.eval([c.state for c in _prompt_cache])  # type: ignore

    # Final callback matching generate_step
    prompt_progress_callback(total, total)

    logger.info(
        f"[R{rank}] Prefill: {n_real} real + {n_leading}+{n_trailing} dummy iterations, "
        f"Processed {processed} tokens in {(time.perf_counter() - t_start) * 1000:.1f}ms"
    )


def _estimate_prefill_peak_bytes(model: Model, num_tokens: int, prefill_step_size: int) -> int:
    """Estimate peak memory for a single prefill chunk.

    Considers SDPA implementation details (learned from oMLX MemoryMonitor):
    - head_dim > 128: MLX SDPA fallback materializes full attention matrix
      [B, num_heads, chunk, kv_len] in float32 — very large peak.
    - head_dim <= 128: MLX fused SDPA kernel uses tiled computation with
      much lower peak memory (only output buffer + small working set).

    Returns estimated peak bytes, or 0 if model config is unavailable.
    """
    config: object | None = getattr(model, "args", None) or getattr(model, "config", None)
    if config is None:
        return 0
    num_layers: int = int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0) or 0)
    num_kv_heads: int = int(getattr(config, "num_key_value_heads", 0) or getattr(config, "num_attention_heads", 0) or 0)
    num_heads: int = int(getattr(config, "num_attention_heads", 0) or num_kv_heads or 0)
    head_dim: int = int(getattr(config, "head_dim", 0) or 0)
    hidden_size: int = int(getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0) or 0)
    if head_dim == 0 and hidden_size > 0 and num_heads > 0:
        head_dim = hidden_size // num_heads
    if num_layers == 0 or num_kv_heads == 0 or head_dim == 0:
        return 0

    chunk_size: int = min(prefill_step_size, num_tokens)
    # KV cache growth for one chunk: 2 (K+V) × layers × kv_heads × head_dim × dtype(2) × chunk_tokens
    kv_per_chunk: int = 2 * num_layers * num_kv_heads * head_dim * 2 * chunk_size

    # SDPA peak depends on head_dim
    sdpa_peak: int
    if head_dim > 128:
        # Fallback: full attention matrix [B=1, num_heads, chunk, total_kv_len] in float32
        # total_kv_len ≈ num_tokens (worst case: last chunk sees all previous tokens)
        sdpa_peak = 1 * num_heads * chunk_size * num_tokens * 4  # float32
    else:
        # Fused kernel: output buffer [B=1, num_heads, chunk, head_dim] in dtype(2) + small overhead
        sdpa_peak = 1 * num_heads * chunk_size * head_dim * 2 * 2  # 2x for working set

    return kv_per_chunk + sdpa_peak


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        (tokens_per_sec, num_tokens, snapshots)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    # Pre-flight memory check: estimate peak memory and reject if it would exceed limit.
    prefill_step_size = int(os.getenv("EXO_PREFILL_STEP_SIZE", 64))
    peak_bytes = _estimate_prefill_peak_bytes(model, num_tokens, prefill_step_size)
    if peak_bytes > 0:
        current_bytes = mx.get_active_memory()
        limit_gb = _get_metal_memory_limit_gb()
        limit_bytes = int(limit_gb * 1024 ** 3)
        projected = current_bytes + peak_bytes
        if projected > limit_bytes:
            projected_gb = projected / (1024 ** 3)
            current_gb = current_bytes / (1024 ** 3)
            peak_gb = peak_bytes / (1024 ** 3)
            logger.error(
                f"Prefill pre-flight check failed: projected peak {projected_gb:.2f}GB "
                f"(current {current_gb:.2f}GB + estimate {peak_gb:.2f}GB) "
                f"exceeds limit {limit_gb:.2f}GB for {num_tokens} tokens"
            )
            raise PrefillOOM(
                f"Prefill would peak at {projected_gb:.2f}GB (limit {limit_gb:.2f}GB) "
                f"for {num_tokens} tokens"
            )
        else:
            logger.info(
                f"Prefill pre-flight: {num_tokens} tokens, estimated peak "
                f"{projected / (1024**3):.2f}GB / {limit_gb:.2f}GB OK"
            )

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []

    _prefill_step_counter = [0]

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.info(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        _check_memory_during_prefill(_prefill_step_counter[0], processed, total)
        # Clear dead intermediate tensors between prefill chunks to prevent
        # accumulation of transient activations (attention matrices, MLP outputs).
        # This is the key difference that allows oMLX to handle large prefills
        # on memory-constrained devices.  (oMLX scheduler._sync_and_clear_cache)
        if _prefill_step_counter[0] > 0 and _prefill_step_counter[0] % _PREFILL_EVAL_INTERVAL == 0:
            mx.eval([c.state for c in cache])
            mx.synchronize()
            mx.clear_cache()
        _prefill_step_counter[0] += 1
        if has_ssm:
            snapshots.append(snapshot_ssm_states(cache))

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    def combined_progress_callback(processed: int, total: int) -> None:
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    set_pipeline_prefill(model, is_prefill=True)

    mx_barrier(group)
    logger.info("Starting prefill")

    is_pipeline = _has_pipeline_communication_layer(model)

    prefill_step_size = int(os.getenv("EXO_PREFILL_STEP_SIZE", 64))
    logger.info(f"Prefill config: is_pipeline={is_pipeline}, num_tokens={num_tokens}, prefill_step_size={prefill_step_size}")

    try:
        if is_pipeline:
            logger.info("Using pipeline_parallel_prefill logic...")
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            pipeline_parallel_prefill(
                model=model,
                prompt=prompt_tokens,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                prompt_progress_callback=progress_callback,
                distributed_prompt_progress_callback=distributed_prompt_progress_callback,
                group=group,
            )
        else:
            logger.info("Using stream_generate logic for prefill...")
            # Use max_tokens=1 because max_tokens=0 does not work.
            # We just throw away the generated token - we only care about filling the cache
            for _ in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=1,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                prompt_progress_callback=combined_progress_callback,
            ):
                break  # Stop after first iteration - cache is now filled
    except PrefillCancelled:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise
    except PrefillOOM:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise
    except PrefillSyncTimeout:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise

    set_pipeline_queue_sends(model, queue_sends=False)
    set_pipeline_prefill(model, is_prefill=False)

    # stream_generate added 1 extra generated token to the cache, so we should trim it.
    # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
    pre_gen = deepcopy(snapshots[-2]) if has_ssm else None
    for i, c in enumerate(cache):
        if has_ssm and isinstance(c, (ArraysCache, RotatingKVCache)):
            assert pre_gen is not None
            if pre_gen.states[i] is not None:
                cache[i] = deepcopy(pre_gen.states[i])  # type: ignore
        else:
            assert not isinstance(c, (ArraysCache, RotatingKVCache))
            c.trim(2)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Exclude the last snapshot
    return tokens_per_sec, num_tokens, snapshots[:-1] if snapshots else []


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
) -> int:
    logger.info(f"warming up inference for instance: {model_id}")

    if os.environ.get("EXO_SKIP_WARMUP", "1") != "0":
        logger.info(
            "Skipping warmup inference; set EXO_SKIP_WARMUP=0 to re-enable it"
        )
        return 50

    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_task_params = TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        max_output_tokens=50,
        temperature=0.0,
    )

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        task_params=warmup_task_params,
    )

    tokens_generated = 0

    mx_barrier(group)

    logger.info("Generating warmup tokens")

    t = time.monotonic()

    for _r in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=warmup_task_params,
        prompt=warmup_prompt,
        kv_prefix_cache=None,
        group=group,
    ):
        tokens_generated += 1

    check_for_cancel_every = min(
        math.ceil(tokens_generated / min(time.monotonic() - t, 0.001)), 100
    )

    mx_barrier(group)

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    if group is not None:
        check_for_cancel_every = int(
            mx.max(
                mx.distributed.all_gather(
                    mx.array([check_for_cancel_every]),
                    group=group,
                )
            ).item()
        )

    logger.info(
        f"runner checking for cancellation every {check_for_cancel_every} tokens"
    )

    return check_for_cancel_every


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def extract_top_logprobs(
    logprobs: mx.array,
    tokenizer: TokenizerWrapper,
    top_logprobs: int,
    selected_token: int,
    precomputed_indices: list[int] | None = None,
    precomputed_values: list[float] | None = None,
    precomputed_selected: float | None = None,
) -> tuple[float, list[TopLogprobItem]]:
    if (
        precomputed_indices is not None
        and precomputed_values is not None
        and precomputed_selected is not None
    ):
        top_indices_list: list[int] = precomputed_indices[:top_logprobs]
        top_values_list: list[float] = precomputed_values[:top_logprobs]
        selected_logprob = precomputed_selected
    else:
        selected_logprob_arr = logprobs[selected_token]
        top_logprobs = min(top_logprobs, logprobs.shape[0] - 1)
        top_indices = mx.argpartition(-logprobs, top_logprobs)[:top_logprobs]
        top_values = logprobs[top_indices]
        sort_order = mx.argsort(-top_values)
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]
        mx.eval(selected_logprob_arr, top_indices, top_values)
        selected_logprob = float(selected_logprob_arr.item())
        top_indices_list = top_indices.tolist()  # type: ignore
        top_values_list = top_values.tolist()  # type: ignore

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for token_id, token_logprob in zip(top_indices_list, top_values_list, strict=True):
        if math.isnan(token_logprob):
            continue

        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=list(token_str.encode("utf-8")),
            )
        )

    return selected_logprob, top_logprob_items


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
    vision_processor: VisionProcessor | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)
    min_prefix_hit_length = max(1000, system_prompt_token_count(task, tokenizer))

    vision: VisionResult | None = None
    if vision_processor is not None:
        try:
            vision = prepare_vision(
                images=task.images,
                chat_template_messages=task.chat_template_messages,
                vision_processor=vision_processor,
                tokenizer=tokenizer,
                model=model,
                model_id=task.model,
                task_params=task,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Vision processing failed, falling back to text-only"
            )
    if vision is not None:
        all_prompt_tokens = vision.prompt_tokens
    media_regions: list[MediaRegion] = vision.media_regions if vision else []

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    if kv_prefix_cache is None:
        caches = make_kv_cache(model=model, max_kv_size=MAX_KV_SIZE, keep=KEEP_KV_SIZE)
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, all_prompt_tokens, media_regions=media_regions
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
        make_logits_processors(
            repetition_penalty=task.repetition_penalty,
            repetition_context_size=task.repetition_context_size,
        )
    )
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)] + logits_processors

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        min_p=task.min_p if task.min_p is not None else 0.05,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    maybe_vision_ctx = (
        patch_embed_tokens(
            model, vision.embeddings, prefix_hit_length, len(prompt_tokens) - 1
        )
        if vision is not None
        else contextlib.nullcontext()
    )
    with maybe_vision_ctx:
        prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
            model,
            tokenizer,
            sampler,
            prompt_tokens[:-1],
            caches,
            group,
            on_prefill_progress,
            distributed_prompt_progress_callback,
        )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    # stream_generate starts from the last token
    last_token = prompt_tokens[-2:]

    max_tokens = resolve_max_output_tokens(
        max_output_tokens=task.max_output_tokens,
        enable_thinking=task.enable_thinking,
        default_max_tokens=MAX_TOKENS,
        thinking_content_token_reserve=THINKING_CONTENT_TOKEN_RESERVE,
    )
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    in_thinking = False
    reasoning_tokens = 0
    think_start = tokenizer.think_start
    think_end = tokenizer.think_end

    logger.info("Starting decode")
    mx_barrier(group)

    for completion_tokens, out in enumerate(
        stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=last_token,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=caches,
            prefill_step_size=1,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
        ),
        start=1,
    ):
        generated_text_parts.append(out.text)
        accumulated_text += out.text

        if think_start is not None and out.text == think_start:
            in_thinking = True
        elif think_end is not None and out.text == think_end:
            in_thinking = False
        if in_thinking:
            reasoning_tokens += 1

        # Check for stop sequences
        text = out.text
        finish_reason: FinishReason | None = cast(
            FinishReason | None, out.finish_reason
        )
        stop_matched = False

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in accumulated_text:
                    # Trim text to just before the stop sequence
                    stop_index = accumulated_text.find(stop_seq)
                    text_before_stop = accumulated_text[:stop_index]
                    chunk_start = len(accumulated_text) - len(out.text)
                    text = text_before_stop[chunk_start:]
                    finish_reason = "stop"
                    stop_matched = True
                    break

        is_done = finish_reason is not None

        stats: GenerationStats | None = None
        if is_done:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )
            if not stop_matched and out.finish_reason not in get_args(FinishReason):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

            total_prompt_tokens = len(all_prompt_tokens)
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
            )

        # Extract logprobs from the full vocabulary logprobs array
        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if task.logprobs:
            with mx.stream(generation_stream):
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=out.logprobs,
                    tokenizer=tokenizer,
                    top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=out.token,
                )

        if is_done:
            # Log generation stats
            generation_elapsed = time.perf_counter() - generation_start_time
            generated_tokens = len(generated_text_parts)
            generation_tps = (
                generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )
            logger.debug(
                f"Generation complete: prefill {prompt_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )
            if kv_prefix_cache is not None:
                generated_tokens_array = mx.array(
                    tokenizer.encode(
                        "".join(generated_text_parts), add_special_tokens=False
                    )
                )
                full_prompt_tokens = mx.concatenate(
                    [all_prompt_tokens, generated_tokens_array]
                )
                hit_ratio = (
                    prefix_hit_length / len(all_prompt_tokens)
                    if len(all_prompt_tokens) > 0
                    else 0.0
                )
                if matched_index is not None and (
                    prefix_hit_length >= min_prefix_hit_length
                    and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
                ):
                    kv_prefix_cache.update_kv_cache(
                        matched_index,
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        restore_pos=prefix_hit_length,
                        media_regions=media_regions,
                    )
                else:
                    kv_prefix_cache.add_kv_cache(
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        media_regions=media_regions,
                    )

        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(
            text=text,
            token=out.token,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
        )

        if is_done:
            mx_barrier(group)
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
