"""SSD-backed KV cache persistence for evicted prefix cache entries.

When the in-memory KVPrefixCache evicts an entry under memory pressure, this
store serializes KV tensors to SSD via safetensors.  On a cache miss in RAM,
entries can be reloaded from SSD — far faster than re-prefilling from scratch.

Inspired by oMLX's PagedSSDCacheManager, simplified to entry-level granularity
since exo uses bounded RotatingKVCache.

Environment:
    EXO_SSD_CACHE_ENABLED: "1" (default) | "0"
    EXO_SSD_CACHE_DIR:     default ~/.cache/exo/kv_cache
    EXO_SSD_CACHE_MAX_GB:  default 10
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, RotatingKVCache

from exo.shared.types.mlx import KVCacheType
from exo.worker.runner.bootstrap import logger

SSD_CACHE_ENABLED: Final[bool] = os.environ.get("EXO_SSD_CACHE_ENABLED", "1") == "1"
_CACHE_DIR: Final[str] = os.environ.get(
    "EXO_SSD_CACHE_DIR", os.path.expanduser("~/.cache/exo/kv_cache")
)
_MAX_GB: Final[float] = float(os.environ.get("EXO_SSD_CACHE_MAX_GB", "10"))
_META_VERSION: Final[int] = 1


@dataclass(frozen=True)
class _LayerMeta:
    layer_type: str  # "rotating_kv" | "arrays" | "skip"
    offset: int = 0
    idx: int = 0
    max_size: int = 0
    keep: int = 0
    num_states: int = 0


@dataclass
class _Entry:
    entry_id: str
    prompt_tokens: mx.array
    token_count: int
    size_bytes: int
    path: Path
    created_at: float
    layers: list[_LayerMeta] = field(default_factory=list)


def _lm_from_dict(d: dict[str, object]) -> _LayerMeta:
    def _int(key: str) -> int:
        v = d.get(key)
        return v if isinstance(v, int) else int(v) if isinstance(v, (float, str)) else 0

    return _LayerMeta(
        layer_type=str(d.get("layer_type", "skip")),
        offset=_int("offset"),
        idx=_int("idx"),
        max_size=_int("max_size"),
        keep=_int("keep"),
        num_states=_int("num_states"),
    )


class SSDKVStore:
    """Disk-backed LRU cache for evicted KV prefix cache entries."""

    def __init__(
        self,
        cache_dir: str | None = None,
        max_size_bytes: int | None = None,
    ):
        self._dir = Path(cache_dir or _CACHE_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_size_bytes or int(_MAX_GB * 1024**3)
        self._entries: list[_Entry] = []
        self._scan_existing()

    # ── public ──────────────────────────────────────────────────────────

    def save(self, prompt_tokens: mx.array, cache: KVCacheType) -> str | None:
        """Serialize a cache entry to SSD.  Returns entry_id or None."""
        entry_id = f"{int(time.time() * 1000):x}_{os.getpid():x}"
        entry_dir = self._dir / entry_id
        try:
            entry_dir.mkdir(parents=True)

            # Materialize lazy arrays before saving
            to_eval: list[mx.array] = [prompt_tokens]
            for c in cache:
                if isinstance(c, RotatingKVCache) and c.keys is not None and c.values is not None:
                    to_eval.extend([c.keys, c.values])
                elif isinstance(c, ArraysCache):
                    for s in c.state:
                        if isinstance(s, mx.array):
                            to_eval.append(s)
            mx.eval(*to_eval)

            # Prompt tokens
            mx.save(str(entry_dir / "prompt.npy"), prompt_tokens)

            # KV tensors
            kv: dict[str, mx.array] = {}
            layer_dicts: list[dict[str, object]] = []
            for i, c in enumerate(cache):
                if isinstance(c, RotatingKVCache):
                    if c.keys is not None and c.values is not None:
                        kv[f"l{i}k"] = c.keys
                        kv[f"l{i}v"] = c.values
                    layer_dicts.append(dict(
                        layer_type="rotating_kv", offset=c.offset,
                        idx=c._idx, max_size=c.max_size, keep=c.keep,
                    ))
                elif isinstance(c, ArraysCache):
                    for j, s in enumerate(c.state):
                        if isinstance(s, mx.array):
                            kv[f"l{i}s{j}"] = s
                    layer_dicts.append(dict(
                        layer_type="arrays", num_states=len(c.state),
                    ))
                else:
                    layer_dicts.append(dict(layer_type="skip"))

            if not kv:
                kv["_empty"] = mx.array([0])
            mx.save_safetensors(str(entry_dir / "kv.safetensors"), kv)  # pyright: ignore[reportUnknownMemberType]

            meta: dict[str, int | str | float | list[dict[str, object]]] = dict(
                version=_META_VERSION, entry_id=entry_id,
                token_count=int(prompt_tokens.shape[0]),
                num_layers=len(cache), created_at=time.time(),
                layers=layer_dicts,
            )
            (entry_dir / "meta.json").write_text(json.dumps(meta))

            size_bytes = sum(p.stat().st_size for p in entry_dir.iterdir())
            ts = time.time()
            self._entries.append(_Entry(
                entry_id=entry_id, prompt_tokens=prompt_tokens,
                token_count=int(prompt_tokens.shape[0]),
                size_bytes=size_bytes, path=entry_dir,
                created_at=ts,
                layers=[_lm_from_dict(d) for d in layer_dicts],
            ))
            logger.info(
                f"SSD cache: saved {entry_id} "
                f"({meta['token_count']} tok, {size_bytes / 1048576:.1f} MB)"
            )
            self._enforce_limit()
            return entry_id

        except Exception:
            logger.opt(exception=True).warning("SSD cache: save failed")
            shutil.rmtree(entry_dir, ignore_errors=True)
            return None

    def find_match(self, prompt_tokens: mx.array) -> tuple[int, int] | None:
        """Best prefix match → (index, match_length) or None."""
        from exo.worker.engines.mlx.cache import get_prefix_length

        best: tuple[int, int] | None = None
        for i, e in enumerate(self._entries):
            n = get_prefix_length(prompt_tokens, e.prompt_tokens)
            if n > 0 and (best is None or n > best[1]):
                best = (i, n)
        return best

    def load(self, index: int) -> KVCacheType | None:
        """Reconstruct KV cache from SSD entry."""
        if not 0 <= index < len(self._entries):
            return None
        entry = self._entries[index]
        try:
            loaded = mx.load(str(entry.path / "kv.safetensors"))
            raw: dict[str, mx.array] = loaded if isinstance(loaded, dict) else {}
            result: list[RotatingKVCache | ArraysCache] = []
            for i, lm in enumerate(entry.layers):
                if lm.layer_type == "rotating_kv":
                    rc = RotatingKVCache(max_size=lm.max_size, keep=lm.keep)
                    k_key, v_key = f"l{i}k", f"l{i}v"
                    if k_key in raw and v_key in raw:
                        rc.keys = raw[k_key]
                        rc.values = raw[v_key]
                        rc.offset = lm.offset
                        rc._idx = lm.idx
                    result.append(rc)
                elif lm.layer_type == "arrays":
                    ac = ArraysCache(size=lm.num_states)
                    ac.state = [raw.get(f"l{i}s{j}") for j in range(lm.num_states)]
                    result.append(ac)
                elif lm.layer_type == "skip":
                    return None  # can't reconstruct unknown layers
            logger.info(
                f"SSD cache: loaded {entry.entry_id} "
                f"({entry.token_count} tok, {entry.size_bytes / 1048576:.1f} MB)"
            )
            return result
        except Exception:
            logger.opt(exception=True).warning(
                f"SSD cache: load failed ({entry.entry_id})"
            )
            return None

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._entries):
            e = self._entries.pop(index)
            shutil.rmtree(e.path, ignore_errors=True)

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    # ── private ─────────────────────────────────────────────────────────

    def _scan_existing(self) -> None:
        """Load metadata + prompt tokens from existing entries on startup."""
        if not self._dir.exists():
            return
        for d in sorted(self._dir.iterdir()):
            if not d.is_dir():
                continue
            try:
                meta_p, prompt_p, kv_p = (
                    d / "meta.json", d / "prompt.npy", d / "kv.safetensors",
                )
                if not all(p.exists() for p in (meta_p, prompt_p, kv_p)):
                    raise FileNotFoundError
                meta_any = json.loads(meta_p.read_text())  # pyright: ignore[reportAny]
                meta = cast(dict[str, object], meta_any)
                if meta.get("version") != _META_VERSION:
                    raise ValueError("version mismatch")
                pt_loaded = mx.load(str(prompt_p))
                pt: mx.array = (
                    next(iter(pt_loaded.values()))
                    if isinstance(pt_loaded, dict)
                    else pt_loaded
                )
                entry_id_val = meta.get("entry_id")
                token_count_val = meta.get("token_count")
                created_at_val = meta.get("created_at")
                layers_val = meta.get("layers")
                layers_parsed: list[dict[str, object]] = []
                if isinstance(layers_val, list):
                    for lm_raw in layers_val:  # pyright: ignore[reportUnknownVariableType]
                        if isinstance(lm_raw, dict):
                            layers_parsed.append(cast(dict[str, object], lm_raw))
                self._entries.append(_Entry(
                    entry_id=str(entry_id_val) if entry_id_val is not None else d.name,
                    prompt_tokens=pt,
                    token_count=int(token_count_val) if isinstance(token_count_val, (int, float)) else 0,
                    size_bytes=sum(f.stat().st_size for f in d.iterdir()),
                    path=d,
                    created_at=float(created_at_val) if isinstance(created_at_val, (int, float)) else 0.0,
                    layers=[_lm_from_dict(lm_d) for lm_d in layers_parsed],
                ))
            except Exception:
                shutil.rmtree(d, ignore_errors=True)
        if self._entries:
            total_mb = sum(e.size_bytes for e in self._entries) / 1048576
            logger.info(
                f"SSD cache: {len(self._entries)} entries "
                f"({total_mb:.0f} MB) in {self._dir}"
            )

    def _enforce_limit(self) -> None:
        total = sum(e.size_bytes for e in self._entries)
        while total > self._max_bytes and self._entries:
            oldest = min(
                range(len(self._entries)),
                key=lambda i: self._entries[i].created_at,
            )
            total -= self._entries[oldest].size_bytes
            self.remove(oldest)
