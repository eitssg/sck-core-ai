"""Lightweight in-process idempotent cache for sck-core-ai.

This mirrors (in spirit) the idempotency approach used in `sck-core-api` but keeps
implementation self-contained and optional. It intentionally avoids external
dependencies for ease of local development. Not multi-process safe (per-container only).
"""

from typing import Callable

import os
import time
import hashlib
import json

import core_logging as log

from core_helper import store

from core_ai.indexing.action_indexer import dataclass


def _now() -> float:
    return time.time()


def _ttl() -> int:
    try:
        return int(os.getenv("CORE_AI_INTERNAL_IDEMPOTENCY_TTL", "600"))  # 10m default
    except ValueError:
        return 600


def _enabled() -> bool:
    return os.getenv("CORE_AI_INTERNAL_IDEMPOTENCY_ENABLED", "true").lower() == "true"


@dataclass
class IdempotentMeta:
    key: str
    hit: bool
    created_at: int | None
    duration_ms: int | None
    hits: int
    expires_at: float
    result: dict


def build_key(
    operation: str,
    payload: dict,
    scope: str | None = None,
    explicit_key: str | None = None,
) -> str:
    """Deterministic cache key.

    Args:
        operation: Logical operation (e.g., templates.generate)
        payload: Request payload dict (order-insensitive)
        scope: Optional scoping string (tenant/client composite)
        explicit_key: Overrides computed hash when provided
    """
    if explicit_key:
        base = explicit_key
    else:
        try:
            canonical = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))
        except Exception:
            canonical = str(payload)
        h = hashlib.sha256()
        h.update(operation.encode())
        h.update(b"\0")
        h.update(canonical.encode())
        base = h.hexdigest()
    scope_part = scope or "global"
    return f"ai-local-idem:{scope_part}:{operation}:{base}"


def run_idempotent(
    operation: str,
    payload: dict,
    producer: Callable[[], dict],
    scope: str | None = None,
    explicit_key: str | None = None,
) -> tuple[dict, dict]:
    """Execute under idempotency returning (result, meta).

    Meta fields:
        key: cache key
        hit: bool
        created_at: epoch seconds first computation
        duration_ms: original computation duration (for hits, preserved)
        hits: number of times value served (including this call)
    """
    key = build_key(operation, payload, scope=scope, explicit_key=explicit_key)
    if not _enabled():
        result = producer()
        return result, {"key": key, "hit": False, "duration_ms": None, "hits": 0}

    envelope: IdempotentMeta | None = store.retrieve(key)

    if envelope and envelope.expires_at > _now():
        envelope.hits += 1
        return envelope.result, envelope.__dict__

    start = _now()
    result = producer()
    duration_ms = int((_now() - start) * 1000)
    record = IdempotentMeta(
        key=key,
        hit=False,
        created_at=int(start),
        duration_ms=duration_ms,
        hits=1,
        expires_at=_now() + _ttl(),
        result=result,
    )

    store.store(key, record, ttl=_ttl())

    log.debug("AI local idempotent store", key=key, duration_ms=duration_ms)
    return result, record.__dict__


def stats() -> dict:
    return {"entries": store.size()}
