"""Tests for core_ai.cache idempotent run helper."""

import time

from core_ai import cache


def test_run_idempotent_hit_and_expiry(monkeypatch):
    # Force short TTL
    monkeypatch.setenv("CORE_AI_INTERNAL_IDEMPOTENCY_TTL", "1")
    monkeypatch.setenv("CORE_AI_INTERNAL_IDEMPOTENCY_ENABLED", "true")

    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return {"value": calls["n"]}

    payload = {"x": 1}
    r1, meta1 = cache.run_idempotent("op.test", payload, producer)
    assert meta1["hit"] is False and r1["value"] == 1

    r2, meta2 = cache.run_idempotent("op.test", payload, producer)
    assert meta2["hit"] is True and r2["value"] == 1  # cached

    # Wait for expiry and ensure producer called again
    time.sleep(1.2)
    r3, meta3 = cache.run_idempotent("op.test", payload, producer)
    assert meta3["hit"] is False and r3["value"] == 2


def test_run_idempotent_disabled(monkeypatch):
    monkeypatch.setenv("CORE_AI_INTERNAL_IDEMPOTENCY_ENABLED", "false")
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return {"value": calls["n"]}

    r1, m1 = cache.run_idempotent("op.disabled", {"a": 1}, producer)
    r2, m2 = cache.run_idempotent("op.disabled", {"a": 1}, producer)
    assert m1["hit"] is False and m2["hit"] is False
    assert r1["value"] == 1 and r2["value"] == 2
