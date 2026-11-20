"""轻量指标采集：JSONL 追加写，线程安全，供状态栏与 /metrics 使用。"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * p
    low, high = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (idx - low)


class MetricsRecorder:
    def __init__(self, path: Path, *, memory_limit: int = 5000):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._memory_limit = memory_limit

    def record(self, event: str, **fields: Any) -> None:
        item = {"ts": datetime.now().isoformat(timespec="milliseconds"), "event": event, **fields}
        with self._lock:
            self._events.append(item)
            if len(self._events) > self._memory_limit:
                self._events = self._events[-self._memory_limit :]
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    @contextmanager
    def timed(self, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        payload: dict[str, Any] = {}
        try:
            yield payload
        finally:
            payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.record(event, **fields, **payload)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        detection = [e["elapsed_ms"] for e in events if e["event"] == "detection" and e.get("ok")]
        advice = [e["elapsed_ms"] for e in events if e["event"] == "advice"]
        advice_total = [e for e in events if e["event"] == "advice"]
        degraded = [e for e in events if e.get("degraded")]
        return {
            "events": len(events),
            "detection": {
                "count": len(detection),
                "p50_ms": round(_percentile(detection, 0.5), 2),
                "p95_ms": round(_percentile(detection, 0.95), 2),
            },
            "advice": {
                "count": len(advice_total),
                "success_rate": round(
                    sum(1 for e in advice_total if not e.get("degraded")) / len(advice_total), 4
                ) if advice_total else 0.0,
                "p95_ms": round(_percentile(advice, 0.95), 2),
            },
            "degrade_count": len(degraded),
        }
