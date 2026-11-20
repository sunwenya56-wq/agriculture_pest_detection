"""领域模型：CV 与 LLM 之间的唯一数据契约。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BBox
    label_en: str = ""


@dataclass
class DetectionResult:
    image_path: str
    detections: list[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    elapsed_ms: float = 0.0
    model_name: str = ""
    model_version: str = "1.0.0"
    degraded: bool = False
    error: str | None = None
    created_at: str = field(default_factory=_now)

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def is_empty(self) -> bool:
        return not self.detections

    @property
    def labels(self) -> list[str]:
        # 去重保序，便于传给 LLM
        seen: dict[str, None] = {}
        for det in self.detections:
            seen.setdefault(det.label, None)
        return list(seen)

    @property
    def max_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["count"] = self.count
        data["labels"] = self.labels
        return data


@dataclass
class AdviceResult:
    text: str
    source: str = "cloud"  # cloud / local / cache / fallback
    model: str = ""
    elapsed_ms: float = 0.0
    cached: bool = False
    degraded: bool = False
    prompt_version: str = "v1"
    error: str | None = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchItem:
    image_path: str
    result: DetectionResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


@dataclass
class BatchResult:
    items: list[BatchItem] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(1 for i in self.items if i.ok)

    @property
    def pest_count(self) -> int:
        return sum(i.result.count for i in self.items if i.result)

    def pest_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for item in self.items:
            if not item.result:
                continue
            for det in item.result.detections:
                dist[det.label] = dist.get(det.label, 0) + 1
        return dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "pest_count": self.pest_count,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "distribution": self.pest_distribution(),
            "items": [
                {
                    "image_path": i.image_path,
                    "ok": i.ok,
                    "error": i.error,
                    "result": i.result.to_dict() if i.result else None,
                }
                for i in self.items
            ],
        }


@dataclass
class HealthReport:
    status: str = "ok"  # ok / degraded
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    checked_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
