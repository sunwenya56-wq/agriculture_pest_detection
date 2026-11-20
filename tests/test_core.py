"""核心层测试：配置校验、DTO 契约、指标统计。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.dto import BBox, Detection, DetectionResult
from app.core.metrics import MetricsRecorder


def test_settings_rejects_invalid_confidence(tmp_path):
    with pytest.raises(ValidationError):
        Settings(models_dir=tmp_path, confidence=1.5)


def test_settings_effective_backend_prefers_cloud_when_key_present(tmp_path):
    settings = Settings(models_dir=tmp_path, qwen_api_key="test-key", llm_backend="auto")
    assert settings.effective_llm_backend == "cloud"
    assert settings.has_cloud_key is True


def test_detection_result_labels_are_deduplicated(tmp_path):
    result = DetectionResult(
        image_path="a.jpg",
        detections=[
            Detection(0, "地老虎", 0.9, BBox(1, 2, 3, 4)),
            Detection(0, "地老虎", 0.8, BBox(5, 6, 7, 8)),
            Detection(1, "田蟋蟀", 0.7, BBox(9, 10, 11, 12)),
        ],
    )
    assert result.count == 3
    assert result.labels == ["地老虎", "田蟋蟀"]
    assert result.max_confidence == 0.9
    payload = result.to_dict()
    assert payload["count"] == 3
    assert payload["detections"][0]["bbox"]["x1"] == 1


def test_metrics_snapshot_computes_percentiles(tmp_path):
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl")
    for value in (10, 20, 30, 40):
        recorder.record("detection", ok=True, elapsed_ms=value)
    recorder.record("advice", degraded=True, elapsed_ms=100)
    snapshot = recorder.snapshot()
    assert snapshot["detection"]["count"] == 4
    assert snapshot["detection"]["p50_ms"] >= 20
    assert snapshot["detection"]["p95_ms"] > 30
    assert snapshot["degrade_count"] == 1
