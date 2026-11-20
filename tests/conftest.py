"""测试夹具：构造隔离的临时配置与依赖。"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.metrics import MetricsRecorder


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        models_dir=tmp_path / "models",
        export_dir=tmp_path / "exports",
        log_dir=tmp_path / "logs",
        db_path=tmp_path / "history.db",
        qwen_api_key="",
    )


@pytest.fixture()
def metrics(tmp_path: Path) -> MetricsRecorder:
    return MetricsRecorder(tmp_path / "metrics.jsonl")
