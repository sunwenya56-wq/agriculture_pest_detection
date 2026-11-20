"""API 装配冒烟测试（不触发模型加载与网络请求）。"""
from __future__ import annotations

from app.api.server import create_app


def test_app_builds_without_container():
    app = create_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/health" in paths
    assert "/v1/detect" in paths
    assert "/v1/advice" in paths
    assert "/v1/metrics" in paths
