"""健康检查：启动自检与 /health 端点共用。"""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.dto import HealthReport
from app.core.logging import get_logger

logger = get_logger(__name__)


class HealthService:
    def __init__(self, settings: Settings, model_path: str | Path, llm_client: object | None, tts: object | None):
        self.settings = settings
        self.model_path = Path(model_path)
        self.llm = llm_client
        self.tts = tts

    def check(self, *, deep: bool = False) -> HealthReport:
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        checks["config"] = True
        details["config"] = f"env={self.settings.env}, backend={self.settings.effective_llm_backend}"

        checks["model"] = self.model_path.exists()
        details["model"] = str(self.model_path)

        try:
            self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
            checks["storage"] = self.settings.db_path.exists() or self.settings.db_path.parent.exists()
            details["storage"] = str(self.settings.db_path)
        except OSError as exc:
            checks["storage"] = False
            details["storage"] = str(exc)

        if self.llm is None:
            checks["llm"] = False
            details["llm"] = "未配置可用通道，将使用规则兜底"
        elif deep and hasattr(self.llm, "healthcheck"):
            ok = bool(self.llm.healthcheck())
            checks["llm"] = ok
            details["llm"] = getattr(self.llm, "source", "unknown")
        else:
            checks["llm"] = True
            details["llm"] = getattr(self.llm, "source", "unknown")

        checks["tts"] = bool(self.tts and getattr(self.tts, "is_available", lambda: False)())

        status = "ok" if checks["config"] and checks["model"] and checks["storage"] else "degraded"
        report = HealthReport(status=status, checks=checks, details=details)
        logger.info("健康检查完成", extra={"ctx_status": status, "ctx_checks": checks})
        return report
