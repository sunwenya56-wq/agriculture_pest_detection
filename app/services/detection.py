"""检测服务：推理 + 指标 + 落库的编排入口。"""
from __future__ import annotations

from pathlib import Path

from app.adapters.storage import SqliteRepository
from app.adapters.yolo import YoloDetector
from app.core.dto import DetectionResult
from app.core.errors import DetectionError
from app.core.logging import get_logger, new_trace_id
from app.core.metrics import MetricsRecorder

logger = get_logger(__name__)


class DetectionService:
    def __init__(self, detector: YoloDetector, repo: SqliteRepository, metrics: MetricsRecorder):
        self.detector = detector
        self.repo = repo
        self.metrics = metrics

    def set_confidence(self, value: float) -> None:
        self.detector.update_confidence(value)

    def switch_model(self, model_path: str | Path) -> None:
        self.detector.switch_model(model_path)

    def detect_image(self, image_path: str | Path) -> DetectionResult:
        path = str(image_path)
        if not Path(path).exists():
            raise DetectionError(f"图片不存在: {path}")
        new_trace_id("det-")
        try:
            result = self.detector.detect(path)
        except DetectionError as exc:
            self.metrics.record("detection", ok=False, image=path, error=str(exc))
            logger.error("检测失败", extra={"ctx_image": path, "ctx_error": str(exc)})
            raise

        self.metrics.record(
            "detection",
            ok=True,
            image=path,
            elapsed_ms=result.elapsed_ms,
            count=result.count,
            degraded=result.degraded,
        )
        logger.info(
            "检测完成",
            extra={"ctx_image": path, "ctx_count": result.count, "ctx_elapsed_ms": result.elapsed_ms},
        )
        return result

    def detect_frame(self, frame: object) -> DetectionResult:
        result = self.detector.detect_frame(frame)
        self.metrics.record("detection", ok=True, image="<camera>", elapsed_ms=result.elapsed_ms,
                            count=result.count)
        return result

    def save_record(self, result: DetectionResult, advice: str, model_name: str, export_path: str = "") -> int:
        return self.repo.add_detection(result, advice, model_name, export_path)
