"""YOLOv11 检测适配器：重型依赖懒加载，便于服务层脱离 GUI 单测。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.core.dto import BBox, Detection, DetectionResult
from app.core.errors import DetectionError, ModelLoadError
from app.core.logging import get_logger

logger = get_logger(__name__)


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        iou: float = 0.15,
        imgsz: int = 640,
        max_det: int = 50,
        device: str = "auto",
    ):
        self.model_path = str(model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        self._model: Any = None
        self._device: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.model_path).exists():
            raise ModelLoadError(f"模型文件不存在: {self.model_path}")
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self._device = self._resolve_device()
            logger.info("模型加载完成", extra={"ctx_path": self.model_path, "ctx_device": self._device})
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"加载模型失败: {exc}") from exc

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def update_confidence(self, value: float) -> None:
        self.confidence = max(0.0, min(1.0, float(value)))

    def switch_model(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self._model = None
        self.load()

    def detect(self, image_path: str | Path) -> DetectionResult:
        self.load()
        started = time.perf_counter()
        try:
            results = self._model.predict(
                source=str(image_path),
                conf=self.confidence,
                iou=self.iou,
                verbose=False,
                device=self._device,
                max_det=self.max_det,
                imgsz=self.imgsz,
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"推理失败: {exc}") from exc
        return self._to_dto(results, image_path=str(image_path), started=started)

    def detect_frame(self, frame: Any) -> DetectionResult:
        self.load()
        started = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                verbose=False,
                device=self._device,
                max_det=self.max_det,
                imgsz=self.imgsz,
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"实时推理失败: {exc}") from exc
        return self._to_dto(results, image_path="<camera>", started=started)

    def _to_dto(self, results: list[Any], *, image_path: str, started: float) -> DetectionResult:
        elapsed = (time.perf_counter() - started) * 1000
        if not results:
            return DetectionResult(image_path=image_path, elapsed_ms=elapsed, model_name=self.model_path)

        result = results[0]
        shape = getattr(result, "orig_shape", None) or (0, 0)
        height, width = int(shape[0]), int(shape[1])
        names: dict[int, str] = dict(getattr(result, "names", {}) or {})

        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
            conf_list = boxes.conf.cpu().numpy().tolist()
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            for class_id, conf, coords in zip(cls_list, conf_list, xyxy, strict=False):
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        label=names.get(int(class_id), f"未知类别{class_id}"),
                        confidence=float(conf),
                        bbox=BBox(*[float(c) for c in coords]),
                    )
                )

        return DetectionResult(
            image_path=image_path,
            detections=detections,
            image_width=width,
            image_height=height,
            elapsed_ms=round(elapsed, 2),
            model_name=Path(self.model_path).name,
        )
