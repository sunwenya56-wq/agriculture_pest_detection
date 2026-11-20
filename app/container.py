"""依赖装配：统一构建 UI / API 共用的一套对象。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.adapters.exporter import Exporter
from app.adapters.llm import LLMClient, build_llm_client
from app.adapters.storage import SqliteRepository
from app.adapters.tts import TTSPlayer
from app.adapters.yolo import YoloDetector
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.core.metrics import MetricsRecorder
from app.services.advice import AdviceService
from app.services.batch import BatchService
from app.services.detection import DetectionService
from app.services.health import HealthService
from app.services.history import HistoryService

logger = get_logger(__name__)


def resolve_model_path(settings: Settings) -> Path:
    """优先读取 models_config.json 的当前模型，其次回落到 best.pt。"""
    config_file = settings.models_dir / "models_config.json"
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            current = data.get("current_model")
            for model in data.get("models", []):
                if model.get("name") == current:
                    candidate = Path(model.get("path", ""))
                    path = candidate if candidate.is_absolute() else settings.models_dir.parent / candidate
                    if path.exists():
                        return path
        except (OSError, json.JSONDecodeError):
            logger.warning("模型配置解析失败，回落到 best.pt", exc_info=True)
    return settings.models_dir / "best.pt"


@dataclass
class Container:
    settings: Settings
    model_path: Path
    detector: YoloDetector
    repo: SqliteRepository
    exporter: Exporter
    tts: TTSPlayer
    llm: LLMClient | None
    metrics: MetricsRecorder
    detection: DetectionService
    advice: AdviceService
    batch: BatchService
    history: HistoryService
    health: HealthService


def build_container(settings: Settings | None = None, *, configure_logging: bool = True) -> Container:
    settings = settings or get_settings()
    settings.ensure_dirs()
    if configure_logging:
        setup_logging(settings.log_dir, settings.log_level)

    model_path = resolve_model_path(settings)
    detector = YoloDetector(
        model_path,
        confidence=settings.confidence,
        iou=settings.iou,
        imgsz=settings.imgsz,
        max_det=settings.max_det,
        device=settings.device,
    )
    repo = SqliteRepository(settings.db_path)
    exporter = Exporter(settings.export_dir)
    metrics = MetricsRecorder(settings.log_dir / "metrics.jsonl")
    tts = TTSPlayer(settings)

    try:
        llm = build_llm_client(settings)
    except Exception:  # noqa: BLE001
        logger.warning("LLM 初始化失败，将使用规则兜底", exc_info=True)
        llm = None

    detection = DetectionService(detector, repo, metrics)
    advice = AdviceService(llm, repo, metrics, cache_ttl_s=settings.advice_cache_ttl_s)
    batch = BatchService(detection, workers=settings.batch_workers)
    history = HistoryService(repo)
    health = HealthService(settings, model_path, llm, tts)

    logger.info(
        "容器装配完成",
        extra={
            "ctx_model": str(model_path),
            "ctx_llm": advice.backend,
            "ctx_db": str(settings.db_path),
        },
    )
    return Container(
        settings=settings,
        model_path=model_path,
        detector=detector,
        repo=repo,
        exporter=exporter,
        tts=tts,
        llm=llm,
        metrics=metrics,
        detection=detection,
        advice=advice,
        batch=batch,
        history=history,
        health=health,
    )
