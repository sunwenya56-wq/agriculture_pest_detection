"""QThread 封装：所有耗时操作（推理、建议、批量、导出、语音、摄像头）均在后台执行。"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QThread, pyqtSignal

from app.core.dto import AdviceResult, BatchItem, BatchResult, DetectionResult
from app.core.errors import AppError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.container import Container

logger = get_logger(__name__)


def describe_error(exc: BaseException) -> str:
    """统一错误文案：业务异常取 message，其余取 str。"""
    if isinstance(exc, AppError):
        return exc.message
    return str(exc) or exc.__class__.__name__


class DetectWorker(QThread):
    """单张图片检测线程。"""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, container: Container, image_path: str, model_path: Path | None = None, parent: Any = None):
        super().__init__(parent)
        self.container = container
        self.image_path = image_path
        self.model_path = Path(model_path) if model_path else None

    def run(self) -> None:
        try:
            if self.model_path is not None:
                current = Path(self.container.detector.model_path)
                if not _same_path(current, self.model_path):
                    self.container.detection.switch_model(self.model_path)
            result = self.container.detection.detect_image(self.image_path)
            self.finished_ok.emit(result)
        except BaseException as exc:  # noqa: BLE001
            logger.exception("单张检测线程失败", extra={"ctx_image": self.image_path})
            self.failed.emit(describe_error(exc))


class ModelSwitchWorker(QThread):
    """模型切换线程（加载权重可能耗时）。"""

    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, container: Container, model_path: Path, parent: Any = None):
        super().__init__(parent)
        self.container = container
        self.model_path = Path(model_path)

    def run(self) -> None:
        try:
            self.container.detection.switch_model(self.model_path)
            self.finished_ok.emit(str(self.model_path))
        except BaseException as exc:  # noqa: BLE001
            logger.exception("模型切换失败", extra={"ctx_model": str(self.model_path)})
            self.failed.emit(describe_error(exc))


class AdviceWorker(QThread):
    """防治建议生成线程。"""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        container: Container,
        pest_names: Sequence[str],
        parent: Any = None,
        *,
        generation: int = 0,
    ):
        super().__init__(parent)
        self.container = container
        self.pest_names = list(pest_names)
        self.generation = generation

    def run(self) -> None:
        try:
            result: AdviceResult = self.container.advice.generate(self.pest_names, use_cache=True)
            self.finished_ok.emit(result)
        except BaseException as exc:  # noqa: BLE001
            logger.exception("防治建议生成失败", extra={"ctx_pests": self.pest_names})
            self.failed.emit(describe_error(exc))


class BatchWorker(QThread):
    """批量检测线程，透传单文件进度。"""

    progress = pyqtSignal(int, int, object)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, container: Container, image_paths: Sequence[str], parent: Any = None):
        super().__init__(parent)
        self.container = container
        self.image_paths = list(image_paths)

    def run(self) -> None:
        try:
            result: BatchResult = self.container.batch.run(self.image_paths, self._on_progress)
            self.finished_ok.emit(result)
        except BaseException as exc:  # noqa: BLE001
            logger.exception("批量检测失败")
            self.failed.emit(describe_error(exc))

    def _on_progress(self, current: int, total: int, item: BatchItem) -> None:
        self.progress.emit(current, total, item)


class ExportWorker(QThread):
    """通用导出线程：调用方传入只依赖 container 的可调用对象。"""

    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, action: Callable[[], object], parent: Any = None):
        super().__init__(parent)
        self.action = action

    def run(self) -> None:
        try:
            target = self.action()
            self.finished_ok.emit(str(target))
        except BaseException as exc:  # noqa: BLE001
            logger.exception("导出线程失败")
            self.failed.emit(describe_error(exc))


class SpeechWorker(QThread):
    """语音播报开关线程，避免 stop() 的等待阻塞 UI。"""

    state_changed = pyqtSignal(bool)
    failed = pyqtSignal(str)

    def __init__(self, container: Container, text: str = "", *, stop: bool = False, parent: Any = None):
        super().__init__(parent)
        self.container = container
        self.text = text
        self.stop_requested = stop

    def run(self) -> None:
        try:
            if self.stop_requested:
                self.container.tts.stop()
                self.state_changed.emit(False)
            else:
                started = bool(self.container.tts.speak(self.text))
                self.state_changed.emit(started)
        except BaseException as exc:  # noqa: BLE001
            logger.exception("语音播报失败")
            self.failed.emit(describe_error(exc))


class CameraWorker(QThread):
    """摄像头取帧 + 定时推理线程；帧渲染与画框全部交给 UI 线程。"""

    frame_ready = pyqtSignal(object, object)  # RGB ndarray, DetectionResult | None
    opened = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        container: Container,
        *,
        fps: int = 10,
        detect_interval_s: float = 1.0,
        parent: Any = None,
    ):
        super().__init__(parent)
        self.container = container
        self._fps = max(1, int(fps))
        self._detect_interval_s = max(0.1, float(detect_interval_s))
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()

    @property
    def fps(self) -> int:
        with self._lock:
            return self._fps

    @property
    def detect_interval_s(self) -> float:
        with self._lock:
            return self._detect_interval_s

    def update_rates(self, fps: int, detect_interval_s: float) -> None:
        with self._lock:
            self._fps = max(1, int(fps))
            self._detect_interval_s = max(0.1, float(detect_interval_s))

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        capture = None
        try:
            import cv2  # 懒加载，UI 模块导入期不依赖 OpenCV

            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                self.failed.emit("无法打开摄像头\n请检查摄像头连接")
                return

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.opened.emit()

            last_detection = 0.0
            while self._running.is_set():
                frame_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                result: DetectionResult | None = None
                now = time.monotonic()
                if now - last_detection >= self.detect_interval_s:
                    try:
                        result = self.container.detection.detect_frame(frame)
                    except BaseException:  # noqa: BLE001
                        logger.exception("实时推理失败，跳过当前帧")
                    last_detection = time.monotonic()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb, result)

                delay = 1.0 / self.fps - (time.perf_counter() - frame_started)
                if delay > 0:
                    time.sleep(delay)
        except BaseException as exc:  # noqa: BLE001
            logger.exception("摄像头线程异常")
            self.failed.emit(describe_error(exc))
        finally:
            if capture is not None:
                capture.release()


def _same_path(left: Path | str, right: Path | str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)
