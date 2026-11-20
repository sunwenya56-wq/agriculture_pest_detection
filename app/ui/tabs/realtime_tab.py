"""实时摄像头标签页：开关、帧率/检测间隔控制、实时框选、停止释放摄像头。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.dto import AdviceResult, DetectionResult
from app.core.logging import get_logger
from app.ui.tabs.detect_tab import display_target, render_with_boxes
from app.ui.workers import AdviceWorker, CameraWorker

if TYPE_CHECKING:
    from app.container import Container

logger = get_logger(__name__)


class RealtimeTab(QWidget):
    status_message = pyqtSignal(str)
    metrics_refresh = pyqtSignal(str)

    def __init__(self, container: Container, parent: QWidget | None = None):
        super().__init__(parent)
        self.container = container
        self._worker: CameraWorker | None = None
        self._advice_worker: AdviceWorker | None = None
        self._active = False
        self._last_result: DetectionResult | None = None
        self._last_advice_labels: list[str] = []
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        control_group = QGroupBox("实时检测控制")
        control_layout = QHBoxLayout(control_group)
        self.start_btn = QPushButton("▶️ 开始实时检测")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("⏹️ 停止检测")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(10)
        self.fps_spin.setSuffix(" fps")
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.2, 5.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(float(self.container.settings.detection_interval_s))
        self.interval_spin.setSuffix(" 秒")
        self.fps_spin.valueChanged.connect(self._apply_rates)
        self.interval_spin.valueChanged.connect(self._apply_rates)

        for widget in (
            self.start_btn,
            self.stop_btn,
            QLabel("画面帧率:"),
            self.fps_spin,
            QLabel("检测间隔:"),
            self.interval_spin,
        ):
            control_layout.addWidget(widget)
        control_layout.addStretch()

        display_group = QGroupBox("实时画面")
        display_layout = QVBoxLayout(display_group)
        self.image_label = QLabel("摄像头已关闭\n点击'开始实时检测'重新启动")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet(
            "border: 2px solid #e74c3c; border-radius: 8px; background-color: #2c3e50; color: white;"
        )
        display_layout.addWidget(self.image_label)

        result_group = QGroupBox("实时检测结果")
        result_layout = QHBoxLayout(result_group)
        detail_group = QGroupBox("检测详情")
        detail_layout = QVBoxLayout(detail_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        detail_layout.addWidget(self.result_text)
        advice_group = QGroupBox("防治建议")
        advice_layout = QVBoxLayout(advice_group)
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        advice_layout.addWidget(self.advice_text)
        result_layout.addWidget(detail_group)
        result_layout.addWidget(advice_group)

        layout.addWidget(control_group)
        layout.addWidget(display_group)
        layout.addWidget(result_group)

    # ---------- 启停 ----------
    def start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._active = True
        self._last_result = None
        self._last_advice_labels = []
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("正在启动摄像头...\n请稍候")
        self.result_text.setPlainText("正在启动摄像头...")
        self.advice_text.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        worker = CameraWorker(
            self.container,
            fps=self.fps_spin.value(),
            detect_interval_s=self.interval_spin.value(),
            parent=self,
        )
        worker.opened.connect(lambda: self.status_message.emit("摄像头已启动"))
        worker.frame_ready.connect(self._on_frame)
        worker.failed.connect(self._on_camera_failed)
        self._worker = worker
        worker.start()
        self.status_message.emit("正在启动摄像头...")

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.stop()
            if worker.isRunning():
                worker.wait(3000)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.image_label.clear()
        self.image_label.setText("摄像头已关闭\n点击'开始实时检测'重新启动")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_text.setPlainText("检测已停止")
        self.advice_text.clear()
        self._last_result = None
        self.status_message.emit("实时检测已停止，摄像头已释放")

    def _apply_rates(self, _value: object = None) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.update_rates(self.fps_spin.value(), self.interval_spin.value())

    def _on_camera_failed(self, message: str) -> None:
        self._active = False
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.image_label.setText(f"摄像头启动失败:\n{message}")
        self.result_text.setPlainText("检测已停止")
        self.status_message.emit("摄像头启动失败")
        QMessageBox.critical(self, "摄像头错误", message)

    # ---------- 帧处理 ----------
    def _on_frame(self, rgb: object, result: DetectionResult | None) -> None:
        if self.sender() is not self._worker:
            return
        pixmap = _rgb_to_pixmap(rgb)
        if pixmap.isNull():
            return

        if result is not None:
            self._last_result = result
            self._update_result_text(result)
            self._maybe_generate_advice(result)

        rendered = render_with_boxes(pixmap, self._last_result, display_target(self.image_label, 800, 500))
        self.image_label.setPixmap(rendered)

    def _update_result_text(self, result: DetectionResult) -> None:
        if result.detections:
            lines = [f"✅ 检测到 {result.count} 个害虫:"]
            for i, det in enumerate(result.detections, start=1):
                lines.append(f"  {i}. {det.label} ({det.confidence:.1%})")
            self.result_text.setPlainText("\n".join(lines))
        else:
            self.result_text.setPlainText("当前未检测到害虫")

    def _maybe_generate_advice(self, result: DetectionResult) -> None:
        labels = result.labels
        if not labels:
            if self._last_advice_labels:
                self._last_advice_labels = []
                self.advice_text.setPlainText("当前未检测到害虫，无需防治。")
            return
        if labels == self._last_advice_labels:
            return
        if self._advice_worker is not None and self._advice_worker.isRunning():
            return

        self._last_advice_labels = labels
        self.advice_text.setPlainText("正在生成防治建议，请稍候...")
        worker = AdviceWorker(self.container, labels, self)
        worker.finished_ok.connect(self._on_advice_finished)
        worker.failed.connect(self._on_advice_failed)
        self._advice_worker = worker
        worker.start()

    def _on_advice_finished(self, advice: AdviceResult) -> None:
        self.advice_text.setPlainText(advice.text)
        self.status_message.emit(f"实时防治建议已更新（{advice.source}）")

    def _on_advice_failed(self, message: str) -> None:
        self.advice_text.setPlainText(f"建议生成失败: {message}")

    # ---------- 生命周期 ----------
    def shutdown(self) -> None:
        self._active = False
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.stop()
            if worker.isRunning():
                worker.wait(3000)
        if self._advice_worker is not None and self._advice_worker.isRunning():
            self._advice_worker.wait(1000)


def _rgb_to_pixmap(rgb: object) -> QPixmap:
    """RGB ndarray -> QPixmap；tobytes() 拷贝避免依赖 numpy 缓冲区生命周期。"""
    height, width = rgb.shape[:2]  # type: ignore[attr-defined]
    buffer = rgb.tobytes()  # type: ignore[attr-defined]
    image = QImage(buffer, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)
