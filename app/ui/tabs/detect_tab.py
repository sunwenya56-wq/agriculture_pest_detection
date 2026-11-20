"""单张图片检测标签页：预览、检测、建议、置信度/模型、导出、语音播报。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImageReader, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.dto import AdviceResult, DetectionResult
from app.core.logging import get_logger
from app.ui.workers import AdviceWorker, DetectWorker, ExportWorker, ModelSwitchWorker, SpeechWorker

if TYPE_CHECKING:
    from app.container import Container

logger = get_logger(__name__)


def render_with_boxes(
    pixmap: QPixmap, result: DetectionResult | None, target: QSize
) -> QPixmap:
    """按目标尺寸缩放并用 QPainter 画检测框，UI 层不使用 cv2。"""
    if pixmap.isNull():
        return pixmap
    scaled = pixmap.scaled(
        target,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    if result is None or not result.detections:
        return scaled

    canvas = scaled.copy()
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    base_w = result.image_width or pixmap.width() or 1
    base_h = result.image_height or pixmap.height() or 1
    scale_x = canvas.width() / base_w
    scale_y = canvas.height() / base_h

    box_pen = QPen(QColor("#e74c3c"))
    box_pen.setWidth(2)
    font = QFont("Microsoft YaHei")
    font.setPointSize(9)
    painter.setFont(font)
    metrics = painter.fontMetrics()

    for det in result.detections:
        rect = QRectF(
            det.bbox.x1 * scale_x,
            det.bbox.y1 * scale_y,
            max(1.0, (det.bbox.x2 - det.bbox.x1) * scale_x),
            max(1.0, (det.bbox.y2 - det.bbox.y1) * scale_y),
        )
        painter.setPen(box_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        text = f"{det.label} {det.confidence:.2f}"
        text_w = metrics.horizontalAdvance(text) + 8
        text_h = metrics.height() + 4
        text_x = max(0.0, rect.left())
        text_y = rect.top() - text_h
        if text_y < 0:
            text_y = rect.top() + 2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(231, 76, 60, 220))
        painter.drawRect(QRectF(text_x, text_y, text_w, text_h))
        painter.setPen(QColor("white"))
        painter.drawText(
            QRectF(text_x + 4, text_y + 2, text_w, text_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    painter.end()
    return canvas


def load_pixmap(image_path: str) -> QPixmap:
    """统一图片加载，兼容 Qt 默认插件不支持的 TIFF 等格式。"""
    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        reader = QImageReader(image_path)
        reader.setAutoTransform(True)
        image = reader.read()
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
    return pixmap


def display_target(label: QWidget, width: int = 800, height: int = 500) -> QSize:
    size = label.size()
    if size.width() < 50 or size.height() < 50:
        return QSize(width, height)
    return size


def load_model_catalog(models_dir: Path, fallback: Path) -> list[tuple[str, Path]]:
    """读取 models_config.json 生成下拉项，失败时回落到当前模型。"""
    config_file = models_dir / "models_config.json"
    entries: list[tuple[str, Path]] = []
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            for item in data.get("models", []):
                raw = Path(str(item.get("path", "")))
                path = raw if raw.is_absolute() else models_dir.parent / raw
                name = str(item.get("name") or path.stem)
                entries.append((name, path))
        except (OSError, ValueError):
            logger.warning("模型配置读取失败，使用当前模型", exc_info=True)
    if not entries:
        entries = [(fallback.name, fallback)]
    return entries


class DetectTab(QWidget):
    status_message = pyqtSignal(str)
    detection_latency = pyqtSignal(float)
    model_changed = pyqtSignal(str)
    metrics_refresh = pyqtSignal()
    request_camera = pyqtSignal()

    def __init__(self, container: Container, parent: QWidget | None = None):
        super().__init__(parent)
        self.container = container
        self.current_image_path: str | None = None
        self.current_result: DetectionResult | None = None
        self.current_model_path = Path(container.model_path)
        self._detection_model_name = self.current_model_path.name
        self._advice_text = ""
        self._speaking = False
        self._models_loading = False
        self._generation = 0
        self._pending_advice: list[str] | None = None
        self._detect_worker: DetectWorker | None = None
        self._advice_worker: AdviceWorker | None = None
        self._switch_worker: ModelSwitchWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._speech_worker: SpeechWorker | None = None

        self._speech_timer = QTimer(self)
        self._speech_timer.setInterval(800)
        self._speech_timer.timeout.connect(self._poll_speech_state)

        self._build_ui()
        self._load_models()
        self._fill_recent_history()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)

        button_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁 选择图片")
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.clicked.connect(self.select_image)
        self.camera_btn = QPushButton("📷 开启摄像头")
        self.camera_btn.setMinimumHeight(40)
        self.camera_btn.clicked.connect(lambda: self.request_camera.emit())
        self.export_btn = QPushButton("💾 导出结果")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        self.speak_btn = QPushButton("🔊 语音播报")
        self.speak_btn.setMinimumHeight(40)
        self.speak_btn.setEnabled(bool(self.container.settings.tts_enabled))
        self.speak_btn.clicked.connect(self.toggle_speech)
        for button in (self.upload_btn, self.camera_btn, self.export_btn, self.speak_btn):
            button_row.addWidget(button)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(round(self.container.settings.confidence * 100)))
        self.conf_label = QLabel(f"{self.container.settings.confidence:.2f}")
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)
        param_row.addWidget(self.conf_slider)
        param_row.addWidget(self.conf_label)
        param_row.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        param_row.addWidget(self.model_combo)

        control_layout.addLayout(button_row)
        control_layout.addLayout(param_row)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet(
            "border: 2px solid #3498db; border-radius: 8px; background-color: #ecf0f1;"
        )

        detail_group = QGroupBox("检测详情")
        detail_layout = QVBoxLayout(detail_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        detail_layout.addWidget(self.result_text)

        left_layout.addWidget(control_group)
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(detail_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        advice_group = QGroupBox("防治建议")
        advice_layout = QVBoxLayout(advice_group)
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setMinimumHeight(500)
        advice_layout.addWidget(self.advice_text)

        history_group = QGroupBox("最近检测")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        self.history_list.itemClicked.connect(self._on_history_item_clicked)
        history_layout.addWidget(self.history_list)

        right_layout.addWidget(advice_group)
        right_layout.addWidget(history_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 800])
        layout.addWidget(splitter)

    # ---------- 模型与置信度 ----------
    def _load_models(self) -> None:
        self._models_loading = True
        try:
            self.model_combo.clear()
            entries = load_model_catalog(self.container.settings.models_dir, self.current_model_path)
            matched = any(self._same_path(path, self.current_model_path) for _, path in entries)
            if not matched:
                entries.insert(0, (self.current_model_path.name, self.current_model_path))
            for name, path in entries:
                self.model_combo.addItem(name, str(path))
            index = 0
            for i, (_, path) in enumerate(entries):
                if self._same_path(path, self.current_model_path):
                    index = i
                    break
            self.model_combo.setCurrentIndex(index)
            self.current_model_path = Path(self.model_combo.currentData() or self.current_model_path)
            self._detection_model_name = self.model_combo.currentText()
        finally:
            self._models_loading = False

    @staticmethod
    def _same_path(left: Path | str, right: Path | str) -> bool:
        try:
            return Path(left).resolve() == Path(right).resolve()
        except OSError:
            return str(left) == str(right)

    def _on_confidence_changed(self, value: int) -> None:
        confidence = value / 100
        self.conf_label.setText(f"{confidence:.2f}")
        self.container.detection.set_confidence(confidence)

    def _on_model_selected(self, _index: int) -> None:
        if self._models_loading:
            return
        data = self.model_combo.currentData()
        if not data:
            return
        target = Path(str(data))
        if self._same_path(target, self.current_model_path):
            return
        if self._switch_worker is not None and self._switch_worker.isRunning():
            return
        self.conf_slider.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.status_message.emit(f"正在切换模型: {target.name}")
        worker = ModelSwitchWorker(self.container, target, self)
        worker.finished_ok.connect(self._on_model_switched)
        worker.failed.connect(self._on_model_switch_failed)
        worker.finished.connect(lambda: self._enable_model_controls())
        self._switch_worker = worker
        worker.start()

    def _enable_model_controls(self) -> None:
        self.conf_slider.setEnabled(True)
        self.model_combo.setEnabled(True)

    def _on_model_switched(self, path: str) -> None:
        self.current_model_path = Path(path)
        self._detection_model_name = self.model_combo.currentText()
        self.model_changed.emit(self._detection_model_name)
        self.status_message.emit(f"模型已切换: {self._detection_model_name}")

    def _on_model_switch_failed(self, message: str) -> None:
        self.status_message.emit("模型切换失败")
        QMessageBox.warning(self, "模型切换失败", message)
        self._models_loading = True
        index = self.model_combo.findData(str(self.current_model_path))
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self._models_loading = False

    # ---------- 单张检测 ----------
    def select_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*.*)",
        )
        if file_path:
            self._start_detection(file_path)

    def _start_detection(self, image_path: str) -> None:
        if self._detect_worker is not None and self._detect_worker.isRunning():
            QMessageBox.information(self, "提示", "检测正在进行中，请稍候")
            return

        self.current_image_path = image_path
        self.current_result = None
        self._advice_text = ""
        self._generation += 1
        self._pending_advice = None
        self._detection_model_name = self.model_combo.currentText() or self.current_model_path.name
        self.export_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.conf_slider.setEnabled(False)
        self.model_combo.setEnabled(False)
        self._show_original(image_path)
        self.result_text.setPlainText("正在检测中，请稍候...")
        self.advice_text.setPlainText("检测完成后将生成防治建议...")
        self.status_message.emit("正在检测中...")

        self.container.detection.set_confidence(self.conf_slider.value() / 100)
        worker = DetectWorker(self.container, image_path, self.current_model_path, self)
        worker.finished_ok.connect(self._on_detection_finished)
        worker.failed.connect(self._on_detection_failed)
        worker.finished.connect(self._restore_detection_controls)
        self._detect_worker = worker
        worker.start()

    def _restore_detection_controls(self) -> None:
        self.upload_btn.setEnabled(True)
        self.conf_slider.setEnabled(True)
        self.model_combo.setEnabled(True)

    def _on_detection_finished(self, result: DetectionResult) -> None:
        self.current_result = result
        self.export_btn.setEnabled(True)
        self._show_with_boxes(result)

        if result.detections:
            lines = [f"✅ 检测到 {result.count} 个害虫:"]
            for i, det in enumerate(result.detections, start=1):
                lines.append(f"  {i}. {det.label} (置信度: {det.confidence:.2%})")
            self.result_text.setPlainText("\n".join(lines))
            self.advice_text.setPlainText("正在生成防治建议，请稍候...")
            self._start_advice(result.labels)
        else:
            self.result_text.setPlainText("✅ 未检测到害虫")
            self.advice_text.setPlainText("未检测到害虫，无需防治。")
            self._advice_text = "未检测到害虫，无需防治。"
            self._save_record()

        self.detection_latency.emit(result.elapsed_ms)
        self.metrics_refresh.emit()
        self.status_message.emit(f"检测完成，耗时 {result.elapsed_ms:.0f} ms")

    def _on_detection_failed(self, message: str) -> None:
        self.result_text.setPlainText(message)
        self.status_message.emit("检测失败")
        QMessageBox.critical(self, "检测错误", message)

    def _start_advice(self, labels: list[str]) -> None:
        self._pending_advice = list(labels)
        if self._advice_worker is not None and self._advice_worker.isRunning():
            return
        self._launch_pending_advice()

    def _launch_pending_advice(self) -> None:
        labels = self._pending_advice
        if not labels:
            return
        self._pending_advice = None
        worker = AdviceWorker(self.container, labels, self, generation=self._generation)
        worker.finished_ok.connect(self._on_advice_finished)
        worker.failed.connect(self._on_advice_failed)
        self._advice_worker = worker
        worker.start()

    def _on_advice_finished(self, result: AdviceResult) -> None:
        worker = self.sender()
        generation = getattr(worker, "generation", self._generation)
        if generation != self._generation:
            if self._pending_advice is not None:
                self._launch_pending_advice()
            return

        self._advice_text = result.text
        self.advice_text.setPlainText(result.text)
        source_text = f"建议来源: {result.source}"
        if result.cached:
            source_text += "（缓存）"
        if result.degraded:
            source_text += "（规则兜底）"
        self.status_message.emit(f"{source_text}，耗时 {result.elapsed_ms:.0f} ms")
        self.metrics_refresh.emit()
        self._save_record()

    def _on_advice_failed(self, message: str) -> None:
        worker = self.sender()
        generation = getattr(worker, "generation", self._generation)
        if generation != self._generation:
            if self._pending_advice is not None:
                self._launch_pending_advice()
            return

        self._advice_text = f"建议生成失败: {message}"
        self.advice_text.setPlainText(self._advice_text)
        self.status_message.emit("建议生成失败")
        QMessageBox.warning(self, "建议生成失败", message)
        self._save_record()

    def _save_record(self) -> None:
        if self.current_result is None:
            return
        try:
            self.container.detection.save_record(
                self.current_result,
                self._advice_text,
                self._detection_model_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存检测记录失败: %s", exc)
            self.status_message.emit("检测记录保存失败")
            return
        self._fill_recent_history()

    # ---------- 图片渲染 ----------
    def _show_original(self, image_path: str) -> None:
        pixmap = load_pixmap(image_path)
        if pixmap.isNull():
            self.status_message.emit("无法读取图片")
            return
        target = display_target(self.image_label)
        self.image_label.setPixmap(
            pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _show_with_boxes(self, result: DetectionResult) -> None:
        if not Path(result.image_path).exists():
            return
        pixmap = load_pixmap(result.image_path)
        if pixmap.isNull():
            return
        self.image_label.setPixmap(render_with_boxes(pixmap, result, display_target(self.image_label)))

    # ---------- 最近检测 ----------
    def _fill_recent_history(self) -> None:
        self.history_list.clear()
        try:
            records = self.container.history.list(limit=10)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取最近检测失败: %s", exc)
            return
        for record in records:
            pests = "、".join(record.get("pests", [])[:3])
            if len(record.get("pests", [])) > 3:
                pests += "..."
            time_text = str(record.get("detection_time", ""))
            time_part = time_text.split()[-1][:5] if time_text else "--:--"
            filename = Path(str(record.get("image_path", ""))).name
            item = QListWidgetItem(f"{time_part} - {filename} - {pests or '无害虫'}")
            item.setData(Qt.ItemDataRole.UserRole, record.get("id"))
            self.history_list.addItem(item)

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        record_id = item.data(Qt.ItemDataRole.UserRole)
        if record_id is None:
            return
        try:
            record = self.container.history.get(int(record_id))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "警告", f"加载历史记录失败: {exc}")
            return
        if not record:
            return

        pests = "、".join(record.get("pests", [])) or "无"
        self.result_text.setPlainText(
            f"历史记录 #{record.get('id')}\n"
            f"检测时间: {record.get('detection_time')}\n"
            f"模型: {record.get('model_used') or '默认'}\n"
            f"检出害虫: {pests}\n"
            f"耗时: {float(record.get('elapsed_ms') or 0):.0f} ms"
        )
        self.advice_text.setPlainText(record.get("advice") or "该记录未保存防治建议。")
        image_path = str(record.get("image_path", ""))
        if Path(image_path).exists():
            self._show_original(image_path)
        self.current_result = None
        self.export_btn.setEnabled(False)

    # ---------- 导出 ----------
    def export_results(self) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "警告", "请先检测图片")
            return
        formats = ["带标记图片", "JSON", "CSV", "Excel"]
        choice, ok = QInputDialog.getItem(self, "选择导出格式", "请选择导出格式:", formats, 0, False)
        if not ok:
            return

        result = self.current_result
        advice = self.advice_text.toPlainText()
        exporter = self.container.exporter
        if choice == "带标记图片":
            action = lambda: exporter.export_annotated_image(result)  # noqa: E731
        elif choice == "JSON":
            action = lambda: exporter.export_json(result, advice)  # noqa: E731
        elif choice == "CSV":
            action = lambda: exporter.export_csv(result)  # noqa: E731
        else:
            action = lambda: exporter.export_xlsx(result, advice)  # noqa: E731

        if self._export_worker is not None and self._export_worker.isRunning():
            return
        worker = ExportWorker(action, self)
        worker.finished_ok.connect(lambda path: QMessageBox.information(self, "成功", f"导出成功:\n{path}"))
        worker.failed.connect(lambda message: QMessageBox.critical(self, "错误", f"导出失败: {message}"))
        self._export_worker = worker
        worker.start()
        self.status_message.emit("正在导出结果...")

    # ---------- 语音播报 ----------
    def toggle_speech(self) -> None:
        if not self.container.settings.tts_enabled:
            return
        if self._speech_worker is not None and self._speech_worker.isRunning():
            return
        if self._speaking:
            worker = SpeechWorker(self.container, stop=True, parent=self)
            worker.state_changed.connect(self._on_speech_state)
            worker.failed.connect(self._on_speech_failed)
            self._speech_worker = worker
            worker.start()
            return

        text = self.advice_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "暂无可播报的防治建议")
            return
        worker = SpeechWorker(self.container, text=text, parent=self)
        worker.state_changed.connect(self._on_speech_state)
        worker.failed.connect(self._on_speech_failed)
        self._speech_worker = worker
        worker.start()

    def _on_speech_state(self, speaking: bool) -> None:
        self._speaking = speaking
        if speaking:
            self.speak_btn.setText("⏹️ 停止播报")
            self._speech_timer.start()
            self.status_message.emit("正在语音播报")
        else:
            self.speak_btn.setText("🔊 语音播报")
            self._speech_timer.stop()
            self.status_message.emit("语音播报已停止")

    def _on_speech_failed(self, message: str) -> None:
        self._on_speech_state(False)
        QMessageBox.warning(self, "语音播报失败", message)

    def _poll_speech_state(self) -> None:
        if self._speaking and not self.container.tts.is_speaking:
            self._on_speech_state(False)

    # ---------- 生命周期 ----------
    def shutdown(self) -> None:
        try:
            self.container.tts.stop()
        except Exception:  # noqa: BLE001
            pass
        self._speech_timer.stop()
        for worker in (self._detect_worker, self._advice_worker, self._switch_worker,
                       self._export_worker, self._speech_worker):
            if worker is not None and worker.isRunning():
                worker.wait(1500)
