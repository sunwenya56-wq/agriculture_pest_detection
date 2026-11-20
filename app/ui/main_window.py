"""主窗口壳：菜单栏、状态栏、三个标签页与历史入口。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.logging import get_logger
from app.ui.dialogs.history_dialog import HistoryDialog
from app.ui.styles import APP_STYLESHEET, DARK_STYLESHEET
from app.ui.tabs.batch_tab import BatchTab
from app.ui.tabs.detect_tab import DetectTab
from app.ui.tabs.realtime_tab import RealtimeTab

if TYPE_CHECKING:
    from app.container import Container
    from app.core.dto import HealthReport

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, container: Container, parent: QWidget | None = None):
        super().__init__(parent)
        self.container = container
        self.setWindowTitle(container.settings.app_name)
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(APP_STYLESHEET)
        self._dark_mode = False

        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        self._connect_tabs()

        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(5000)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start()
        self._refresh_metrics()

        QTimer.singleShot(0, self._run_health_check)

    # ---------- 布局 ----------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.tabs = QTabWidget()
        self.detect_tab = DetectTab(self.container, self)
        self.batch_tab = BatchTab(self.container, self)
        self.realtime_tab = RealtimeTab(self.container, self)
        self.tabs.addTab(self.detect_tab, "单张检测")
        self.tabs.addTab(self.batch_tab, "批量检测")
        self.tabs.addTab(self.realtime_tab, "实时检测")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        import_action = QAction("导入图片", self)
        import_action.triggered.connect(self.detect_tab.select_image)
        file_menu.addAction(import_action)
        batch_action = QAction("批量导入", self)
        batch_action.triggered.connect(self.batch_tab.select_folder)
        file_menu.addAction(batch_action)
        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("视图")
        history_action = QAction("历史记录", self)
        history_action.triggered.connect(self._show_history)
        view_menu.addAction(history_action)
        self.dark_action = QAction("深色模式", self)
        self.dark_action.setCheckable(True)
        self.dark_action.triggered.connect(self._toggle_dark_mode)
        view_menu.addAction(self.dark_action)

        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #ecf0f1;")
        self.model_label = QLabel(f"模型: {self.container.model_path.name}")
        self.llm_label = QLabel(f"LLM: {self._llm_backend()}")
        self.elapsed_label = QLabel("耗时: -")
        self.metrics_label = QLabel("检测P95: -")
        self.health_label = QLabel("自检: 进行中...")
        self.status_bar.addWidget(self.status_label)
        for widget in (
            self.model_label,
            self.llm_label,
            self.elapsed_label,
            self.metrics_label,
            self.health_label,
        ):
            widget.setStyleSheet("color: #ecf0f1; padding: 0 6px;")
            self.status_bar.addPermanentWidget(widget)

    def _connect_tabs(self) -> None:
        for tab in (self.detect_tab, self.batch_tab, self.realtime_tab):
            tab.status_message.connect(self.status_label.setText)
        self.detect_tab.model_changed.connect(lambda name: self.model_label.setText(f"模型: {name}"))
        self.detect_tab.detection_latency.connect(
            lambda elapsed: self.elapsed_label.setText(f"耗时: {elapsed:.0f} ms")
        )
        self.detect_tab.metrics_refresh.connect(self._refresh_metrics)
        self.detect_tab.request_camera.connect(self._open_camera)
        self.batch_tab.metrics_refresh.connect(lambda _message: self._refresh_metrics())
        self.realtime_tab.metrics_refresh.connect(lambda _message: self._refresh_metrics())

    # ---------- 状态 ----------
    def _llm_backend(self) -> str:
        try:
            return self.container.advice.backend
        except Exception:  # noqa: BLE001
            return "unknown"

    def _refresh_metrics(self) -> None:
        try:
            snapshot = self.container.metrics.snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("指标读取失败: %s", exc)
            return
        detection = snapshot.get("detection") or {}
        advice = snapshot.get("advice") or {}
        self.metrics_label.setText(
            f"检测P95: {float(detection.get('p95_ms', 0)):.0f} ms | "
            f"建议成功率: {float(advice.get('success_rate', 0)):.0%}"
        )

    def _run_health_check(self) -> None:
        try:
            report: HealthReport = self.container.health.check(deep=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("启动自检失败")
            self.health_label.setText("自检: 失败")
            self.health_label.setToolTip(str(exc))
            self.status_label.setText("就绪（自检失败）")
            return

        checks = report.checks
        ok_text = f"自检: {'正常' if report.status == 'ok' else '降级'}"
        symbols = {
            "model": "模型",
            "storage": "存储",
            "llm": "大模型",
            "tts": "语音",
        }
        brief = " ".join(
            f"{label}{'✓' if checks.get(key) else '✗'}" for key, label in symbols.items()
        )
        self.health_label.setText(f"{ok_text} | {brief}")
        self.health_label.setStyleSheet(
            f"color: {'#2ecc71' if report.status == 'ok' else '#f39c12'};"
        )
        detail_lines = [f"{key}: {'正常' if value else '不可用'} - {report.details.get(key, '')}"
                        for key, value in checks.items()]
        self.health_label.setToolTip("\n".join(detail_lines))
        self.llm_label.setText(f"LLM: {self._llm_backend()}")
        self.status_label.setText(
            f"就绪 · 启动自检{'通过' if report.status == 'ok' else '降级，将使用兜底能力'}"
        )

    # ---------- 菜单动作 ----------
    def _on_tab_changed(self, index: int) -> None:
        if index != 2:
            self.realtime_tab.stop()

    def _open_camera(self) -> None:
        self.tabs.setCurrentIndex(2)
        self.realtime_tab.start()

    def _show_history(self) -> None:
        dialog = HistoryDialog(self.container, self)
        dialog.exec()

    def _toggle_dark_mode(self, checked: bool) -> None:
        self._dark_mode = checked
        self.setStyleSheet(DARK_STYLESHEET if checked else APP_STYLESHEET)

    def _show_about(self) -> None:
        settings = self.container.settings
        about_text = f"""
        <h2>{settings.app_name}</h2>
        <p><b>版本:</b> {settings.version}</p>
        <p><b>功能:</b></p>
        <ul>
            <li>单张图片检测</li>
            <li>批量图片处理</li>
            <li>实时摄像头检测</li>
            <li>病虫害防治建议生成</li>
            <li>历史记录管理</li>
            <li>多种导出格式</li>
        </ul>
        <p><b>技术支持:</b> AI农业解决方案团队</p>
        """
        QMessageBox.about(self, "关于", about_text)

    # ---------- 生命周期 ----------
    def closeEvent(self, event) -> None:
        try:
            self._metrics_timer.stop()
            self.realtime_tab.shutdown()
            self.detect_tab.shutdown()
            self.batch_tab.shutdown()
            self.container.tts.stop()
        except Exception:  # noqa: BLE001
            logger.exception("关闭应用时发生错误")
        event.accept()
