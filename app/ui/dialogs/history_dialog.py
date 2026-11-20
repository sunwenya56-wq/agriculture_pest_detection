"""检测历史对话框：列表、详情、删除单条、清空、导出。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from app.core.logging import get_logger
from app.ui.workers import ExportWorker

if TYPE_CHECKING:
    from app.container import Container

logger = get_logger(__name__)


class HistoryDialog(QDialog):
    def __init__(self, container: Container, parent: Any = None):
        super().__init__(parent)
        self.container = container
        self._records: dict[int, dict[str, Any]] = {}
        self._export_worker: ExportWorker | None = None
        self.setWindowTitle("检测历史记录")
        self.setGeometry(200, 200, 1000, 700)
        self._build_ui()
        self.load_history()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.load_history)
        self.detail_btn = QPushButton("查看详情")
        self.detail_btn.clicked.connect(self.show_detail)
        self.delete_btn = QPushButton("删除选中")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.export_btn = QPushButton("导出历史")
        self.export_btn.clicked.connect(self.export_history)
        self.clear_btn = QPushButton("清空记录")
        self.clear_btn.setObjectName("danger")
        self.clear_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.clear_btn.clicked.connect(self.clear_history)

        for button in (self.refresh_btn, self.detail_btn, self.delete_btn, self.export_btn):
            toolbar.addWidget(button)
        toolbar.addStretch()
        toolbar.addWidget(self.clear_btn)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "图片", "检测时间", "检测到的害虫", "数量", "模型"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda *_: self.show_detail())

        layout.addLayout(toolbar)
        layout.addWidget(self.table)

    # ---------- 数据 ----------
    def load_history(self) -> None:
        try:
            records = self.container.history.list(limit=200)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"加载历史记录失败: {exc}")
            return

        self._records = {int(record["id"]): record for record in records}
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            record_id = int(record.get("id"))
            id_item = QTableWidgetItem(str(record_id))
            id_item.setData(Qt.ItemDataRole.UserRole, record_id)
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, QTableWidgetItem(Path(str(record.get("image_path", ""))).name))
            self.table.setItem(row, 2, QTableWidgetItem(str(record.get("detection_time", ""))))
            pests = record.get("pests", [])
            self.table.setItem(row, 3, QTableWidgetItem("、".join(pests) if pests else "无"))
            self.table.setItem(row, 4, QTableWidgetItem(str(len(pests))))
            self.table.setItem(row, 5, QTableWidgetItem(record.get("model_used") or "默认"))

    def _selected_record(self) -> dict[str, Any] | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        record_id = item.data(Qt.ItemDataRole.UserRole)
        if record_id is None:
            return None
        return self._records.get(int(record_id)) or self.container.history.get(int(record_id))

    # ---------- 操作 ----------
    def show_detail(self) -> None:
        try:
            record = self._selected_record()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"读取历史记录失败: {exc}")
            return
        if not record:
            QMessageBox.information(self, "提示", "请先选择一条历史记录")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"检测记录详情 #{record.get('id')}")
        dialog.setGeometry(300, 300, 700, 600)
        layout = QVBoxLayout(dialog)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setMinimumHeight(240)
        image_path = Path(str(record.get("image_path", "")))
        if image_path.exists():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                image_label.setPixmap(
                    pixmap.scaled(
                        660,
                        240,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        else:
            image_label.setText("原图不存在或已被移动")

        pests = "、".join(record.get("pests", [])) or "无"
        confidences = record.get("confidences", [])
        confidence_text = "、".join(f"{float(value):.2%}" for value in confidences) or "无"
        detail_text = QTextEdit()
        detail_text.setReadOnly(True)
        detail_text.setPlainText(
            f"记录 ID: {record.get('id')}\n"
            f"图片路径: {record.get('image_path')}\n"
            f"检测时间: {record.get('detection_time')}\n"
            f"使用模型: {record.get('model_used') or '默认'}\n"
            f"检测到的害虫: {pests}\n"
            f"置信度: {confidence_text}\n"
            f"推理耗时: {float(record.get('elapsed_ms') or 0):.0f} ms\n"
            f"导出路径: {record.get('export_path') or '无'}\n\n"
            f"=== 防治建议 ===\n{record.get('advice') or '无'}"
        )

        layout.addWidget(image_label)
        layout.addWidget(detail_text)
        dialog.exec()

    def delete_selected(self) -> None:
        try:
            record = self._selected_record()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"读取历史记录失败: {exc}")
            return
        if not record:
            QMessageBox.information(self, "提示", "请先选择一条历史记录")
            return
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除历史记录 #{record.get('id')} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            success = self.container.history.delete(int(record["id"]))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"删除失败: {exc}")
            return
        if success:
            self.load_history()
        else:
            QMessageBox.warning(self, "失败", "删除记录失败")

    def clear_history(self) -> None:
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空所有历史记录吗？此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self.container.history.clear()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"清空失败: {exc}")
            return
        self.load_history()
        QMessageBox.information(self, "成功", "已清空所有历史记录")

    def export_history(self) -> None:
        formats = ["CSV", "Excel"]
        choice, ok = QInputDialog.getItem(self, "导出历史", "请选择导出格式:", formats, 0, False)
        if not ok:
            return
        default_name = "history.csv" if choice == "CSV" else "history.xlsx"
        filter_text = "CSV文件 (*.csv)" if choice == "CSV" else "Excel文件 (*.xlsx)"
        file_path, _ = QFileDialog.getSaveFileName(self, "导出历史记录", default_name, filter_text)
        if not file_path:
            return

        fmt = "csv" if choice == "CSV" else "xlsx"
        target = Path(file_path)
        action = lambda: self.container.history.export(target, fmt)  # noqa: E731
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        worker = ExportWorker(action, self)
        worker.finished_ok.connect(lambda path: QMessageBox.information(self, "成功", f"导出成功:\n{path}"))
        worker.failed.connect(lambda message: QMessageBox.critical(self, "错误", f"导出失败: {message}"))
        self._export_worker = worker
        worker.start()

    def closeEvent(self, event) -> None:
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.wait(2000)
        super().closeEvent(event)
