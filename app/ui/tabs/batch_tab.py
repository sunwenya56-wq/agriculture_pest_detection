"""批量检测标签页：文件夹/多选、进度、失败项、分布统计、报告导出。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.dto import AdviceResult, BatchItem, BatchResult
from app.core.logging import get_logger
from app.services.batch import list_images
from app.ui.workers import AdviceWorker, BatchWorker, ExportWorker

if TYPE_CHECKING:
    from app.container import Container

logger = get_logger(__name__)


class BatchTab(QWidget):
    status_message = pyqtSignal(str)
    metrics_refresh = pyqtSignal(str)

    def __init__(self, container: Container, parent: QWidget | None = None):
        super().__init__(parent)
        self.container = container
        self.batch_image_paths: list[str] = []
        self.batch_result: BatchResult | None = None
        self._summary_text = ""
        self._batch_worker: BatchWorker | None = None
        self._advice_worker: AdviceWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        control_group = QGroupBox("批量处理控制")
        control_layout = QVBoxLayout(control_group)
        button_row = QHBoxLayout()
        self.select_folder_btn = QPushButton("📁 选择图片文件夹")
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.select_files_btn = QPushButton("🖼️ 多选图片")
        self.select_files_btn.clicked.connect(self.select_files)
        self.start_btn = QPushButton("▶️ 开始批量处理")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_batch)
        self.export_btn = QPushButton("💾 批量导出")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_batch)
        for button in (self.select_folder_btn, self.select_files_btn, self.start_btn, self.export_btn):
            button_row.addWidget(button)

        list_row = QHBoxLayout()
        file_group = QGroupBox("待处理文件")
        file_layout = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(200)
        file_layout.addWidget(self.file_list)

        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("等待开始...")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        list_row.addWidget(file_group)
        list_row.addWidget(progress_group)

        control_layout.addLayout(button_row)
        control_layout.addLayout(list_row)

        result_group = QGroupBox("批量检测结果")
        result_layout = QVBoxLayout(result_group)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["序号", "文件名", "检测结果", "害虫数量", "处理状态", "错误信息"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        result_layout.addWidget(self.table)

        left_layout.addWidget(control_group)
        left_layout.addWidget(result_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        stats_group = QGroupBox("批量检测统计")
        stats_layout = QGridLayout(stats_group)
        self.total_label = QLabel("文件总数: 0")
        self.processed_label = QLabel("已处理: 0")
        self.success_label = QLabel("成功: 0")
        self.failed_label = QLabel("失败: 0")
        self.pest_total_label = QLabel("害虫总数: 0")
        stats_layout.addWidget(self.total_label, 0, 0)
        stats_layout.addWidget(self.processed_label, 0, 1)
        stats_layout.addWidget(self.success_label, 1, 0)
        stats_layout.addWidget(self.failed_label, 1, 1)
        stats_layout.addWidget(self.pest_total_label, 2, 0, 1, 2)

        dist_group = QGroupBox("害虫分布")
        dist_layout = QVBoxLayout(dist_group)
        self.distribution_text = QTextEdit()
        self.distribution_text.setReadOnly(True)
        self.distribution_text.setMaximumHeight(150)
        dist_layout.addWidget(self.distribution_text)

        advice_group = QGroupBox("防治建议汇总")
        advice_layout = QVBoxLayout(advice_group)
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setPlaceholderText("批量处理完成后，将显示综合防治建议...")
        advice_layout.addWidget(self.advice_text)

        action_group = QGroupBox("数据分析")
        action_layout = QHBoxLayout(action_group)
        self.analyze_btn = QPushButton("📊 分析数据")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_data)
        self.report_btn = QPushButton("📋 生成报告")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self.generate_report)
        action_layout.addWidget(self.analyze_btn)
        action_layout.addWidget(self.report_btn)

        right_layout.addWidget(stats_group)
        right_layout.addWidget(dist_group)
        right_layout.addWidget(advice_group)
        right_layout.addWidget(action_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 500])
        layout.addWidget(splitter)

    # ---------- 文件选择 ----------
    def select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        paths = [str(path) for path in list_images(folder)]
        self._set_paths(paths)
        if not paths:
            self.progress_label.setText("未找到图片文件")
            QMessageBox.information(self, "提示", "该文件夹内未找到图片文件")

    def select_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "多选图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not files:
            return
        merged = list(dict.fromkeys([*self.batch_image_paths, *files]))
        self._set_paths(merged)

    def _set_paths(self, paths: list[str]) -> None:
        self.batch_image_paths = paths
        self.file_list.clear()
        for path in paths:
            self.file_list.addItem(Path(path).name)
        self.start_btn.setEnabled(bool(paths))
        self.export_btn.setEnabled(False)
        self.report_btn.setEnabled(False)
        self.progress_label.setText(
            f"找到 {len(paths)} 个图片文件" if paths else "未找到图片文件"
        )

    # ---------- 批量执行 ----------
    def start_batch(self) -> None:
        if not self.batch_image_paths:
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return

        self.batch_result = None
        self._summary_text = ""
        self.table.setRowCount(len(self.batch_image_paths))
        for row, path in enumerate(self.batch_image_paths):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(Path(path).name))
            self.table.setItem(row, 2, QTableWidgetItem("-"))
            self.table.setItem(row, 3, QTableWidgetItem("-"))
            self.table.setItem(row, 4, QTableWidgetItem("等待中"))
            self.table.setItem(row, 5, QTableWidgetItem(""))
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_label.setText("正在处理中...")
        self.start_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.select_files_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.report_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.advice_text.clear()
        self.distribution_text.clear()
        self._reset_statistics(len(self.batch_image_paths))

        worker = BatchWorker(self.container, self.batch_image_paths, self)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_batch_finished)
        worker.failed.connect(self._on_batch_failed)
        worker.finished.connect(self._restore_batch_controls)
        self._batch_worker = worker
        worker.start()
        self.status_message.emit("批量检测已开始")

    def _restore_batch_controls(self) -> None:
        self.select_folder_btn.setEnabled(True)
        self.select_files_btn.setEnabled(True)
        self.start_btn.setEnabled(bool(self.batch_image_paths))

    def _on_progress(self, current: int, total: int, item: BatchItem) -> None:
        if total <= 0:
            return
        self.progress_bar.setValue(int(current / total * 100))
        self.progress_label.setText(f"正在处理: {current}/{total}")
        row = current - 1
        if row < 0 or row >= self.table.rowCount():
            return

        if item.ok and item.result is not None:
            labels = item.result.labels
            pests = "、".join(labels[:3])
            if len(labels) > 3:
                pests += f"...等{len(labels)}种"
            self.table.setItem(row, 2, QTableWidgetItem(pests or "无害虫"))
            self.table.setItem(row, 3, QTableWidgetItem(str(item.result.count)))
            self.table.setItem(row, 4, QTableWidgetItem("✓ 成功"))
            self.table.setItem(row, 5, QTableWidgetItem(""))
        else:
            self.table.setItem(row, 2, QTableWidgetItem("处理失败"))
            self.table.setItem(row, 3, QTableWidgetItem("-"))
            failure = QTableWidgetItem("✗ 失败")
            failure.setForeground(QBrush(QColor("#e74c3c")))
            self.table.setItem(row, 4, failure)
            self.table.setItem(row, 5, QTableWidgetItem(item.error or "未知错误"))

    def _on_batch_finished(self, batch: BatchResult) -> None:
        self.batch_result = batch
        self.progress_bar.setValue(100)
        self.progress_label.setText("批量处理完成")
        self.export_btn.setEnabled(True)
        self.report_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)

        self.total_label.setText(f"文件总数: {batch.total}")
        self.processed_label.setText(f"已处理: {batch.total}")
        self.success_label.setText(f"成功: {batch.succeeded}")
        self.failed_label.setText(f"失败: {batch.total - batch.succeeded}")
        self.pest_total_label.setText(f"害虫总数: {batch.pest_count}")

        distribution = batch.pest_distribution()
        if distribution:
            lines = ["害虫分布统计:"]
            for name, count in list(distribution.items())[:10]:
                lines.append(f"  {name}: {count}次")
            self.distribution_text.setPlainText("\n".join(lines))
        else:
            self.distribution_text.setPlainText("未检测到害虫")

        self._summary_text = self.container.batch.summarize(batch)
        self.advice_text.setPlainText(self._summary_text)
        self.status_message.emit(
            f"批量检测完成：成功 {batch.succeeded}/{batch.total}，耗时 {batch.elapsed_ms:.0f} ms"
        )
        self.metrics_refresh.emit("批量检测完成")
        self._start_advice()

    def _on_batch_failed(self, message: str) -> None:
        self.progress_label.setText("批量处理失败")
        self.status_message.emit("批量处理失败")
        QMessageBox.critical(self, "批量处理错误", message)

    def _reset_statistics(self, total: int) -> None:
        self.total_label.setText(f"文件总数: {total}")
        self.processed_label.setText("已处理: 0")
        self.success_label.setText("成功: 0")
        self.failed_label.setText("失败: 0")
        self.pest_total_label.setText("害虫总数: 0")

    # ---------- 分析与报告 ----------
    def _unique_labels(self) -> list[str]:
        labels: list[str] = []
        if self.batch_result is None:
            return labels
        for item in self.batch_result.items:
            if not item.result:
                continue
            for label in item.result.labels:
                if label not in labels:
                    labels.append(label)
        return labels

    def _start_advice(self) -> None:
        if self.batch_result is None:
            return
        labels = self._unique_labels()
        if not labels:
            return
        if self._advice_worker is not None and self._advice_worker.isRunning():
            return
        self.advice_text.setPlainText(self._summary_text + "\n\n正在生成综合防治建议...")
        worker = AdviceWorker(self.container, labels, self)
        worker.finished_ok.connect(self._on_advice_finished)
        worker.failed.connect(self._on_advice_failed)
        self._advice_worker = worker
        worker.start()

    def _on_advice_finished(self, advice: AdviceResult) -> None:
        self.advice_text.setPlainText(
            f"{self._summary_text}\n\n=== 综合防治建议 ===\n\n{advice.text}"
        )
        self.status_message.emit(
            f"综合防治建议已生成（{advice.source}，{advice.elapsed_ms:.0f} ms）"
        )

    def _on_advice_failed(self, message: str) -> None:
        self.advice_text.setPlainText(f"{self._summary_text}\n\n综合防治建议生成失败: {message}")

    def analyze_data(self) -> None:
        if self.batch_result is None:
            return
        self._start_advice()
        if self._summary_text:
            self.status_message.emit("正在分析批量数据...")

    def export_batch(self) -> None:
        if self.batch_result is None:
            return
        formats = ["JSON", "CSV"]
        choice, ok = QInputDialog.getItem(self, "批量导出", "请选择导出格式:", formats, 0, False)
        if not ok:
            return
        batch = self.batch_result
        exporter = self.container.exporter
        action = (
            (lambda: exporter.export_batch(batch))
            if choice == "JSON"
            else (lambda: exporter.export_batch_csv(batch))
        )
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        worker = ExportWorker(action, self)
        worker.finished_ok.connect(lambda path: QMessageBox.information(self, "成功", f"批量导出成功:\n{path}"))
        worker.failed.connect(lambda message: QMessageBox.critical(self, "错误", f"导出失败: {message}"))
        self._export_worker = worker
        worker.start()
        self.status_message.emit("正在导出批量结果...")

    def generate_report(self) -> None:
        if self.batch_result is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存分析报告", "", "文本文件 (*.txt)")
        if not file_path:
            return
        content = (
            "=== 批量检测分析报告 ===\n\n"
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{self.advice_text.toPlainText() or self._summary_text}\n"
        )
        target = Path(file_path)
        action = lambda: _write_text(target, content)  # noqa: E731
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        worker = ExportWorker(action, self)
        worker.finished_ok.connect(lambda path: QMessageBox.information(self, "成功", f"报告已保存至:\n{path}"))
        worker.failed.connect(lambda message: QMessageBox.critical(self, "错误", f"生成报告失败: {message}"))
        self._export_worker = worker
        worker.start()
        self.status_message.emit("正在生成报告...")

    # ---------- 生命周期 ----------
    def shutdown(self) -> None:
        for worker in (self._batch_worker, self._advice_worker, self._export_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(2000)


def _write_text(target: Path, content: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
