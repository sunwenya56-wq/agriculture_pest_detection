"""界面样式：从旧 ui/styles.py 原样迁移的 QSS。"""
from __future__ import annotations

APP_STYLESHEET = """
/* 主窗口样式 */
QMainWindow {
    background-color: #f5f7fa;
}

/* 按钮样式 */
QPushButton {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #45a049;
}

QPushButton:pressed {
    background-color: #3d8b40;
}

QPushButton:disabled {
    background-color: #cccccc;
    color: #666666;
}

/* 特殊按钮样式 */
QPushButton#danger {
    background-color: #f44336;
}

QPushButton#danger:hover {
    background-color: #d32f2f;
}

QPushButton#warning {
    background-color: #ff9800;
}

QPushButton#warning:hover {
    background-color: #f57c00;
}

/* 标签样式 */
QLabel {
    color: #333333;
    font-size: 13px;
}

QLabel#title {
    font-size: 14px;
    font-weight: bold;
    color: #2c3e50;
}

/* 分组框样式 */
QGroupBox {
    border: 2px solid #3498db;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #2c3e50;
    background-color: rgba(52, 152, 219, 0.05);
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}

/* 文本框样式 */
QTextEdit {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 4px;
    font-size: 12px;
    background-color: white;
    selection-background-color: #3498db;
}

QTextEdit:focus {
    border: 1px solid #3498db;
}

/* 单行文本框样式 */
QLineEdit {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 6px;
    font-size: 12px;
    background-color: white;
}

QLineEdit:focus {
    border: 1px solid #3498db;
}

/* 滑动条样式 */
QSlider::groove:horizontal {
    border: 1px solid #bbb;
    background: white;
    height: 6px;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #3498db;
    border: 1px solid #777;
    height: 6px;
    border-radius: 3px;
}

QSlider::add-page:horizontal {
    background: #e0e0e0;
    border: 1px solid #777;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #4CAF50;
    border: 1px solid #5c5c5c;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #45a049;
}

/* 列表视图样式 */
QListView {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: white;
    padding: 4px;
}

QListView::item {
    padding: 6px;
    border-bottom: 1px solid #ecf0f1;
}

QListView::item:selected {
    background-color: #3498db;
    color: white;
    border-radius: 2px;
}

QListView::item:hover {
    background-color: #f8f9fa;
}

/* 表格样式 */
QTableWidget {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: white;
    gridline-color: #ecf0f1;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #3498db;
    color: white;
}

QHeaderView::section {
    background-color: #2c3e50;
    color: white;
    padding: 6px;
    border: 1px solid #34495e;
    font-weight: bold;
}

/* 标签页样式 */
QTabWidget::pane {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: white;
}

QTabBar::tab {
    background-color: #ecf0f1;
    color: #34495e;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #3498db;
    color: white;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #bdc3c7;
}

/* 进度条样式 */
QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    text-align: center;
    background-color: white;
}

QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 4px;
}

/* 状态栏样式 */
QStatusBar {
    background-color: #34495e;
    color: white;
}

/* 菜单栏样式 */
QMenuBar {
    background-color: #2c3e50;
    color: white;
}

QMenuBar::item:selected {
    background-color: #3498db;
}

QMenu {
    background-color: white;
    border: 1px solid #bdc3c7;
}

QMenu::item:selected {
    background-color: #3498db;
    color: white;
}

/* 分割器样式 */
QSplitter::handle {
    background-color: #bdc3c7;
    margin: 2px;
}

QSplitter::handle:hover {
    background-color: #3498db;
}

/* 滚动条样式 */
QScrollBar:vertical {
    border: none;
    background-color: #f8f9fa;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #bdc3c7;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #95a5a6;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 工具提示样式 */
QToolTip {
    background-color: #2c3e50;
    color: white;
    border: 1px solid #34495e;
    padding: 4px;
    border-radius: 3px;
    font-size: 11px;
}
"""

DARK_STYLESHEET = """
/* 深色主题样式 */
QMainWindow {
    background-color: #2c3e50;
}

QPushButton {
    background-color: #3498db;
    color: white;
}

QPushButton:hover {
    background-color: #2980b9;
}

QTextEdit, QLineEdit, QListView, QTableWidget {
    background-color: #34495e;
    color: #ecf0f1;
    border-color: #4a6572;
}

QGroupBox {
    border-color: #3498db;
    background-color: rgba(52, 152, 219, 0.1);
    color: #ecf0f1;
}
"""
