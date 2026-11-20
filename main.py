"""应用入口：高 DPI 初始化、依赖装配、全局异常钩子与主窗口启动。"""
from __future__ import annotations

import os
import sys
import traceback

# Qt6 默认启用高 DPI 缩放；该变量兼容部分平台的缩放行为
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QMessageBox

from app.container import build_container
from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


def install_exception_hook() -> None:
    """全局异常钩子：日志留痕 + 单次弹窗，避免重复弹窗导致递归。"""
    state = {"handling": False}

    def hook(exctype: type[BaseException], value: BaseException, tb: object) -> None:
        trace = "".join(traceback.format_exception(exctype, value, tb))  # type: ignore[arg-type]
        logger.critical("未捕获异常: %s", trace)
        try:
            sys.__excepthook__(exctype, value, tb)
        except Exception:  # noqa: BLE001
            pass

        if state["handling"]:
            return
        state["handling"] = True
        try:
            if QApplication.instance() is not None:
                message = value.message if isinstance(value, AppError) else str(value)
                QMessageBox.critical(None, "程序异常", f"发生未处理的异常:\n{message}")
        except Exception:  # noqa: BLE001
            pass
        finally:
            state["handling"] = False

    sys.excepthook = hook


def main() -> int:
    install_exception_hook()
    sys.setrecursionlimit(10000)

    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except (AttributeError, RuntimeError):
        logger.warning("当前 Qt 版本不支持高 DPI 舍入策略设置")

    app = QApplication(sys.argv)
    app.setApplicationName("智慧农业病虫害识别系统")

    try:
        container = build_container()
        from app.ui.main_window import MainWindow

        window = MainWindow(container)
        window.show()
        return app.exec()
    except Exception as exc:  # noqa: BLE001
        logger.exception("程序启动失败")
        message = exc.message if isinstance(exc, AppError) else str(exc)
        QMessageBox.critical(
            None,
            "启动错误",
            f"程序启动失败: {message}\n\n详细信息已写入日志。",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
