"""批量检测服务：进度回调、单文件失败隔离、分布统计。"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.dto import BatchItem, BatchResult
from app.core.errors import DetectionError
from app.core.logging import get_logger
from app.services.detection import DetectionService

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file())


class BatchService:
    def __init__(self, detection: DetectionService, *, workers: int = 1):
        self.detection = detection
        self.workers = max(1, workers)
        self._detector_lock = threading.Lock()

    def run(
        self,
        image_paths: Sequence[str | Path],
        progress_cb: Callable[[int, int, BatchItem], None] | None = None,
    ) -> BatchResult:
        started = time.perf_counter()
        total = len(image_paths)
        result = BatchResult()

        def process(index_path: tuple[int, str | Path]) -> BatchItem:
            index, path = index_path
            try:
                # ultralytics 模型对象非严格线程安全，串行化推理调用
                with self._detector_lock:
                    detection = self.detection.detect_image(path)
                return BatchItem(image_path=str(path), result=detection)
            except DetectionError as exc:
                logger.warning("批量单文件失败", extra={"ctx_image": str(path), "ctx_error": str(exc)})
                return BatchItem(image_path=str(path), error=str(exc))
            except Exception as exc:  # noqa: BLE001
                logger.exception("批量单文件未知异常", extra={"ctx_image": str(path)})
                return BatchItem(image_path=str(path), error=str(exc))

        if self.workers == 1 or total <= 1:
            for index, path in enumerate(image_paths, start=1):
                item = process((index, path))
                result.items.append(item)
                if progress_cb:
                    progress_cb(index, total, item)
        else:
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="batch") as pool:
                futures = [pool.submit(process, (i, p)) for i, p in enumerate(image_paths, start=1)]
                for index, future in enumerate(futures, start=1):
                    item = future.result()
                    result.items.append(item)
                    if progress_cb:
                        progress_cb(index, total, item)

        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("批量检测完成", extra={"ctx_total": total, "ctx_succeeded": result.succeeded})
        return result

    def summarize(self, result: BatchResult) -> str:
        distribution = result.pest_distribution()
        lines = [
            f"共处理 {result.total} 张图片，成功 {result.succeeded} 张，失败 {result.total - result.succeeded} 张",
            f"检出害虫 {result.pest_count} 个，涉及 {len(distribution)} 类",
        ]
        if distribution:
            lines.append("分布：" + "、".join(f"{name} {count} 个" for name, count in distribution.items()))
        return "\n".join(lines)
