"""历史服务：查询、删除、统计与导出。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.storage import SqliteRepository
from app.core.errors import ExportError
from app.core.logging import get_logger

logger = get_logger(__name__)


class HistoryService:
    def __init__(self, repo: SqliteRepository):
        self.repo = repo

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.repo.list_detections(limit)

    def get(self, record_id: int) -> dict[str, Any] | None:
        return self.repo.get_detection(record_id)

    def delete(self, record_id: int) -> bool:
        return self.repo.delete_detection(record_id)

    def clear(self) -> None:
        self.repo.clear_detections()

    def statistics(self) -> dict[str, Any]:
        return self.repo.statistics()

    def export(self, file_path: str | Path, fmt: str = "csv") -> Path:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.repo.list_detections(limit=1000)
        try:
            import pandas as pd

            data = [
                {
                    "ID": row["id"],
                    "图片路径": row["image_path"],
                    "检测时间": row["detection_time"],
                    "检测到的害虫": "、".join(row["pests"]),
                    "平均置信度": round(sum(row["confidences"]) / len(row["confidences"]), 4)
                    if row["confidences"]
                    else 0,
                    "使用模型": row["model_used"],
                    "防治建议": (row["advice"][:100] + "...") if len(row["advice"]) > 100 else row["advice"],
                }
                for row in rows
            ]
            df = pd.DataFrame(data)
            if fmt == "xlsx":
                df.to_excel(target, index=False, engine="openpyxl")
            else:
                df.to_csv(target, index=False, encoding="utf-8-sig")
            return target
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"导出历史记录失败: {exc}") from exc
