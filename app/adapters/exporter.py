"""导出适配器：JSON / CSV / XLSX / 标注图，批量报告。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.dto import BatchResult, DetectionResult
from app.core.errors import ExportError
from app.core.logging import get_logger

logger = get_logger(__name__)


class Exporter:
    def __init__(self, export_dir: str | Path):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _target(self, fmt: str, prefix: str) -> Path:
        stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.export_dir / fmt
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{prefix}_{stamped}.{fmt}"

    def export_json(self, result: DetectionResult, advice: str = "", path: Path | None = None) -> Path:
        target = path or self._target("json", "result")
        try:
            payload = result.to_dict()
            payload["advice"] = advice
            payload["export_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            import json

            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return target
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"导出 JSON 失败: {exc}") from exc

    def export_csv(self, result: DetectionResult, path: Path | None = None) -> Path:
        target = path or self._target("csv", "result")
        try:
            import pandas as pd

            rows = [
                {
                    "序号": i + 1,
                    "害虫名称": det.label,
                    "置信度": round(det.confidence, 4),
                    "X1": round(det.bbox.x1, 2),
                    "Y1": round(det.bbox.y1, 2),
                    "X2": round(det.bbox.x2, 2),
                    "Y2": round(det.bbox.y2, 2),
                    "检测时间": result.created_at,
                }
                for i, det in enumerate(result.detections)
            ] or [{"序号": 1, "害虫名称": "未检测到害虫", "置信度": 0, "X1": 0, "Y1": 0, "X2": 0, "Y2": 0,
                   "检测时间": result.created_at}]
            pd.DataFrame(rows).to_csv(target, index=False, encoding="utf-8-sig")
            return target
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"导出 CSV 失败: {exc}") from exc

    def export_xlsx(self, result: DetectionResult, advice: str = "", path: Path | None = None) -> Path:
        target = path or self._target("xlsx", "result")
        try:
            import pandas as pd

            detections = [
                {
                    "序号": i + 1,
                    "害虫名称": det.label,
                    "置信度": round(det.confidence, 4),
                    "X1": round(det.bbox.x1, 2),
                    "Y1": round(det.bbox.y1, 2),
                    "X2": round(det.bbox.x2, 2),
                    "Y2": round(det.bbox.y2, 2),
                }
                for i, det in enumerate(result.detections)
            ] or [{"序号": 1, "害虫名称": "未检测到害虫", "置信度": 0, "X1": 0, "Y1": 0, "X2": 0, "Y2": 0}]

            with pd.ExcelWriter(target, engine="openpyxl") as writer:
                pd.DataFrame(detections).to_excel(writer, sheet_name="检测结果", index=False)
                pd.DataFrame(
                    [
                        {
                            "检测时间": result.created_at,
                            "图片": result.image_path,
                            "检测数量": result.count,
                            "最高置信度": round(result.max_confidence, 4),
                            "推理耗时(ms)": result.elapsed_ms,
                            "模型": result.model_name,
                        }
                    ]
                ).to_excel(writer, sheet_name="统计信息", index=False)
                if advice:
                    pd.DataFrame([{"防治建议": advice}]).to_excel(writer, sheet_name="防治建议", index=False)
            return target
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"导出 Excel 失败: {exc}") from exc

    def export_annotated_image(self, result: DetectionResult, path: Path | None = None) -> Path:
        target = path or self._target("jpg", "detected")
        try:
            from PIL import Image, ImageDraw, ImageFont

            with Image.open(result.image_path) as img:
                image = img.convert("RGB")
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("msyh.ttc", 20)
            except Exception:  # noqa: BLE001
                font = ImageFont.load_default()
            for det in result.detections:
                box = (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
                draw.rectangle(box, outline=(231, 76, 60), width=3)
                label = f"{det.label} {det.confidence:.2f}"
                text_box = draw.textbbox((det.bbox.x1, det.bbox.y1), label, font=font)
                draw.rectangle(text_box, fill=(231, 76, 60))
                draw.text((det.bbox.x1, det.bbox.y1), label, fill="white", font=font)
            image.save(target, quality=92)
            return target
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"导出标注图失败: {exc}") from exc

    def export_batch(self, batch: BatchResult, path: Path | None = None) -> Path:
        target = path or self._target("json", "batch")
        try:
            import json

            target.write_text(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return target
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"批量导出失败: {exc}") from exc

    def export_batch_csv(self, batch: BatchResult, path: Path | None = None) -> Path:
        target = path or self._target("csv", "batch")
        try:
            import pandas as pd

            rows: list[dict[str, Any]] = []
            for item in batch.items:
                if not item.result:
                    continue
                for det in item.result.detections:
                    rows.append(
                        {
                            "图片": item.image_path,
                            "害虫名称": det.label,
                            "置信度": round(det.confidence, 4),
                            "检测时间": item.result.created_at,
                        }
                    )
            pd.DataFrame(rows or [{"图片": "", "害虫名称": "未检测到害虫", "置信度": 0, "检测时间": ""}]).to_csv(
                target, index=False, encoding="utf-8-sig"
            )
            return target
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"批量导出 CSV 失败: {exc}") from exc
