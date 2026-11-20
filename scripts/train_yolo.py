"""YOLOv11 训练脚本：可复现数据、超参、验证流程。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLOv11 病虫害检测模型")
    parser.add_argument("--data", required=True, help="数据集 yaml 路径，例如 ultralytics-main/mydata.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练权重或模型结构")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="显存不足时改为 8 或 4")
    parser.add_argument("--device", default="auto", help="auto / 0 / cpu")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"数据集配置不存在: {data_path}", file=sys.stderr)
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print("缺少 ultralytics，请先执行 pip install -r requirements.txt", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=None if args.device == "auto" else args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        exist_ok=True,
        plots=True,
    )

    metrics = getattr(results, "results_dict", {}) or {}
    print("训练完成，关键指标：")
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    print(f"产物目录: {Path(args.project) / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
