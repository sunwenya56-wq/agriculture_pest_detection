"""存储与批量服务测试：落库、统计、缓存 TTL、失败隔离。"""
from __future__ import annotations

from app.adapters.storage import SqliteRepository
from app.core.dto import BBox, Detection, DetectionResult
from app.services.batch import BatchService
from app.services.detection import DetectionService


class FakeDetector:
    def __init__(self, fail_on: set[str] | None = None):
        self.fail_on = fail_on or set()
        self.calls = 0

    def update_confidence(self, value: float) -> None:
        pass

    def switch_model(self, path: str) -> None:
        pass

    def detect(self, image_path: str) -> DetectionResult:
        self.calls += 1
        if image_path in self.fail_on:
            from app.core.errors import DetectionError

            raise DetectionError("模拟推理失败")
        return DetectionResult(
            image_path=image_path,
            detections=[Detection(5, "地老虎", 0.88, BBox(1, 1, 50, 50))],
            elapsed_ms=12.3,
            model_name="best.pt",
        )

    def detect_frame(self, frame: object) -> DetectionResult:
        return DetectionResult(image_path="<camera>")


def test_repository_persists_and_stats(tmp_path):
    repo = SqliteRepository(tmp_path / "history.db")
    result = DetectionResult(
        image_path="img.jpg",
        detections=[Detection(0, "地老虎", 0.9, BBox(0, 0, 1, 1))],
        model_name="best.pt",
        elapsed_ms=10,
    )
    record_id = repo.add_detection(result, "建议文本", "best.pt")
    assert record_id > 0

    rows = repo.list_detections()
    assert rows[0]["pests"] == ["地老虎"]
    assert rows[0]["advice"] == "建议文本"

    stats = repo.statistics()
    assert stats["total_detections"] == 1
    assert stats["pest_distribution"]["地老虎"] == 1


def test_repository_cache_ttl(tmp_path):
    repo = SqliteRepository(tmp_path / "history.db")
    repo.cache_set("k", "text", "cloud", "model", "v1")
    assert repo.cache_get("k", ttl_s=3600)["text"] == "text"
    assert repo.cache_get("k", ttl_s=0)["text"] == "text"


def test_repository_migrates_legacy_schema(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                detection_time TEXT NOT NULL,
                pests_detected TEXT NOT NULL,
                confidence_scores TEXT NOT NULL,
                advice_generated TEXT,
                model_used TEXT,
                export_path TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO detection_history (image_path, detection_time, pests_detected,"
            " confidence_scores, advice_generated, model_used) VALUES ('a.jpg', '2026-01-01 10:00:00',"
            " '[\"地老虎\"]', '[0.9]', '旧建议', 'best.pt')"
        )

    repo = SqliteRepository(db_path)
    rows = repo.list_detections()
    assert rows[0]["pests"] == ["地老虎"]
    assert rows[0]["elapsed_ms"] == 0
    assert rows[0]["degraded"] is False



def test_batch_isolates_single_failure(tmp_path, metrics):
    repo = SqliteRepository(tmp_path / "history.db")
    detection = DetectionService(FakeDetector(fail_on={str(tmp_path / "b.jpg")}), repo, metrics)
    service = BatchService(detection, workers=1)

    paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        path = tmp_path / name
        path.write_bytes(b"fake")
        paths.append(str(path))

    progress: list[tuple[int, int]] = []
    result = service.run(paths, progress_cb=lambda i, t, _: progress.append((i, t)))

    assert result.total == 3
    assert result.succeeded == 2
    assert result.items[1].error == "模拟推理失败"
    assert result.pest_count == 2
    assert progress[-1] == (3, 3)
