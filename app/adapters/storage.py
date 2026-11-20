"""SQLite 存储适配器：检测历史、建议缓存、模型使用统计。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from app.core.dto import DetectionResult
from app.core.errors import StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


class SqliteRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    detection_time TEXT NOT NULL,
                    pests_detected TEXT NOT NULL,
                    confidence_scores TEXT NOT NULL,
                    advice_generated TEXT,
                    model_used TEXT,
                    export_path TEXT,
                    elapsed_ms REAL DEFAULT 0,
                    degraded INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS advice_cache (
                    cache_key TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT,
                    model TEXT,
                    prompt_version TEXT,
                    created_ts REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_stats (
                    model_name TEXT PRIMARY KEY,
                    use_count INTEGER DEFAULT 0,
                    last_used TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_history_time ON detection_history(detection_time);
                """
            )
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """兼容旧版本数据库：按列增量补齐，避免老 history.db 直接报错。"""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(detection_history)")}
        additions = {
            "elapsed_ms": "REAL DEFAULT 0",
            "degraded": "INTEGER DEFAULT 0",
        }
        for column, ddl in additions.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE detection_history ADD COLUMN {column} {ddl}")
                logger.info("数据库迁移：新增列", extra={"ctx_column": column})

    # ---------- 检测历史 ----------

    def add_detection(self, result: DetectionResult, advice: str, model_name: str, export_path: str = "") -> int:
        if result.error:
            degraded = 1
        else:
            degraded = 1 if result.degraded else 0
        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO detection_history
                    (image_path, detection_time, pests_detected, confidence_scores,
                     advice_generated, model_used, export_path, elapsed_ms, degraded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.image_path,
                        result.created_at,
                        json.dumps(result.labels, ensure_ascii=False),
                        json.dumps([round(d.confidence, 4) for d in result.detections], ensure_ascii=False),
                        advice,
                        model_name,
                        export_path,
                        result.elapsed_ms,
                        degraded,
                    ),
                )
                if model_name:
                    conn.execute(
                        """
                        INSERT INTO model_stats (model_name, use_count, last_used)
                        VALUES (?, 1, ?)
                        ON CONFLICT(model_name) DO UPDATE SET
                            use_count = use_count + 1,
                            last_used = excluded.last_used
                        """,
                        (model_name, result.created_at),
                    )
                return int(cursor.lastrowid or 0)
        except sqlite3.Error as exc:
            raise StorageError(f"写入检测记录失败: {exc}") from exc

    def list_detections(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM detection_history ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"读取检测记录失败: {exc}") from exc
        return [self._row_to_dict(row) for row in rows]

    def get_detection(self, record_id: int) -> dict[str, Any] | None:
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute("SELECT * FROM detection_history WHERE id = ?", (record_id,)).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"读取检测记录失败: {exc}") from exc
        return self._row_to_dict(row) if row else None

    def delete_detection(self, record_id: int) -> bool:
        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise StorageError(f"删除检测记录失败: {exc}") from exc

    def clear_detections(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM detection_history")
                conn.execute("DELETE FROM model_stats")
        except sqlite3.Error as exc:
            raise StorageError(f"清空检测记录失败: {exc}") from exc

    def statistics(self) -> dict[str, Any]:
        try:
            with self._lock, self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) FROM detection_history").fetchone()[0]
                today = conn.execute(
                    "SELECT COUNT(*) FROM detection_history WHERE DATE(detection_time) = DATE('now','localtime')"
                ).fetchone()[0]
                pest_rows = conn.execute("SELECT pests_detected FROM detection_history").fetchall()
                model_rows = conn.execute(
                    "SELECT model_name, use_count FROM model_stats ORDER BY use_count DESC"
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"统计失败: {exc}") from exc

        distribution: dict[str, int] = {}
        for row in pest_rows:
            for name in json.loads(row[0] or "[]"):
                distribution[name] = distribution.get(name, 0) + 1
        return {
            "total_detections": total,
            "today_detections": today,
            "pest_distribution": dict(sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)),
            "model_stats": [{"model_name": r[0], "use_count": r[1]} for r in model_rows],
        }

    # ---------- 建议缓存 ----------

    def cache_get(self, key: str, ttl_s: int) -> dict[str, Any] | None:
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute("SELECT * FROM advice_cache WHERE cache_key = ?", (key,)).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"读取建议缓存失败: {exc}") from exc
        if not row:
            return None
        if ttl_s > 0 and time.time() - float(row["created_ts"]) > ttl_s:
            return None
        return {"text": row["text"], "source": row["source"], "model": row["model"],
                "prompt_version": row["prompt_version"]}

    def cache_set(self, key: str, text: str, source: str, model: str, prompt_version: str) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO advice_cache (cache_key, text, source, model, prompt_version, created_ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        text = excluded.text, source = excluded.source, model = excluded.model,
                        prompt_version = excluded.prompt_version, created_ts = excluded.created_ts
                    """,
                    (key, text, source, model, prompt_version, time.time()),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"写入建议缓存失败: {exc}") from exc

    def cache_prune(self, ttl_s: int) -> int:
        if ttl_s <= 0:
            return 0
        cutoff = time.time() - ttl_s
        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute("DELETE FROM advice_cache WHERE created_ts < ?", (cutoff,))
                return cursor.rowcount
        except sqlite3.Error:
            return 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "detection_time": row["detection_time"],
            "pests": json.loads(row["pests_detected"] or "[]"),
            "confidences": json.loads(row["confidence_scores"] or "[]"),
            "advice": row["advice_generated"] or "",
            "model_used": row["model_used"] or "",
            "export_path": row["export_path"] or "",
            "elapsed_ms": row["elapsed_ms"] or 0,
            "degraded": bool(row["degraded"]),
        }
