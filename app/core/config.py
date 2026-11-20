"""集中式配置：全部通过环境变量 / .env 注入，禁止在代码中硬编码密钥。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGRI_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "dev"
    app_name: str = "智慧农业病虫害识别与防治系统"
    version: str = "1.0.0"

    models_dir: Path = PROJECT_ROOT / "models"
    export_dir: Path = PROJECT_ROOT / "exports"
    log_dir: Path = PROJECT_ROOT / "logs"
    db_path: Path = PROJECT_ROOT / "history.db"
    log_level: str = "INFO"

    # LLM
    llm_backend: Literal["auto", "cloud", "local"] = "auto"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://api.siliconflow.cn/v1"
    qwen_model: str = "Qwen/Qwen2.5-7B-Instruct"
    qwen_timeout_s: float = 30.0
    qwen_max_tokens: int = 1000
    qwen_temperature: float = 0.7
    qwen_retries: int = 3
    local_model_path: Path = PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"
    local_adapter_path: Path = PROJECT_ROOT / "models" / "qwen_agri_lora"
    local_device: str = "auto"
    advice_cache_ttl_s: int = 86400

    # 检测
    device: str = "auto"
    imgsz: int = 640
    confidence: float = 0.25
    iou: float = 0.15
    max_det: int = 50
    detection_interval_s: float = 1.0
    batch_workers: int = 1

    # 语音
    tts_enabled: bool = False
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_rate: str = "+0%"

    @field_validator("models_dir", "export_dir", "log_dir", "db_path", "local_model_path", "local_adapter_path")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"非法日志级别: {value}")
        return level

    @field_validator("confidence")
    @classmethod
    def _check_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence 必须在 [0, 1]")
        return value

    @field_validator("imgsz")
    @classmethod
    def _check_imgsz(cls, value: int) -> int:
        if value < 32:
            raise ValueError("imgsz 必须 >= 32")
        return value

    @property
    def has_cloud_key(self) -> bool:
        return bool(self.qwen_api_key.strip())

    @property
    def effective_llm_backend(self) -> str:
        if self.llm_backend == "auto":
            return "cloud" if self.has_cloud_key else ("local" if self.local_adapter_path.exists() else "cloud")
        return self.llm_backend

    def ensure_dirs(self) -> None:
        for directory in (self.models_dir, self.export_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
