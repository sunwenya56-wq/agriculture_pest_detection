"""统一异常体系：区分错误类型，便于 UI 提示与服务层降级决策。"""
from __future__ import annotations


class AppError(Exception):
    code = "APP_ERROR"
    user_message = "系统异常，请稍后重试"

    def __init__(self, message: str | None = None, *, detail: str | None = None):
        self.message = message or self.user_message
        self.detail = detail
        super().__init__(self.message)

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigError(AppError):
    code = "CONFIG_ERROR"
    user_message = "配置缺失或非法"


class ModelLoadError(AppError):
    code = "MODEL_LOAD_ERROR"
    user_message = "模型加载失败，请检查模型文件"


class DetectionError(AppError):
    code = "DETECTION_ERROR"
    user_message = "图像检测失败"


class LLMError(AppError):
    code = "LLM_ERROR"
    user_message = "智能建议生成失败"


class LLMTimeoutError(LLMError):
    code = "LLM_TIMEOUT"
    user_message = "智能建议生成超时"


class StorageError(AppError):
    code = "STORAGE_ERROR"
    user_message = "数据存取失败"


class ExportError(AppError):
    code = "EXPORT_ERROR"
    user_message = "结果导出失败"


class TTSUnavailable(AppError):
    code = "TTS_UNAVAILABLE"
    user_message = "语音播报不可用"
