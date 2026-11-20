"""LLM 适配器：云端 SiliconFlow 与本地 Qwen+LoRA adapter 双通道，统一生成接口。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from app.core.config import Settings
from app.core.errors import ConfigError, LLMError, LLMTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    source: str
    tokens: int = 0
    elapsed_ms: float = 0.0


class LLMClient(Protocol):
    model: str
    source: str

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...


class SiliconFlowClient:
    """云端 OpenAI 兼容接口，带指数退避重试。"""

    source = "cloud"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 30.0,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        retries: int = 3,
    ):
        if not api_key:
            raise ConfigError("缺少 AGRI_QWEN_API_KEY，请在 .env 中配置")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = max(0, retries)
        self._session = requests.Session()

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise LLMError("云端返回空结果")
                    usage = data.get("usage") or {}
                    return LLMResponse(
                        text=choices[0]["message"]["content"],
                        model=self.model,
                        source=self.source,
                        tokens=int(usage.get("total_tokens", 0)),
                        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = LLMError(f"云端限流或服务异常: {response.status_code}")
                else:
                    raise LLMError(f"云端请求失败: {response.status_code} {response.text[:200]}")
            except requests.exceptions.Timeout as exc:
                last_error = LLMTimeoutError(f"云端请求超时（{self.timeout_s}s）")
                last_error.__cause__ = exc
            except requests.exceptions.RequestException as exc:
                last_error = LLMError(f"网络异常: {exc}")

            if attempt < self.retries:
                backoff = 0.5 * (2**attempt)
                logger.warning(
                    "LLM 请求重试",
                    extra={"ctx_attempt": attempt + 1, "ctx_backoff_s": backoff, "ctx_error": str(last_error)},
                )
                time.sleep(backoff)

        assert last_error is not None
        raise last_error

    def healthcheck(self) -> bool:
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=min(self.timeout_s, 5),
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class LocalQwenClient:
    """本地 Qwen2.5-7B + LoRA adapter 推理，缺少依赖时明确报错，由上层降级。"""

    source = "local"

    def __init__(
        self,
        model_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
        device: str = "auto",
        max_new_tokens: int = 1000,
        temperature: float = 0.7,
    ):
        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = self.model_path.name
        self._tokenizer: Any = None
        self._net: Any = None

    def _ensure_loaded(self) -> None:
        if self._net is not None:
            return
        if not self.model_path.exists():
            raise ConfigError(f"本地模型目录不存在: {self.model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
            kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch.float16}
            if self.device != "auto":
                kwargs["device_map"] = self.device
            else:
                kwargs["device_map"] = "auto"
            self._net = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)

            if self.adapter_path and self.adapter_path.exists():
                from peft import PeftModel

                self._net = PeftModel.from_pretrained(self._net, str(self.adapter_path))
                logger.info("已加载 LoRA adapter", extra={"ctx_adapter": str(self.adapter_path)})
        except ImportError as exc:
            raise ConfigError("本地推理缺少依赖，请安装 requirements-train.txt") from exc
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"本地模型加载失败: {exc}") from exc

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        self._ensure_loaded()
        started = time.perf_counter()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer([text], return_tensors="pt").to(self._net.device)
            output = self._net.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
            )
            generated = output[0][inputs["input_ids"].shape[1] :]
            content = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"本地生成失败: {exc}") from exc

        return LLMResponse(
            text=content,
            model=self.model,
            source=self.source,
            tokens=int(generated.shape[0]),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def healthcheck(self) -> bool:
        return self.model_path.exists()


def build_llm_client(settings: Settings) -> LLMClient | None:
    """按配置选择通道；不可用时返回 None，由建议服务走规则兜底。"""
    backend = settings.effective_llm_backend
    if backend == "local":
        if settings.local_model_path.exists():
            return LocalQwenClient(
                settings.local_model_path,
                adapter_path=settings.local_adapter_path,
                device=settings.local_device,
                max_new_tokens=settings.qwen_max_tokens,
                temperature=settings.qwen_temperature,
            )
        if settings.has_cloud_key:
            logger.warning("本地模型缺失，回落到云端通道")
            backend = "cloud"
        else:
            logger.warning("本地模型与云端 Key 均不可用，将使用规则兜底")
            return None

    if backend == "cloud":
        if not settings.has_cloud_key:
            logger.warning("缺少云端 Key，将使用规则兜底")
            return None
        return SiliconFlowClient(
            settings.qwen_api_key,
            settings.qwen_base_url,
            settings.qwen_model,
            timeout_s=settings.qwen_timeout_s,
            max_tokens=settings.qwen_max_tokens,
            temperature=settings.qwen_temperature,
            retries=settings.qwen_retries,
        )
    return None
