"""建议服务：模板化 Prompt + 缓存 + 云端/本地双通道 + 规则兜底。"""
from __future__ import annotations

import hashlib

from app.adapters.agri_kb import fallback_advice, normalize_name
from app.adapters.llm import LLMClient
from app.adapters.storage import SqliteRepository
from app.core.dto import AdviceResult
from app.core.errors import AppError
from app.core.logging import get_logger, new_trace_id
from app.core.metrics import MetricsRecorder

logger = get_logger(__name__)

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "你是一位农业植保专家，熟悉中国主要农作物病虫害的综合防治（IPM）。"
    "回答必须科学、可操作、环保，避免使用未登记的农药。"
)

PROMPT_TEMPLATE = """检测到的病虫害：{pests}

请按以下结构给出防治方案，语言精炼、可直接执行：
1. 农业防治措施
2. 物理防治方法
3. 生物防治手段
4. 化学防治方案（药剂通用名、用量区间、施药时机与安全间隔期）
5. 预防与监测建议

要求：
- 优先非化学措施；化学防治必须给出轮换用药建议
- 不确定的剂量用区间表述，不要编造具体品牌
"""


def build_prompt(pest_names: list[str]) -> str:
    normalized = [normalize_name(p) for p in pest_names]
    return PROMPT_TEMPLATE.format(pests="、".join(normalized))


def cache_key_for(pest_names: list[str], prompt_version: str = PROMPT_VERSION) -> str:
    normalized = sorted({normalize_name(p) for p in pest_names})
    raw = f"{prompt_version}|{'|'.join(normalized)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class AdviceService:
    def __init__(
        self,
        llm: LLMClient | None,
        repo: SqliteRepository,
        metrics: MetricsRecorder,
        *,
        cache_ttl_s: int = 86400,
        prompt_version: str = PROMPT_VERSION,
    ):
        self.llm = llm
        self.repo = repo
        self.metrics = metrics
        self.cache_ttl_s = cache_ttl_s
        self.prompt_version = prompt_version

    @property
    def backend(self) -> str:
        return getattr(self.llm, "source", "fallback") if self.llm else "fallback"

    def generate(self, pest_names: list[str], *, use_cache: bool = True) -> AdviceResult:
        clean = [p for p in pest_names if str(p).strip()]
        if not clean:
            return AdviceResult(text="未检测到害虫，无需防治。", source="fallback", prompt_version=self.prompt_version)

        key = cache_key_for(clean, self.prompt_version)
        if use_cache:
            cached = self.repo.cache_get(key, self.cache_ttl_s)
            if cached:
                self.metrics.record("advice", ok=True, degraded=False, source="cache", cached=True, elapsed_ms=0)
                return AdviceResult(
                    text=cached["text"],
                    source="cache",
                    model=cached["model"] or "",
                    cached=True,
                    prompt_version=cached["prompt_version"] or self.prompt_version,
                )

        new_trace_id("adv-")
        prompt = build_prompt(clean)

        if self.llm is not None:
            try:
                response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
                result = AdviceResult(
                    text=response.text,
                    source=response.source,
                    model=response.model,
                    elapsed_ms=response.elapsed_ms,
                    prompt_version=self.prompt_version,
                )
                self._store(key, result)
                self.metrics.record(
                    "advice", ok=True, degraded=False, source=result.source,
                    elapsed_ms=result.elapsed_ms, tokens=response.tokens,
                )
                return result
            except AppError as exc:
                logger.warning("LLM 不可用，走规则兜底", extra={"ctx_error": str(exc)})
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM 调用异常，走规则兜底")
                error = str(exc)
        else:
            error = "未配置可用的大模型通道"

        result = AdviceResult(
            text=fallback_advice(clean),
            source="fallback",
            model="rule-based",
            degraded=True,
            error=error,
            prompt_version=self.prompt_version,
        )
        # 兜底结果只缓存短周期也无意义，直接不写缓存，便于恢复后即时切换
        self.metrics.record("advice", ok=True, degraded=True, source="fallback", elapsed_ms=0)
        return result

    def _store(self, key: str, result: AdviceResult) -> None:
        if not result.text.strip():
            return
        try:
            self.repo.cache_set(key, result.text, result.source, result.model, result.prompt_version)
        except Exception:  # noqa: BLE001
            logger.warning("建议缓存写入失败，不影响主流程", exc_info=True)
