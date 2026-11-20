"""建议服务测试：缓存命中、LLM 失败兜底、规则知识库。"""
from __future__ import annotations

from app.adapters.agri_kb import fallback_advice, normalize_name
from app.adapters.llm import LLMResponse
from app.adapters.storage import SqliteRepository
from app.core.errors import LLMTimeoutError
from app.services.advice import AdviceService, build_prompt, cache_key_for


class FakeLLM:
    source = "cloud"
    model = "fake-qwen"

    def __init__(self, text: str = "云端防治方案", *, fail: Exception | None = None):
        self.text = text
        self.fail = fail
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        self.calls += 1
        if self.fail:
            raise self.fail
        return LLMResponse(text=self.text, model=self.model, source=self.source, tokens=123, elapsed_ms=45.6)


def test_build_prompt_normalizes_english_labels():
    prompt = build_prompt(["cutworm", "field_cricket"])
    assert "地老虎" in prompt
    assert "田蟋蟀" in prompt


def test_cache_key_is_order_insensitive():
    assert cache_key_for(["地老虎", "田蟋蟀"]) == cache_key_for(["田蟋蟀", "地老虎"])


def test_advice_uses_llm_then_cache(tmp_path, metrics):
    repo = SqliteRepository(tmp_path / "db.sqlite")
    llm = FakeLLM()
    service = AdviceService(llm, repo, metrics, cache_ttl_s=3600)

    first = service.generate(["地老虎"])
    assert first.source == "cloud"
    assert first.text == "云端防治方案"
    assert llm.calls == 1

    second = service.generate(["地老虎"])
    assert second.cached is True
    assert second.source == "cache"
    assert llm.calls == 1


def test_advice_falls_back_when_llm_times_out(tmp_path, metrics):
    repo = SqliteRepository(tmp_path / "db.sqlite")
    service = AdviceService(FakeLLM(fail=LLMTimeoutError("timeout")), repo, metrics)

    result = service.generate(["地老虎"])
    assert result.source == "fallback"
    assert result.degraded is True
    assert "农业防治" in result.text


def test_advice_falls_back_without_client(tmp_path, metrics):
    repo = SqliteRepository(tmp_path / "db.sqlite")
    service = AdviceService(None, repo, metrics)
    result = service.generate(["cutworm"])
    assert "地老虎" in result.text
    assert result.degraded is True


def test_fallback_knowledge_covers_all_training_classes():
    names = [
        "Aug_Pericallia_ricini",
        "field_cricket",
        "Aug_Spodoptera_exigua",
        "Aug_Microtermes_obesi",
        "Aug_Tetranychus_bioculatus",
        "cutworm",
        "Aug_Nupserha_bicolor",
        "Jute_stem_weevil",
        "Aug_Scopula_emissaria",
        "Aug_Luperomorpha_vittata",
        "Aug_Ferisia_pseudococcus",
        "Aug_Earias_cupreoviridis",
    ]
    for name in names:
        assert normalize_name(name) in fallback_advice([name])
    assert "通用建议" in fallback_advice(names)
