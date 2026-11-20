"""HTTP API：/health /v1/detect /v1/advice /v1/history /v1/metrics。"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.container import Container, build_container
from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)

_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = build_container()
    return _container


class AdviceRequest(BaseModel):
    pest_names: list[str] = Field(default_factory=list)
    use_cache: bool = True


def create_app(container: Container | None = None) -> FastAPI:
    app = FastAPI(title="智慧农业病虫害识别与防治 API", version="1.0.0")

    def container_or_init() -> Container:
        return container if container is not None else get_container()

    @app.exception_handler(AppError)
    async def _app_error_handler(_, exc: AppError) -> JSONResponse:
        status = 503 if exc.code in {"MODEL_LOAD_ERROR", "LLM_ERROR", "LLM_TIMEOUT"} else 400
        return JSONResponse(status_code=status, content={"error": exc.to_dict()})

    @app.get("/health")
    def health(deep: bool = Query(default=False)) -> dict[str, Any]:
        return container_or_init().health.check(deep=deep).to_dict()

    @app.get("/v1/metrics")
    def metrics() -> dict[str, Any]:
        return container_or_init().metrics.snapshot()

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        c = container_or_init()
        return {
            "path": str(c.model_path),
            "loaded": c.detector.is_loaded,
            "confidence": c.detector.confidence,
            "llm_backend": c.advice.backend,
            "version": c.settings.version,
        }

    @app.post("/v1/detect")
    async def detect(
        file: UploadFile = File(...),
        generate_advice: bool = Query(default=True),
        use_cache: bool = Query(default=True),
    ) -> dict[str, Any]:
        c = container_or_init()
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(await file.read())
            tmp.close()
            result = c.detection.detect_image(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        advice_payload: dict[str, Any] | None = None
        if generate_advice and not result.is_empty:
            advice = c.advice.generate(result.labels, use_cache=use_cache)
            advice_payload = advice.to_dict()
            try:
                c.detection.save_record(result, advice.text, result.model_name)
            except AppError:
                logger.warning("检测记录落库失败", exc_info=True)

        return {"detection": result.to_dict(), "advice": advice_payload}

    @app.post("/v1/advice")
    def advice(request: AdviceRequest) -> dict[str, Any]:
        if not request.pest_names:
            raise HTTPException(status_code=400, detail="pest_names 不能为空")
        return container_or_init().advice.generate(request.pest_names, use_cache=request.use_cache).to_dict()

    @app.get("/v1/history")
    def history(limit: int = Query(default=50, ge=1, le=1000)) -> dict[str, Any]:
        c = container_or_init()
        return {"items": c.history.list(limit), "statistics": c.history.statistics()}

    @app.delete("/v1/history/{record_id}")
    def delete_history(record_id: int) -> dict[str, Any]:
        return {"deleted": container_or_init().history.delete(record_id)}

    @app.post("/v1/history/clear")
    def clear_history() -> dict[str, Any]:
        container_or_init().history.clear()
        return {"cleared": True}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.server:app", host="127.0.0.1", port=8000, reload=False)
