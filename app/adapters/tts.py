"""离线语音播报（pyttsx3 / Windows SAPI），后台线程播放，可中断。"""
from __future__ import annotations

import re
import threading
from pathlib import Path

from app.core.config import Settings
from app.core.errors import TTSUnavailable
from app.core.logging import get_logger

logger = get_logger(__name__)


def _strip_markup(text: str) -> str:
    text = re.sub(r"[#*`_>]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunks(text: str, limit: int = 400) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if not part.strip():
            continue
        if len(buffer) + len(part) > limit and buffer:
            chunks.append(buffer)
            buffer = part
        else:
            buffer += part
    if buffer.strip():
        chunks.append(buffer)
    return chunks


class TTSPlayer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @staticmethod
    def is_available() -> bool:
        try:
            import pyttsx3  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    @property
    def is_speaking(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def speak(self, text: str) -> bool:
        if not self.settings.tts_enabled:
            logger.info("语音播报未启用")
            return False
        if not self.is_available():
            raise TTSUnavailable("未安装 pyttsx3，请执行 pip install -r requirements.txt")
        content = _strip_markup(text)[:2000]
        if not content:
            return False
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(content,), daemon=True, name="tts-player")
        self._thread.start()
        return True

    def _run(self, content: str) -> None:
        with self._lock:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                engine.setProperty("volume", 1.0)
                self._apply_voice(engine)
                for chunk in _chunks(content):
                    if self._stop_event.is_set():
                        break
                    engine.say(chunk)
                    engine.runAndWait()
                    if self._stop_event.is_set():
                        break
                engine.stop()
            except Exception as exc:  # noqa: BLE001
                logger.error("语音播报失败", extra={"ctx_error": str(exc)})

    def _apply_voice(self, engine: object) -> None:
        try:
            voices = engine.getProperty("voices")  # type: ignore[attr-defined]
            for voice in voices:
                if "chinese" in (voice.name or "").lower() or "zh" in (voice.id or "").lower():
                    engine.setProperty("voice", voice.id)  # type: ignore[attr-defined]
                    break
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def export_wav(self, text: str, output: str | Path) -> Path:
        """离线合成到文件，便于交付与离线演示。"""
        if not self.is_available():
            raise TTSUnavailable("未安装 pyttsx3")
        target = Path(output)
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.save_to_file(_strip_markup(text)[:2000], str(target))
            engine.runAndWait()
            return target
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailable(f"语音合成失败: {exc}") from exc
