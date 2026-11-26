import io
import re
from collections import Counter
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from openai import OpenAI

from . import runctx
from .config import settings

# whisper verbose_json trả tên ngôn ngữ đầy đủ
LANG_CODES = {
    "english": "en", "hindi": "hi", "portuguese": "pt", "arabic": "ar", "spanish": "es",
    "french": "fr", "german": "de", "italian": "it", "polish": "pl", "turkish": "tr",
    "russian": "ru", "dutch": "nl", "czech": "cs", "chinese": "zh-cn", "hungarian": "hu",
    "korean": "ko", "japanese": "ja",
}
TIMESTAMP_MODELS = {"whisper-1"}
PAD, MIN_SPAN, MIN_SILENCE = 0.08, 0.25, 0.3


def lang_code(name: str | None) -> str | None:
    if not name:
        return None
    name = name.lower().strip()
    return LANG_CODES.get(name, name if len(name) <= 5 else None)


def speech_spans(audio: Path, total: float) -> list[tuple[float, float]]:
    """Tìm các đoạn có tiếng nói dựa trên khoảng lặng của vocal stem."""
    err = runctx.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(audio), "-af", f"silencedetect=n=-35dB:d={MIN_SILENCE}", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    spans, cursor = [], 0.0
    for i, s in enumerate(starts):
        spans.append((cursor, s))
        cursor = ends[i] if i < len(ends) else total
    spans.append((cursor, total))
    return [(max(0.0, a - PAD), min(total, b + PAD)) for a, b in spans if b - a >= MIN_SPAN]


def _transcribe_bytes(client: OpenAI, data: bytes, language: str | None, response_format: str = "json"):
    return client.audio.transcriptions.create(
        model=settings.transcribe_model,
        file=("audio.wav", data),
        response_format=response_format,
        **({"language": language.split("-")[0]} if language else {}),
        **({"timestamp_granularities": ["segment"]} if response_format == "verbose_json" else {}),
    )


def _wav_bytes(audio, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe(client: OpenAI, audio: Path, language: str | None = None) -> dict:
    """Trả về {"language", "text", "segments": [{id, start, end, text}]} cho file mono 16k."""
    data = audio.read_bytes()
    if settings.transcribe_model in TIMESTAMP_MODELS:
        r = _transcribe_bytes(client, data, language, "verbose_json")
        segments = [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()} for s in r.segments or []
        ]
        language = lang_code(getattr(r, "language", None)) or language
        text = r.text
    else:
        # gpt-transcribe không trả timestamp: cắt câu theo khoảng lặng, transcribe từng câu, ngôn ngữ lấy theo đa số
        wav, sr = sf.read(audio, dtype="float32")
        spans = speech_spans(audio, len(wav) / sr)
        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(
                lambda sp: _transcribe_bytes(client, _wav_bytes(wav[int(sp[0] * sr) : int(sp[1] * sr)], sr), language),
                spans,
            ))
        if not language:
            votes = Counter(
                lang_code(l["code"] if isinstance(l, dict) else l.code)
                for r in results for l in (getattr(r, "languages", None) or [])[:1]
            )
            votes.pop(None, None)
            language = votes.most_common(1)[0][0] if votes else None
        segments = [{"start": round(a, 3), "end": round(b, 3), "text": r.text.strip()} for (a, b), r in zip(spans, results)]
        text = " ".join(seg["text"] for seg in segments)
    segments = [{"id": i, **s} for i, s in enumerate(s for s in segments if s["text"])]
    return {"language": language, "text": text, "segments": segments}
