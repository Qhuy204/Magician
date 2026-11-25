import json
import subprocess
from pathlib import Path

import httpx

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def run(*args: str) -> str:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-800:]}")
    return proc.stdout


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def has_audio(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path)["streams"])


def download(url: str, dest: Path, attempts: int = 4) -> Path:
    """Tải video; check SSRF ở mọi hop redirect và giới hạn dung lượng. Retry khi lỗi mạng tạm thời."""
    import logging
    import time
    from urllib.parse import urlsplit

    from .config import settings
    from .security import check_public_url

    limit = settings.max_video_mb * 1024 * 1024
    hook = lambda req: check_public_url(str(req.url), settings.video_url_allowlist)
    timeout = httpx.Timeout(120, connect=30)
    for i in range(attempts):
        try:
            with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, max_redirects=5, timeout=timeout,
                              event_hooks={"request": [hook]}) as client, client.stream("GET", url) as r:
                r.raise_for_status()
                if int(r.headers.get("content-length") or 0) > limit:
                    raise ValueError(f"video lớn hơn {settings.max_video_mb}MB")
                size = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(1 << 16):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"video lớn hơn {settings.max_video_mb}MB")
                        f.write(chunk)
            return dest
        except httpx.TransportError as e:
            if i == attempts - 1:
                raise RuntimeError(f"không tải được video từ {urlsplit(url).netloc} sau {attempts} lần: {e}") from e
            logging.getLogger(__name__).warning("download lỗi (%s), thử lại %d/%d", e, i + 1, attempts - 1)
            time.sleep(2 * (i + 1))


def extract_audio(video: Path, out: Path, sr: int = 44100, channels: int = 2) -> Path:
    run("-i", str(video), "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_f32le", str(out))
    return out


def to_mono_16k(src: Path, out: Path) -> Path:
    run("-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out))
    return out


_TRIM = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.03"
TRIM_BOTH = f"{_TRIM},areverse,{_TRIM},areverse"


def condition_tts(src: Path, out: Path, tempo: float = 1.0, sr: int = 44100) -> Path:
    """Trim silence đầu/cuối, đổi tốc độ (giữ pitch) và resample về stereo `sr`."""
    filters = TRIM_BOTH + (f",atempo={tempo:.4f}" if abs(tempo - 1.0) > 1e-3 else "")
    run("-i", str(src), "-af", filters, "-ac", "2", "-ar", str(sr), "-c:a", "pcm_f32le", str(out))
    return out


def mux(video: Path, audio: Path, out: Path) -> Path:
    run(
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out),
    )
    return out
