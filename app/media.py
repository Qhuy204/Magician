import json
import subprocess
from pathlib import Path

import httpx

from . import runctx
from .errors import InvalidInput, NetworkError, SourceExpired

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def run(*args: str) -> str:
    proc = runctx.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True,
                      encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-800:]}")
    return proc.stdout


def probe(path: Path) -> dict:
    proc = runctx.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                      capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise InvalidInput(f"file video không đọc được: {proc.stderr.strip()[-300:]}")
    out = proc.stdout
    return json.loads(out)


def has_audio(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path)["streams"])


def download(url: str, dest: Path) -> Path:
    """Tải video (ghi ra .part rồi rename, nên file dở dang không bị dùng lại khi retry).
    Check SSRF ở mọi hop redirect, giới hạn dung lượng, kill được qua runctx. Retry do scheduler lo."""
    from urllib.parse import urlsplit

    from .config import settings
    from .security import check_public_url

    limit = settings.max_video_mb * 1024 * 1024
    host = urlsplit(url).netloc
    hook = lambda req: check_public_url(str(req.url), settings.video_url_allowlist)
    timeout = httpx.Timeout(120, connect=settings.download_connect_timeout)
    part = dest.with_name(dest.name + ".part")
    try:
        with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, max_redirects=5, timeout=timeout,
                          event_hooks={"request": [hook]}) as client, client.stream("GET", url) as r:
            if r.status_code in (401, 403, 404, 410):
                raise SourceExpired(f"{host} trả HTTP {r.status_code}: link video hết hạn hoặc sai chữ ký, gửi link mới qua /retry")
            if r.status_code == 429 or r.status_code >= 500:
                raise NetworkError(f"{host} trả HTTP {r.status_code}")
            r.raise_for_status()
            if int(r.headers.get("content-length") or 0) > limit:
                raise InvalidInput(f"video lớn hơn {settings.max_video_mb}MB")
            size = 0
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    runctx.check()
                    size += len(chunk)
                    if size > limit:
                        raise InvalidInput(f"video lớn hơn {settings.max_video_mb}MB")
                    f.write(chunk)
    except httpx.TransportError as e:
        part.unlink(missing_ok=True)
        raise NetworkError(f"không kết nối được {host}: {e.__class__.__name__}: {e}") from e
    except ValueError as e:  # SSRF check của redirect
        part.unlink(missing_ok=True)
        raise InvalidInput(str(e)) from e
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


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
