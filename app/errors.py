"""Phân loại lỗi của item để quyết định có retry không, chờ bao lâu, và báo cho dashboard."""
import random
from dataclasses import dataclass


class MagicianError(Exception):
    code = "internal"
    retryable = False

    def __init__(self, message: str, code: str | None = None, retryable: bool | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.retry_after = retry_after


class Cancelled(MagicianError):
    code = "cancelled"


class StageTimeout(MagicianError):
    code, retryable = "stage_timeout", True


class NetworkError(MagicianError):
    code, retryable = "network", True


class SourceExpired(MagicianError):
    """Link video 401/403/404/410: link signed hết hạn hoặc sai chữ ký. Phải gửi link mới qua /retry."""
    code = "source_expired"


class InvalidInput(MagicianError):
    code = "invalid_input"


class WorkerCrash(MagicianError):
    code, retryable = "worker_crash", True


@dataclass
class Policy:
    max_attempts: int
    delays: tuple[float, ...]  # giây chờ trước lần thử thứ 2, 3, ...


POLICIES: dict[str, Policy] = {
    "network": Policy(4, (5, 15, 45)),
    "upstream_rate_limit": Policy(6, (10, 30, 60, 120, 240)),
    "upstream_error": Policy(4, (10, 30, 90)),
    "gpu_oom": Policy(3, (20, 60)),
    "worker_crash": Policy(3, (5, 20)),
    "stage_timeout": Policy(2, (10,)),
}


@dataclass
class Classified:
    code: str
    message: str
    retryable: bool
    retry_after: float | None = None


def classify(exc: BaseException) -> Classified:
    import httpx
    import openai

    msg = str(exc) or exc.__class__.__name__
    if isinstance(exc, MagicianError):
        return Classified(exc.code, msg, exc.retryable, exc.retry_after)
    if isinstance(exc, openai.RateLimitError):
        ra = None
        try:
            ra = float(exc.response.headers.get("retry-after"))
        except (TypeError, ValueError, AttributeError):
            pass
        # hết quota thì retry vô ích
        quota = "insufficient_quota" in msg
        return Classified("upstream_quota" if quota else "upstream_rate_limit", msg, not quota, ra)
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return Classified("upstream_auth", msg, False)
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)):
        return Classified("upstream_error", msg, True)
    if isinstance(exc, openai.APIStatusError):
        return Classified("upstream_error", msg, exc.status_code >= 500)
    if isinstance(exc, httpx.TransportError) or isinstance(exc, (TimeoutError, ConnectionError)):
        return Classified("network", msg, True)
    low = msg.lower()
    if "out of memory" in low or exc.__class__.__name__ == "OutOfMemoryError" or "paging file" in low:
        return Classified("gpu_oom", msg, True)
    if "tts worker" in low and ("thoát" in low or "died" in low or "không khởi động" in low):
        return Classified("worker_crash", msg, True)
    if "ffmpeg failed" in low or "invalid data found" in low or "moov atom" in low:
        return Classified("invalid_input", msg, False)
    return Classified("internal", msg, False)


def next_delay(c: Classified, attempt: int) -> float | None:
    """attempt = số lần đã chạy (1 = vừa xong lần đầu). None = không retry nữa."""
    if not c.retryable:
        return None
    p = POLICIES.get(c.code, Policy(2, (15,)))
    if attempt >= p.max_attempts:
        return None
    base = p.delays[min(attempt - 1, len(p.delays) - 1)]
    if c.retry_after:
        base = max(base, min(c.retry_after, 600))
    return base * random.uniform(0.85, 1.25)
