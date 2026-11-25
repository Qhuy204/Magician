"""Xác thực client, ký request, chống replay, rate limit, chặn SSRF và ký URL tải file."""
import hashlib
import hmac
import ipaddress
import socket
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlencode, urlparse

from .config import settings

SIGNATURE_WINDOW_S = 300


class AuthError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def clients() -> dict[str, str]:
    """client_id -> secret, từ API_KEYS="id1:secret1,id2:secret2" (+ API_KEY cũ dưới tên "default")."""
    out = {}
    for entry in filter(None, (e.strip() for e in settings.api_keys.split(","))):
        cid, _, secret = entry.partition(":")
        if cid and secret:
            out[cid] = secret
    if settings.api_key:
        out.setdefault("default", settings.api_key)
    return out


def body_signature(secret: str, timestamp: str, nonce: str, method: str, path: str, body: bytes) -> str:
    msg = "\n".join([timestamp, nonce, method.upper(), path, hashlib.sha256(body).hexdigest()])
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


class _NonceCache:
    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, key: str) -> bool:
        """False nếu nonce đã dùng trong cửa sổ thời gian (replay)."""
        now = time.time()
        with self._lock:
            if len(self._seen) > 10000:
                self._seen = {k: t for k, t in self._seen.items() if t > now}
            if self._seen.get(key, 0) > now:
                return False
            self._seen[key] = now + 2 * SIGNATURE_WINDOW_S
            return True


class RateLimiter:
    """Sliding window: tối đa `limit` sự kiện / 60s cho mỗi key."""

    def __init__(self) -> None:
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> bool:
        now = time.time()
        with self._lock:
            q = self._events[key]
            while q and q[0] <= now - 60:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def count(self, key: str) -> int:
        now = time.time()
        with self._lock:
            q = self._events[key]
            while q and q[0] <= now - 60:
                q.popleft()
            return len(q)


nonces = _NonceCache()
limiter = RateLimiter()


MAX_AUTH_FAILURES_PER_MIN = 10


def authenticate(headers, method: str, path_qs: str, body: bytes, ip: str) -> str:
    """Trả về client_id. Chấp nhận request ký HMAC; X-API-Key thuần chỉ khi REQUIRE_SIGNED_REQUESTS=false.
    IP sai quá 10 lần/phút bị chặn 429 trong 1 phút (chống brute force)."""
    known = clients()
    if not known:
        return "anonymous"
    if limiter.count(f"fail:{ip}") >= MAX_AUTH_FAILURES_PER_MIN:
        raise AuthError(429, "too many failed attempts")
    if not limiter.hit(f"ip:{ip}", settings.rate_limit_per_minute * 2):
        raise AuthError(429, "too many requests")
    try:
        return _verify(known, headers, method, path_qs, body)
    except AuthError as exc:
        if exc.status == 401:
            limiter.hit(f"fail:{ip}", 10**9)
        raise


def _verify(known: dict[str, str], headers, method: str, path_qs: str, body: bytes) -> str:
    cid, sig = headers.get("x-client-id"), headers.get("x-signature")
    if cid and sig:
        ts, nonce = headers.get("x-timestamp", ""), headers.get("x-nonce", "")
        secret = known.get(cid)
        if not secret or not ts.isdigit() or not 8 <= len(nonce) <= 128:
            raise AuthError(401, "invalid credentials")
        if abs(time.time() - int(ts)) > SIGNATURE_WINDOW_S:
            raise AuthError(401, "timestamp outside allowed window")
        expected = body_signature(secret, ts, nonce, method, path_qs, body)
        if not hmac.compare_digest(expected, sig.lower()):
            raise AuthError(401, "invalid credentials")
        if not nonces.add(f"{cid}:{nonce}"):
            raise AuthError(401, "replayed request")
    elif (key := headers.get("x-api-key")) and not settings.require_signed_requests:
        cid = next((c for c, s in known.items() if hmac.compare_digest(s.encode(), key.encode())), None)
        if not cid:
            raise AuthError(401, "invalid credentials")
    else:
        raise AuthError(401, "signed request required" if settings.require_signed_requests else "missing credentials")

    if not limiter.hit(f"client:{cid}", settings.rate_limit_per_minute):
        raise AuthError(429, "rate limit exceeded")
    return cid


def check_public_url(url: str, allowed_hosts: str = "") -> None:
    """Chặn SSRF: chỉ http(s), host phải resolve ra IP public, và khớp allowlist nếu có cấu hình."""
    p = urlparse(url)
    if p.scheme not in ("https", "http") or not p.hostname:
        raise ValueError(f"URL không hợp lệ: {url[:100]}")
    if p.scheme == "http" and not settings.allow_http_urls:
        raise ValueError("chỉ chấp nhận https URL")
    host = p.hostname.lower().rstrip(".")
    allow = [h.strip().lower() for h in allowed_hosts.split(",") if h.strip()]
    if allow and not any(host == h or host.endswith("." + h) for h in allow):
        raise ValueError(f"host không nằm trong allowlist: {host}")
    if settings.allow_private_urls:
        return
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"không resolve được host {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError(f"host {host} trỏ vào địa chỉ nội bộ {ip}")


def sign_file_path(path: str) -> str:
    """Thêm ?exp=&sig= vào đường dẫn /files/... để link tải có hạn dùng."""
    exp = int(time.time()) + settings.file_url_ttl_hours * 3600
    sig = hmac.new(settings.file_signing_key.encode(), f"{path}:{exp}".encode(), hashlib.sha256).hexdigest()
    return f"{path}?{urlencode({'exp': exp, 'sig': sig})}"


def verify_file_path(path: str, exp: str | None, sig: str | None) -> bool:
    if not exp or not sig or not exp.isdigit() or int(exp) < time.time():
        return False
    expected = hmac.new(settings.file_signing_key.encode(), f"{path}:{exp}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def webhook_headers(body: bytes) -> dict[str, str]:
    """Header ký webhook: X-Magician-Signature = HMAC_SHA256(secret, "{timestamp}.{body}")."""
    if not settings.callback_secret:
        return {}
    ts = str(int(time.time()))
    sig = hmac.new(settings.callback_secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    return {"X-Magician-Timestamp": ts, "X-Magician-Signature": f"sha256={sig}"}
