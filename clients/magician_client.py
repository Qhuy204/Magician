"""Client mẫu gọi Magician API có ký HMAC. Chỉ cần thư viện chuẩn + httpx.

    python magician_client.py submit payload.json --callback https://your-host/webhook [--key <idempotency-key>]
    python magician_client.py status <job_id>
    python magician_client.py wait <job_id>
    python magician_client.py events <job_id>
    python magician_client.py retry <job_id> [smart_ad_id=new_video_url ...]
    python magician_client.py cancel <job_id> [smart_ad_id ...]
    python magician_client.py list [state]

Env: MAGICIAN_URL, MAGICIAN_CLIENT_ID, MAGICIAN_SECRET
"""
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from urllib.parse import urlencode, urlparse

import httpx


class MagicianClient:
    def __init__(self, base_url: str, client_id: str, secret: str) -> None:
        self.base_url, self.client_id, self.secret = base_url.rstrip("/"), client_id, secret

    def _headers(self, method: str, path_qs: str, body: bytes) -> dict[str, str]:
        ts, nonce = str(int(time.time())), secrets.token_hex(16)
        msg = "\n".join([ts, nonce, method.upper(), path_qs, hashlib.sha256(body).hexdigest()])
        return {
            "X-Client-Id": self.client_id,
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "X-Signature": hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).hexdigest(),
        }

    def request(self, method: str, path: str, params: dict | None = None, json_body=None,
                extra_headers: dict | None = None) -> dict:
        path_qs = path + (f"?{urlencode(params)}" if params else "")
        body = json.dumps(json_body, ensure_ascii=False).encode() if json_body is not None else b""
        headers = {**self._headers(method, path_qs, body), **(extra_headers or {})}
        if body:
            headers["Content-Type"] = "application/json"
        r = httpx.request(method, self.base_url + path_qs, content=body, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def submit(self, payload: dict, callback_url: str | None = None, idempotency_key: str | None = None) -> dict:
        """idempotency_key: gửi lại cùng key (vd retry khi mạng lỗi) thì nhận lại đúng job cũ."""
        return self.request("POST", "/jobs", {"callback_url": callback_url} if callback_url else None, payload,
                            {"Idempotency-Key": idempotency_key} if idempotency_key else None)

    def status(self, job_id: str) -> dict:
        return self.request("GET", f"/jobs/{job_id}")

    def list_jobs(self, state: str | None = None, limit: int = 50) -> dict:
        return self.request("GET", "/jobs", {k: v for k, v in {"state": state, "limit": limit}.items() if v})

    def events(self, job_id: str, after: int = 0) -> dict:
        return self.request("GET", f"/jobs/{job_id}/events", {"after": after} if after else None)

    def retry(self, job_id: str, smart_ad_ids: list[str] | None = None, video_urls: dict[str, str] | None = None,
              force: bool = False) -> dict:
        """Retry item failed/cancelled. video_urls = {smart_ad_id: link mới} cho lỗi source_expired."""
        body = {"video_urls": video_urls or {}, "force": force}
        if smart_ad_ids is not None:
            body["smart_ad_ids"] = smart_ad_ids
        return self.request("POST", f"/jobs/{job_id}/retry", json_body=body)

    def cancel(self, job_id: str, smart_ad_ids: list[str] | None = None) -> dict:
        """Huỷ / kill: bỏ trống smart_ad_ids = huỷ cả job."""
        return self.request("POST", f"/jobs/{job_id}/cancel",
                            json_body={"smart_ad_ids": smart_ad_ids} if smart_ad_ids is not None else None)

    def wait(self, job_id: str, timeout: float = 1800, interval: float = 5) -> dict:
        """Poll tới khi job xong (done / cancelled / failed). Item `retrying` vẫn tính là chưa xong."""
        end = time.time() + timeout
        while True:
            job = self.status(job_id)
            if job["state"] in ("done", "cancelled", "failed") or time.time() > end:
                return job
            time.sleep(interval)


def verify_webhook(secret: str, body: bytes, timestamp: str, signature: str, max_age: int = 300) -> bool:
    """Kiểm tra webhook Magician gửi về (header X-Magician-Timestamp, X-Magician-Signature)."""
    if not timestamp.isdigit() or abs(time.time() - int(timestamp)) > max_age:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


if __name__ == "__main__":
    c = MagicianClient(os.environ.get("MAGICIAN_URL", "https://magician.qhuy204.id.vn"),
                       os.environ["MAGICIAN_CLIENT_ID"], os.environ["MAGICIAN_SECRET"])
    cmd, args = sys.argv[1], sys.argv[2:]
    opt = lambda name: args[args.index(name) + 1] if name in args else None
    if cmd == "submit":
        out = c.submit(json.load(open(args[0], encoding="utf-8")), opt("--callback"), opt("--key"))
    elif cmd == "status":
        out = c.status(args[0])
    elif cmd == "wait":
        out = c.wait(args[0])
    elif cmd == "events":
        out = c.events(args[0])
    elif cmd == "retry":
        urls = dict(a.split("=", 1) for a in args[1:] if "=" in a)
        out = c.retry(args[0], list(urls) or None, urls)
    elif cmd == "cancel":
        out = c.cancel(args[0], args[1:] or None)
    elif cmd == "list":
        out = c.list_jobs(args[0] if args else None)
    else:
        raise SystemExit(__doc__)
    print(json.dumps(out, indent=2, ensure_ascii=False))
