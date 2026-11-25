"""Client mẫu gọi Magician API có ký HMAC. Chỉ cần thư viện chuẩn + httpx.

    python magician_client.py submit payload.json --callback https://your-host/webhook
    python magician_client.py status <job_id>

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

    def request(self, method: str, path: str, params: dict | None = None, json_body=None) -> dict:
        path_qs = path + (f"?{urlencode(params)}" if params else "")
        body = json.dumps(json_body, ensure_ascii=False).encode() if json_body is not None else b""
        headers = self._headers(method, path_qs, body)
        if body:
            headers["Content-Type"] = "application/json"
        r = httpx.request(method, self.base_url + path_qs, content=body, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def submit(self, payload: dict, callback_url: str | None = None) -> dict:
        return self.request("POST", "/jobs", {"callback_url": callback_url} if callback_url else None, payload)

    def status(self, job_id: str) -> dict:
        return self.request("GET", f"/jobs/{job_id}")


def verify_webhook(secret: str, body: bytes, timestamp: str, signature: str, max_age: int = 300) -> bool:
    """Kiểm tra webhook Magician gửi về (header X-Magician-Timestamp, X-Magician-Signature)."""
    if not timestamp.isdigit() or abs(time.time() - int(timestamp)) > max_age:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


if __name__ == "__main__":
    c = MagicianClient(os.environ.get("MAGICIAN_URL", "https://magician.qhuy204.id.vn"),
                       os.environ["MAGICIAN_CLIENT_ID"], os.environ["MAGICIAN_SECRET"])
    cmd = sys.argv[1]
    if cmd == "submit":
        cb = sys.argv[sys.argv.index("--callback") + 1] if "--callback" in sys.argv else None
        print(json.dumps(c.submit(json.load(open(sys.argv[2], encoding="utf-8")), cb), indent=2))
    elif cmd == "status":
        print(json.dumps(c.status(sys.argv[2]), indent=2, ensure_ascii=False))
