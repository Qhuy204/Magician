"""E2E với pipeline thật (GPU + OpenAI) trên 1 instance test riêng. Không chạy trong pytest.

  1. Start instance:  DATA_DIR=D:/cache/e2e API_KEYS=e2e:e2esecret ALLOW_PRIVATE_URLS=true ALLOW_HTTP_URLS=true \
                      VIDEO_URL_ALLOWLIST= python -m uvicorn app.main:app --port 8010
  2. python tests/e2e_real.py <scenario>   (scenario: full | resume | cancel | expired)
Video mẫu lấy từ data/test/visual (http.server nội bộ port 8099).
"""
import hashlib
import hmac
import http.server
import json
import secrets
import sys
import threading
import time
from functools import partial
from pathlib import Path

import httpx

BASE, CID, SECRET = "http://127.0.0.1:8010", "e2e", "e2esecret"
VIS = Path(__file__).resolve().parents[1] / "data" / "test" / "visual"


def serve():
    h = partial(http.server.SimpleHTTPRequestHandler, directory=str(VIS))
    h.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8099), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def call(method, path, payload=None):
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else b""
    ts, nonce = str(int(time.time())), secrets.token_hex(12)
    msg = "\n".join([ts, nonce, method, path, hashlib.sha256(body).hexdigest()])
    h = {"X-Client-Id": CID, "X-Timestamp": ts, "X-Nonce": nonce,
         "X-Signature": hmac.new(SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()}
    if body:
        h["Content-Type"] = "application/json"
    r = httpx.request(method, BASE + path, content=body, headers=h, timeout=30)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


def ad(ad_id: str, video: str = "synthetic2.mp4"):
    a = json.loads((VIS / "ad_synth2.json").read_text(encoding="utf-8"))
    a.update(smart_ad_id=ad_id, video_url=f"http://127.0.0.1:8099/{video}")
    return {"advertiser_id": "e2e", "campaign": {"name": "e2e"}, "ads": [a]}


def item(job):
    return job["items"][0]


def poll(job_id, pred, timeout=400, every=1.5, show=True):
    end, last = time.time() + timeout, None
    while time.time() < end:
        try:
            code, job = call("GET", f"/jobs/{job_id}")
        except httpx.HTTPError:
            time.sleep(every)
            continue
        it = item(job)
        cur = (job["state"], it["state"], it["stage"], it["attempt"])
        if show and cur != last:
            print(f"  {time.strftime('%X')} job={cur[0]} item={cur[1]} stage={cur[2]} attempt={cur[3]}", flush=True)
            last = cur
        if pred(job):
            return job
        time.sleep(every)
    raise SystemExit(f"timeout: {json.dumps(job, ensure_ascii=False)[:800]}")


def events(job_id):
    return [(e["kind"], (e["message"] or "")[:120]) for e in call("GET", f"/jobs/{job_id}/events")[1]["events"]]


def main(scenario: str, ad_id: str | None = None):
    serve()
    ad_id = ad_id or f"{scenario}_{int(time.time())}"
    if scenario == "poll":  # dùng sau khi restart server ở kịch bản resume
        job = poll(sys.argv[3], lambda j: j["state"] in ("done", "failed", "cancelled"))
        it = item(job)
        print(json.dumps({k: it[k] for k in ("state", "attempt", "error_code", "error", "timings", "usage", "output_video_url")},
                         ensure_ascii=False, indent=1))
        print(events(sys.argv[3]))
        return
    payload = ad(ad_id, "missing.mp4" if scenario == "expired" else "synthetic2.mp4")
    code, r = call("POST", "/jobs", payload)
    print("POST /jobs", code, r)
    jid = r["job_id"]
    if scenario == "full":
        job = poll(jid, lambda j: j["state"] != "queued" and j["state"] != "running")
        it = item(job)
        print(json.dumps({k: it[k] for k in ("state", "attempt", "timings", "usage", "output_video_url")}, ensure_ascii=False, indent=1))
    elif scenario == "resume":
        poll(jid, lambda j: item(j)["stage"] in ("tts", "render"))
        print("READY_TO_KILL", jid, flush=True)
    elif scenario == "cancel":
        poll(jid, lambda j: item(j)["stage"] in ("ocr", "plan", "render", "tts"))
        t = time.time()
        print("cancel ->", call("POST", f"/jobs/{jid}/cancel"))
        job = poll(jid, lambda j: j["state"] == "cancelled", timeout=200)
        st = call("GET", "/status")[1]["scheduler"]
        print(f"cancelled after {time.time() - t:.1f}s; scheduler running={st['running']}")
        # chờ thread thật sự dừng
        end = time.time() + 120
        while call("GET", "/status")[1]["scheduler"]["running"] and time.time() < end:
            time.sleep(0.5)
        print(f"slot freed after {time.time() - t:.1f}s")
        print(events(jid))
    elif scenario == "expired":
        job = poll(jid, lambda j: j["state"] == "done")
        it = item(job)
        print("item:", it["state"], it["error_code"], it["retryable"], it["error"])
        code, r = call("POST", f"/jobs/{jid}/retry", {"video_urls": {ad_id: "http://127.0.0.1:8099/synthetic2.mp4"}})
        print("retry", code, r)
        job = poll(jid, lambda j: j["state"] == "done" and item(j)["state"] not in ("pending", "running"))
        it = item(job)
        print("after retry:", it["state"], it["attempt"], it["output_video_url"] is not None)
        print(events(jid))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] != "poll" else None)
