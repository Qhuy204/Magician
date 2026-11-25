"""Test backend (queue / retry / cancel / timeout / resume / webhook / dedup) với pipeline giả, không cần GPU / OpenAI.

Chạy: D:/envs/omnivoice/python.exe -m pytest tests -q
"""
import hashlib
import hmac
import http.server
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="magician_test_"))
os.environ.update({
    "DATA_DIR": str(TMP), "TTS_WARMUP": "false", "API_KEYS": "dash:s3cret,other:0ther",
    "ALLOW_PRIVATE_URLS": "true", "ALLOW_HTTP_URLS": "true", "VIDEO_URL_ALLOWLIST": "", "CALLBACK_URL_ALLOWLIST": "",
    "CALLBACK_SECRET": "hooksecret", "WEBHOOK_RETRY_DELAYS": "0.2,0.2", "STAGE_TIMEOUTS": "slowstage:1",
    "ITEM_CONCURRENCY": "2", "RATE_LIMIT_PER_MINUTE": "100000", "MAX_ACTIVE_JOBS_PER_CLIENT": "50",
    "PUBLIC_BASE_URL": "http://testserver",
})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import errors, jobs as jobs_mod, runctx  # noqa: E402
from app.main import app  # noqa: E402
from app.security import body_signature  # noqa: E402

for p in errors.POLICIES.values():
    p.delays = tuple(0.3 for _ in p.delays)

# ----------------------------------------------------------------- fake pipeline
CALLS: dict[str, int] = {}
GATE = threading.Event()


def fake_process(item, res, reasons, work, out, campaign, cache):
    n = CALLS[item.smart_ad_id] = CALLS.get(item.smart_ad_id, 0) + 1
    work.mkdir(parents=True, exist_ok=True)
    kind = item.smart_ad_id.split("_")[0]
    res.reject_reasons = [r.summary() for r in reasons]
    with runctx.stage("download"):
        if kind == "expired":
            raise errors.SourceExpired("host trả HTTP 403")
        if kind == "flaky" and n <= 2:
            raise errors.NetworkError("WinError 10060 timeout")
        if kind == "internal":
            raise KeyError("bug")
    if kind == "slow":
        with runctx.stage("render"):
            while not GATE.is_set():
                runctx.check()
                time.sleep(0.05)
    if kind == "hang":
        with runctx.stage("slowstage"):
            if n == 1:
                # subprocess bị kill khi quá timeout
                runctx.run([sys.executable, "-c", "import time; time.sleep(60)"])
    if kind == "ffkill":
        with runctx.stage("render"):
            runctx.run([sys.executable, "-c", "import time; time.sleep(60)"])
    if kind == "unfixable":
        res.plan = {"decision": "unfixable"}
        res.state, res.error = "no_changes", "product-level"
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mp4")
    res.state = "done"


jobs_mod.process_item = fake_process


# --------------------------------------------------------------------- webhook
HOOKS: list[dict] = []


class Hook(http.server.BaseHTTPRequestHandler):
    fail_first = {"n": 0}

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        ts, sig = self.headers["X-Magician-Timestamp"], self.headers["X-Magician-Signature"]
        exp = "sha256=" + hmac.new(b"hooksecret", ts.encode() + b"." + body, hashlib.sha256).hexdigest()
        ok_sig = hmac.compare_digest(exp, sig)
        if self.path == "/flaky" and Hook.fail_first["n"] < 2:
            Hook.fail_first["n"] += 1
            self.send_response(500)
            self.end_headers()
            return
        HOOKS.append({"path": self.path, "sig_ok": ok_sig, "event": self.headers["X-Magician-Event"],
                      "event_id": self.headers["X-Magician-Event-Id"], "body": json.loads(body)})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hook)
threading.Thread(target=srv.serve_forever, daemon=True).start()
HOOK_URL = f"http://127.0.0.1:{srv.server_port}"


# ---------------------------------------------------------------------- helpers
def signed(client, method, path, payload=None, cid="dash", secret="s3cret", headers=None):
    body = json.dumps(payload).encode() if payload is not None else b""
    ts, nonce = str(int(time.time())), secrets.token_hex(12)
    h = {"X-Client-Id": cid, "X-Timestamp": ts, "X-Nonce": nonce,
         "X-Signature": body_signature(secret, ts, nonce, method, path, body), **(headers or {})}
    if body:
        h["Content-Type"] = "application/json"
    return client.request(method, path, content=body, headers=h)


def payload(*ad_ids, callback=None):
    return {"advertiser_id": "1", "campaign": {"name": "c", "campaign_id": "2"}, "callback_url": callback,
            "ads": [{"smart_ad_id": a, "review_status": "UNAVAILABLE", "rejected_video_id": f"v_{a}",
                     "video_url": f"http://127.0.0.1/{a}.mp4", "reasons": [{"category": "x", "detail": "y"}]}
                    for a in ad_ids]}


def wait(client, job_id, pred, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        job = signed(client, "GET", f"/jobs/{job_id}").json()
        if pred(job):
            return job
        time.sleep(0.1)
    raise AssertionError(f"timeout, last: {json.dumps(job, ensure_ascii=False)[:1500]}")


def items(job):
    return {i["smart_ad_id"]: i for i in job["items"]}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------------ tests
def test_success_and_unfixable(client):
    r = signed(client, "POST", "/jobs", payload("ok_1", "unfixable_1"))
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"], lambda j: j["state"] == "done")
    it = items(job)
    assert it["ok_1"]["state"] == "done" and "/files/" in it["ok_1"]["output_video_url"]
    assert it["unfixable_1"]["state"] == "no_changes" and it["unfixable_1"]["error_code"] == "unfixable"
    assert job["progress"]["finished"] == 2 and job["finished_at"]
    file = signed(client, "GET", "/jobs").json()
    assert any(j["job_id"] == job["job_id"] for j in file["jobs"])
    url = it["ok_1"]["output_video_url"].replace("http://testserver", "")
    assert client.get(url).content == b"mp4"


def test_auto_retry_network(client):
    r = signed(client, "POST", "/jobs", payload("flaky_1"))
    job = wait(client, r.json()["job_id"], lambda j: j["state"] == "done")
    it = items(job)["flaky_1"]
    assert it["state"] == "done" and it["attempt"] == 3
    kinds = [e["kind"] for e in signed(client, "GET", f"/jobs/{job['job_id']}/events").json()["events"]]
    assert kinds.count("retry_scheduled") == 2


def test_source_expired_then_manual_retry_with_new_url(client):
    r = signed(client, "POST", "/jobs", payload("expired_1"))
    jid = r.json()["job_id"]
    job = wait(client, jid, lambda j: j["state"] == "done")
    it = items(job)["expired_1"]
    assert it["state"] == "failed" and it["error_code"] == "source_expired" and it["retryable"] and it["attempt"] == 1
    # sửa fake: lần sau link mới => thành công
    jobs_mod.manager.requests[jid].items[0].smart_ad_id  # sanity
    orig = jobs_mod.process_item

    def fixed(item, res, *a):
        assert item.video_url.endswith("new.mp4")
        res.state = "done"
        a[2].parent.mkdir(parents=True, exist_ok=True)
        a[2].write_bytes(b"x")

    jobs_mod.process_item = fixed
    try:
        r = signed(client, "POST", f"/jobs/{jid}/retry", {"video_urls": {"expired_1": "http://127.0.0.1/new.mp4"}})
        assert r.status_code == 200 and r.json()["retried"] == ["expired_1"], r.text
        job = wait(client, jid, lambda j: j["state"] == "done" and items(j)["expired_1"]["state"] == "done")
    finally:
        jobs_mod.process_item = orig
    # retry lần nữa khi không còn item lỗi => 409
    assert signed(client, "POST", f"/jobs/{jid}/retry").status_code == 409


def test_internal_error_not_retried(client):
    r = signed(client, "POST", "/jobs", payload("internal_1"))
    it = items(wait(client, r.json()["job_id"], lambda j: j["state"] == "done"))["internal_1"]
    assert it["state"] == "failed" and it["error_code"] == "internal" and it["retryable"] is False and it["attempt"] == 1


def test_cancel_running_and_pending(client):
    GATE.clear()
    r = signed(client, "POST", "/jobs", payload("slow_a", "slow_b", "slow_c"))
    jid = r.json()["job_id"]
    wait(client, jid, lambda j: sum(i["state"] == "running" for i in j["items"]) == 2)
    r = signed(client, "POST", f"/jobs/{jid}/cancel")
    assert sorted(r.json()["cancelled"]) == ["slow_a", "slow_b", "slow_c"]
    job = wait(client, jid, lambda j: j["state"] == "cancelled")
    assert all(i["state"] == "cancelled" for i in job["items"])
    # retry item bị huỷ
    GATE.set()
    signed(client, "POST", f"/jobs/{jid}/retry", {"smart_ad_ids": ["slow_a"]})
    job = wait(client, jid, lambda j: j["state"] == "done")
    assert items(job)["slow_a"]["state"] == "done"


def test_kill_subprocess(client):
    r = signed(client, "POST", "/jobs", payload("ffkill_1"))
    jid = r.json()["job_id"]
    wait(client, jid, lambda j: items(j)["ffkill_1"]["stage"] == "render")
    t = time.time()
    signed(client, "POST", f"/jobs/{jid}/cancel", {"smart_ad_ids": ["ffkill_1"]})
    job = wait(client, jid, lambda j: j["state"] != "running" and j["state"] != "queued", timeout=10)
    assert time.time() - t < 5
    assert items(job)["ffkill_1"]["state"] == "cancelled"


def test_stage_timeout_kills_and_retries(client):
    r = signed(client, "POST", "/jobs", payload("hang_1"))
    job = wait(client, r.json()["job_id"], lambda j: j["state"] == "done", timeout=20)
    it = items(job)["hang_1"]
    assert it["state"] == "done" and it["attempt"] == 2
    evs = signed(client, "GET", f"/jobs/{job['job_id']}/events").json()["events"]
    assert any(e["kind"] == "retry_scheduled" and "stage_timeout" in e["message"] for e in evs)


def test_dedup_and_idempotency(client):
    GATE.clear()
    r1 = signed(client, "POST", "/jobs", payload("slow_dup"))
    r2 = signed(client, "POST", "/jobs", payload("slow_dup"))
    assert r2.json()["deduplicated"] and r2.json()["job_id"] == r1.json()["job_id"]
    # 1 ad trùng + 1 ad mới => job mới, ad trùng skipped kèm duplicate_of
    r3 = signed(client, "POST", "/jobs", payload("slow_dup", "ok_new"))
    assert not r3.json()["deduplicated"]
    it = items(signed(client, "GET", f"/jobs/{r3.json()['job_id']}").json())
    assert it["slow_dup"]["state"] == "skipped" and it["slow_dup"]["duplicate_of"] == r1.json()["job_id"]
    GATE.set()
    k = {"Idempotency-Key": "abc-123"}
    a = signed(client, "POST", "/jobs", payload("ok_idem"), headers=k).json()
    b = signed(client, "POST", "/jobs", payload("ok_idem"), headers=k).json()
    assert a["job_id"] == b["job_id"] and b["deduplicated"]


def test_owner_isolation(client):
    jid = signed(client, "POST", "/jobs", payload("ok_iso")).json()["job_id"]
    assert signed(client, "GET", f"/jobs/{jid}", cid="other", secret="0ther").status_code == 404
    assert signed(client, "POST", f"/jobs/{jid}/cancel", cid="other", secret="0ther").status_code == 404
    assert client.get(f"/jobs/{jid}/events").status_code == 401


def test_webhooks_item_and_job_with_retry(client):
    HOOKS.clear()
    r = signed(client, "POST", "/jobs", payload("ok_h1", "ok_h2", callback=f"{HOOK_URL}/flaky"))
    jid = r.json()["job_id"]
    end = time.time() + 15
    while time.time() < end and not any(h["event"] == "job.done" for h in HOOKS):
        time.sleep(0.1)
    mine = [h for h in HOOKS if h["body"]["job_id"] == jid]
    assert all(h["sig_ok"] for h in mine)
    # đúng thứ tự: 2 item.done rồi mới job.done, dù item.done đầu bị 500 hai lần
    assert [h["event"] for h in mine] == ["item.done", "item.done", "job.done"]
    assert len({h["event_id"] for h in mine}) == 3
    done = next(h for h in mine if h["event"] == "job.done")["body"]
    assert done["state"] == "done" and done["progress"]["finished"] == 2
    deliveries = signed(client, "GET", f"/jobs/{jid}/webhooks").json()["deliveries"]
    assert all(d["state"] == "delivered" for d in deliveries) and max(d["attempts"] for d in deliveries) >= 2
    assert signed(client, "POST", f"/jobs/{jid}/webhooks/resend").json()["queued"] == 1


def test_ready_status_metrics(client):
    assert client.get("/ready").json() == {"ready": True}
    st = signed(client, "GET", "/status").json()
    assert st["ready"]["db"] and "scheduler" in st
    m = client.get("/metrics")  # TestClient không phải loopback => phải ký
    assert m.status_code == 401
    assert "magician_items_finished_total" in signed(client, "GET", "/metrics").text


def test_resume_after_restart():
    """Tắt server khi item đang chạy => start lại thì item chạy tiếp tới xong."""
    GATE.clear()
    with TestClient(app) as c:
        jid = signed(c, "POST", "/jobs", payload("slow_restart")).json()["job_id"]
        wait(c, jid, lambda j: items(j)["slow_restart"]["state"] == "running")
    # mô phỏng process mới: xoá toàn bộ state trong RAM
    jobs_mod.manager.__init__()
    GATE.set()
    with TestClient(app) as c:
        job = wait(c, jid, lambda j: j["state"] == "done")
        it = items(job)["slow_restart"]
        assert it["state"] == "done" and it["attempt"] >= 2
