import asyncio
import logging
import re
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import logsetup
from .config import settings
from .hardware import detect
from .jobs import manager
from .schemas import CancelRequest, Job, JobRequest, RetryRequest
from .security import AuthError, authenticate, check_public_url, verify_file_path

logsetup.setup()
log = logging.getLogger(__name__)

PUBLIC_PATHS = {"/health", "/ready", "/ui", "/"}
FILE_RE = re.compile(r"^/files/([0-9a-f]{32})/([0-9A-Za-z_-]{1,64}\.mp4)$")
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DASHBOARD_HTML = Path(__file__).parent / "static" / "dashboard.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .tts import pool

    logsetup.attach_uvicorn()
    detect()
    manager.start()
    if settings.tts_warmup:
        asyncio.get_running_loop().run_in_executor(None, pool.warmup)
    yield
    await manager.stop()
    pool.shutdown()


docs = {} if settings.enable_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title="Magician - ad voice-over policy fixer", lifespan=lifespan, **docs)


def _client_ip(request: Request) -> str:
    # Server chỉ bind 127.0.0.1 sau cloudflared nên header của Cloudflare là tin được
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    # /metrics cho Prometheus chạy cùng máy: không qua tunnel (không có cf-connecting-ip) và từ loopback
    local = "cf-connecting-ip" not in request.headers and request.client and request.client.host in ("127.0.0.1", "::1")
    if path == "/metrics" and local:
        return await call_next(request)
    if path in PUBLIC_PATHS or path.startswith("/files/") or (settings.enable_docs and path in ("/docs", "/openapi.json")):
        return await call_next(request)
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > settings.max_body_bytes):
        return JSONResponse({"detail": "request body too large"}, status_code=413)
    body = await request.body()
    if len(body) > settings.max_body_bytes:
        return JSONResponse({"detail": "request body too large"}, status_code=413)
    path_qs = path + (f"?{request.url.query}" if request.url.query else "")
    try:
        request.state.client_id = authenticate(request.headers, request.method, path_qs, body, _client_ip(request))
    except AuthError as exc:
        log.warning("auth rejected %s %s from %s: %s", request.method, path, _client_ip(request), exc.detail)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    # Không cho Cloudflare/proxy cache: link /files có hạn dùng và response API là dữ liệu riêng của client
    response.headers["Cache-Control"] = "private, no-store"
    return response


def _own_job(job_id: str, request: Request) -> Job:
    job = manager.jobs.get(job_id) if JOB_ID_RE.match(job_id) else None
    # Không phân biệt "không tồn tại" và "của client khác" để tránh dò job_id
    if not job or job.client_id != request.state.client_id:
        raise HTTPException(404, "job not found")
    return job


def _status_url(job_id: str) -> str:
    return f"{settings.public_base_url.rstrip('/')}/jobs/{job_id}"


@app.post("/jobs", status_code=202)
async def create_job(req: JobRequest, request: Request, callback_url: str | None = Query(None),
                     idempotency_key: str | None = Header(None, max_length=128)):
    client_id = request.state.client_id
    if callback_url:
        req.callback_url = callback_url
    if len(req.items) > settings.max_items_per_job:
        raise HTTPException(413, f"tối đa {settings.max_items_per_job} items mỗi job")
    if manager.paused:
        return JSONResponse({"detail": f"tạm ngừng nhận job: {manager.paused}"}, status_code=503,
                            headers={"Retry-After": "300"})
    try:
        if req.callback_url:
            await asyncio.to_thread(check_public_url, req.callback_url, settings.callback_url_allowlist)
        for item in req.items:
            if item.video_url and item.status in req.process_statuses:
                await asyncio.to_thread(check_public_url, item.video_url, settings.video_url_allowlist)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    known = idempotency_key and (client_id, idempotency_key) in manager._idem
    if not known and manager.active_jobs(client_id) >= settings.max_active_jobs_per_client:
        raise HTTPException(429, f"client đang có {settings.max_active_jobs_per_client} job chưa xong")
    job, dedup = manager.submit(req, client_id, idempotency_key)
    return {"job_id": job.job_id, "status_url": _status_url(job.job_id), "items": len(job.items),
            "deduplicated": dedup, "state": job.state}


@app.get("/jobs")
async def list_jobs(request: Request, state: str | None = None, limit: int = Query(50, ge=1, le=200),
                    before: datetime | None = None):
    mine = [j for j in manager.jobs.values() if j.client_id == request.state.client_id
            and (state is None or j.state == state) and (before is None or (j.created_at and j.created_at < before))]
    mine.sort(key=lambda j: j.created_at.timestamp() if j.created_at else 0, reverse=True)
    page = mine[:limit]
    return {
        "jobs": [{"job_id": j.job_id, "state": j.state, "campaign_name": j.campaign_name, "created_at": j.created_at,
                  "finished_at": j.finished_at, "progress": manager.progress(j), "status_url": _status_url(j.job_id)}
                 for j in page],
        "next_before": page[-1].created_at if len(mine) > limit else None,
    }


@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str, request: Request):
    return manager.view(_own_job(job_id, request))


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=2000)):
    _own_job(job_id, request)
    return {"events": manager.store.events(job_id, after, limit)}


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, request: Request, body: RetryRequest | None = None):
    job = _own_job(job_id, request)
    body = body or RetryRequest()
    try:
        for url in body.video_urls.values():
            await asyncio.to_thread(check_public_url, url, settings.video_url_allowlist)
        retried = manager.retry(job, body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not retried:
        raise HTTPException(409, "không có item nào retry được (đang chạy, sai smart_ad_id, hoặc item done/no_changes "
                                 "mà không có force=true)")
    return {"job_id": job_id, "retried": retried, "state": job.state, "status_url": _status_url(job_id)}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request, body: CancelRequest | None = None):
    """Huỷ / kill: item chờ => cancelled ngay; item đang chạy => kill ffmpeg, dừng ở lần kiểm tra kế tiếp."""
    job = _own_job(job_id, request)
    cancelled = manager.cancel(job, (body or CancelRequest()).smart_ad_ids)
    return {"job_id": job_id, "cancelled": cancelled, "state": job.state}


@app.get("/jobs/{job_id}/webhooks")
async def job_webhooks(job_id: str, request: Request):
    _own_job(job_id, request)
    return {"deliveries": manager.store.webhooks(job_id)}


@app.post("/jobs/{job_id}/webhooks/resend")
async def resend_webhook(job_id: str, request: Request):
    job = _own_job(job_id, request)
    if not job.callback_url:
        raise HTTPException(409, "job không có callback_url")
    if job.state in ("queued", "running"):
        raise HTTPException(409, "job chưa xong")
    return {"queued": manager.resend_webhook(job)}


@app.get("/files/{job_id}/{name}")
async def get_file(job_id: str, name: str, exp: str | None = None, sig: str | None = None):
    path = f"/files/{job_id}/{name}"
    if not FILE_RE.match(path) or not verify_file_path(path, exp, sig):
        raise HTTPException(404, "not found")
    file = manager.outputs_dir / job_id / name
    if not file.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(file, media_type="video/mp4", filename=name)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def dashboard():
    # Shell HTML/JS tĩnh, không chứa secret: người dùng nhập khoá client ở trình duyệt,
    # mọi API call từ dashboard vẫn qua middleware ký HMAC như thường.
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


def _ready_checks() -> dict[str, bool]:
    from .tts import pool

    try:
        manager.store._all("SELECT 1")
        db = True
    except Exception:
        db = False
    return {
        "db": db,
        "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        "scheduler": bool(manager._bg) and all(not t.done() for t in manager._bg),
        "tts_worker": pool.alive_count() > 0 or not settings.tts_warmup,
        "disk": manager.paused is None,
    }


@app.get("/ready")
async def ready():
    """Readiness cho monitor: 503 nếu thiếu DB, ffmpeg, scheduler, worker TTS hoặc sắp hết đĩa. Chi tiết ở /status."""
    ok = all(_ready_checks().values())
    return JSONResponse({"ready": ok}, status_code=200 if ok else 503)


@app.get("/status")
async def status():
    from .tts import pool

    return {"ready": _ready_checks(), "scheduler": manager.stats(), "hardware": detect().as_dict(),
            "tts_pool": pool.stats()}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return manager.metrics()
