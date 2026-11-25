import asyncio
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from .config import settings
from .hardware import detect
from .jobs import manager
from .schemas import Job, JobRequest
from .security import AuthError, authenticate, check_public_url, verify_file_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

PUBLIC_PATHS = {"/health"}
FILE_RE = re.compile(r"^/files/([0-9a-f]{32})/([0-9A-Za-z_-]{1,64}\.mp4)$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .tts import pool

    detect()
    manager.start()
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


@app.post("/jobs", status_code=202)
async def create_job(req: JobRequest, request: Request, callback_url: str | None = Query(None)):
    client_id = request.state.client_id
    if callback_url:
        req.callback_url = callback_url
    if len(req.items) > settings.max_items_per_job:
        raise HTTPException(413, f"tối đa {settings.max_items_per_job} items mỗi job")
    if manager.active_jobs(client_id) >= settings.max_active_jobs_per_client:
        raise HTTPException(429, f"client đang có {settings.max_active_jobs_per_client} job chưa xong")
    try:
        if req.callback_url:
            await asyncio.to_thread(check_public_url, req.callback_url, settings.callback_url_allowlist)
        for item in req.items:
            if item.video_url and item.status in req.process_statuses:
                await asyncio.to_thread(check_public_url, item.video_url, settings.video_url_allowlist)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job = manager.submit(req, client_id)
    return {"job_id": job.job_id, "status_url": f"{settings.public_base_url.rstrip('/')}/jobs/{job.job_id}", "items": len(job.items)}


@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str, request: Request):
    job = manager.jobs.get(job_id)
    # Không phân biệt "không tồn tại" và "của client khác" để tránh dò job_id
    if not job or job.client_id != request.state.client_id:
        raise HTTPException(404, "job not found")
    return job


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


@app.get("/status")
async def status():
    from .tts import pool

    return {"hardware": detect().as_dict(), "tts_pool": pool.stats(), "queue": manager.queue.qsize()}
