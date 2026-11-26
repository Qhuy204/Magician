"""Hàng đợi + scheduler cấp item, retry có phân loại lỗi, huỷ (kill), webhook outbox, dọn dẹp và watchdog.

Trạng thái nằm trong RAM (nhanh cho GET), mọi thay đổi ghi xuống SQLite (`store.py`) nên restart server thì
job/item đang dở được chạy tiếp, webhook chưa gửi được gửi tiếp.
"""
import asyncio
import hashlib
import json
import logging
import shutil
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import runctx
from .config import settings
from .errors import NetworkError, classify, next_delay
from .hardware import detect
from .pipeline import JobCache, process_item
from .runctx import RunContext
from .schemas import TERMINAL_ITEM_STATES, ItemResult, Job, JobRequest, RetryRequest
from .security import check_public_url, sign_file_path, webhook_headers
from .store import Store
from .tiktok import fetch_reject_info, reasons_for_video

log = logging.getLogger(__name__)

# Lỗi mà gọi /retry (thủ công) có ý nghĩa; source_expired cần kèm video_url mới
MANUAL_RETRY_CODES = {"network", "upstream_rate_limit", "upstream_error", "gpu_oom", "worker_crash", "stage_timeout",
                      "source_expired", "interrupted"}
# Field kết quả copy từ bản chạy của pipeline sang ItemResult public
RESULT_FIELDS = ("language", "reject_reasons", "edits", "error", "usage", "timings", "plan", "visual_edits", "state",
                 "policy_findings", "action_required", "targeting_fix", "compliance_check", "ad_text_fix",
                 "landing_page")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(dt: datetime | None) -> float | None:
    return dt.timestamp() if dt else None


def dedup_key(client_id: str, item, reasons_summary: list[str]) -> str:
    video = item.video_id or (urlsplit(item.video_url).path if item.video_url else "")
    raw = json.dumps([client_id, item.smart_ad_id, video, sorted(reasons_summary)], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.requests: dict[str, JobRequest] = {}
        self.store: Store | None = None
        self._caches: dict[str, JobCache] = {}
        self._running: dict[tuple[str, int], RunContext] = {}
        self._tasks: set[asyncio.Task] = set()
        self._bg: list[asyncio.Task] = []
        self._wake = asyncio.Event()
        self._served: dict[str, float] = {}
        self._idem: dict[tuple[str, str], str] = {}
        self._prepared: dict[str, asyncio.Lock] = {}
        self._cancel_requested: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        # OOM: giảm concurrency tới oom_until
        self._oom_level = 0
        self._oom_until = 0.0
        self.paused: str | None = None
        self._health_future = None
        # metrics
        self.counters: Counter = Counter()
        self.stage_seconds: Counter = Counter()
        self.stage_count: Counter = Counter()
        self.recent_errors: deque[tuple[float, str]] = deque(maxlen=2000)
        self.started_at = time.time()

    # ------------------------------------------------------------------ paths
    @property
    def jobs_dir(self) -> Path:
        return settings.data_dir / "jobs"

    @property
    def outputs_dir(self) -> Path:
        return settings.data_dir / "outputs"

    def work_dir(self, job_id: str, smart_ad_id: str) -> Path:
        return self.jobs_dir / job_id / smart_ad_id

    # ---------------------------------------------------------------- startup
    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.db_path)
        self._import_legacy()
        since = time.time() - settings.job_retention_days * 86400
        resumed = 0
        for row in self.store.load_jobs(since):
            job = Job.model_validate_json(row["data"])
            self.jobs[job.job_id] = job
            if row["idem_key"]:
                self._idem[(job.client_id, row["idem_key"])] = job.job_id
            if job.state in ("queued", "running"):
                if not row["request"]:
                    self._fail_unresumable(job)
                    continue
                self.requests[job.job_id] = JobRequest.model_validate_json(row["request"])
                self._caches[job.job_id] = JobCache()
                for res in job.items:
                    if res.state == "running":
                        res.state, res.stage = "pending", None
                        resumed += 1
                        self.event(job, res.smart_ad_id, "resumed", "server restart, chạy tiếp từ bước đã lưu")
                self.save(job)
        if resumed:
            log.info("resumed %d interrupted items", resumed)
        self._bg = [asyncio.create_task(c) for c in (self._scheduler(), self._webhook_sender(), self._maintenance())]

    def _import_legacy(self) -> None:
        """Job cũ lưu dạng data/jobs/<id>/job.json (trước khi có SQLite): import 1 lần, không resume được."""
        for f in self.jobs_dir.glob("*/job.json"):
            job_id = f.parent.name
            if self.store.has_job(job_id):
                continue
            try:
                job = Job.model_validate_json(f.read_text(encoding="utf-8"))
            except ValueError:
                continue
            job.created_at = job.created_at or datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)
            if job.state in ("queued", "running"):
                job.state, job.callback_status = "failed", "interrupted by server restart"
            job.finished_at = job.finished_at or job.created_at
            self.store.save_job(job.job_id, job.client_id, job.state, _ts(job.created_at), _ts(job.finished_at),
                                job.model_dump_json())

    def _fail_unresumable(self, job: Job) -> None:
        for res in job.items:
            if res.state not in TERMINAL_ITEM_STATES:
                res.state, res.error_code, res.error, res.retryable = "failed", "interrupted", "server restart, thiếu request gốc", False
        job.state, job.finished_at = "failed", _now()
        self.save(job)

    async def stop(self) -> None:
        for ctx in list(self._running.values()):
            ctx.kill("shutdown")
        for t in self._bg:
            t.cancel()

    # ------------------------------------------------------------ persistence
    def save(self, job: Job, request: JobRequest | None = None, idem_key: str | None = None) -> None:
        self.store.save_job(job.job_id, job.client_id, job.state, _ts(job.created_at), _ts(job.finished_at),
                            job.model_dump_json(), idem_key, request.model_dump_json() if request else None)

    def event(self, job: Job, smart_ad_id: str | None, kind: str, message: str = "", data: dict | None = None) -> None:
        self.store.add_event(job.job_id, smart_ad_id, kind, message, data)

    # ----------------------------------------------------------------- submit
    def active_jobs(self, client_id: str) -> int:
        return sum(1 for j in self.jobs.values() if j.client_id == client_id and j.state in ("queued", "running"))

    def _active_dedup(self, client_id: str) -> dict[str, str]:
        out = {}
        for job in self.jobs.values():
            if job.client_id != client_id or job.state not in ("queued", "running"):
                continue
            req = self.requests.get(job.job_id)
            if not req:
                continue
            for item, res in zip(req.items, job.items):
                if res.state not in TERMINAL_ITEM_STATES:
                    out[dedup_key(client_id, item, [r.summary() for r in item.reasons or []])] = job.job_id
        return out

    def submit(self, req: JobRequest, client_id: str = "anonymous", idem_key: str | None = None) -> tuple[Job, bool]:
        """Trả (job, deduplicated). Cùng Idempotency-Key hoặc toàn bộ ad đang chạy ở 1 job khác => trả job cũ."""
        if idem_key and (old := self._idem.get((client_id, idem_key))) and old in self.jobs:
            return self.jobs[old], True
        active = self._active_dedup(client_id) if settings.dedup_active else {}
        items, dup_jobs, fresh = [], set(), 0
        for i in req.items:
            res = ItemResult(smart_ad_id=i.smart_ad_id, smart_ad_name=i.smart_ad_name, video_id=i.video_id,
                             state="pending" if i.status in req.process_statuses else "skipped")
            if res.state == "pending":
                other = active.get(dedup_key(client_id, i, [r.summary() for r in i.reasons or []]))
                if other:
                    res.state, res.duplicate_of = "skipped", other
                    res.error = f"ad này đang được xử lý ở job {other}"
                    dup_jobs.add(other)
                else:
                    fresh += 1
            items.append(res)
        if not fresh and len(dup_jobs) == 1:
            job = self.jobs[dup_jobs.pop()]
            if idem_key:
                self._idem[(client_id, idem_key)] = job.job_id
            return job, True
        job = Job(job_id=uuid.uuid4().hex, client_id=client_id, advertiser_id=req.advertiser_id,
                  campaign_id=req.campaign_id, campaign_name=req.campaign_name, callback_url=req.callback_url,
                  items=items, created_at=_now())
        self.jobs[job.job_id] = job
        self.requests[job.job_id] = req
        self._caches[job.job_id] = JobCache()
        if idem_key:
            self._idem[(client_id, idem_key)] = job.job_id
        self.save(job, req, idem_key)
        self.event(job, None, "submitted", f"{fresh} item cần xử lý / {len(items)}")
        self.counters["jobs_submitted"] += 1
        self._maybe_finish(job)
        self._wake.set()
        return job, False

    # -------------------------------------------------------------- scheduler
    def concurrency(self) -> int:
        base = detect().item_concurrency
        if self._oom_level and time.time() > self._oom_until:
            log.info("OOM cooldown over, concurrency back to %d", base)
            self._oom_level = 0
        return max(1, base >> self._oom_level)

    def _runnable(self, now: float) -> list[tuple[Job, int]]:
        out = []
        for job in self.jobs.values():
            if job.state not in ("queued", "running") or job.job_id not in self.requests:
                continue
            for idx, res in enumerate(job.items):
                if (job.job_id, idx) in self._running:
                    continue
                if res.state == "pending" or (res.state == "retrying" and (_ts(res.next_retry_at) or 0) <= now):
                    out.append((job, idx))
        return out

    def _pick(self, now: float) -> tuple[Job, int] | None:
        cands = self._runnable(now)
        if not cands:
            return None
        # xoay vòng giữa các client (client lâu chưa được phục vụ nhất trước), trong 1 client thì FIFO theo job
        clients = {j.client_id for j, _ in cands}
        client = min(clients, key=lambda c: self._served.get(c, 0.0))
        mine = [(j, i) for j, i in cands if j.client_id == client]
        return min(mine, key=lambda x: (_ts(x[0].created_at) or 0, x[1]))

    async def _scheduler(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            if self.paused:
                continue
            now = time.time()
            while len(self._running) < self.concurrency() and (pick := self._pick(now)):
                job, idx = pick
                self._served[job.client_id] = time.monotonic()
                ctx = RunContext(settings.stage_timeout_map())
                ctx.job_id, ctx.smart_ad_id = job.job_id, job.items[idx].smart_ad_id
                self._running[(job.job_id, idx)] = ctx
                task = asyncio.create_task(self._run_item(job, idx, ctx))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    def queue_position(self, job: Job) -> int | None:
        if not any(r.state in ("pending", "retrying") for r in job.items):
            return None
        created = _ts(job.created_at) or 0
        return sum(1 for j in self.jobs.values() if j.job_id != job.job_id and j.state in ("queued", "running")
                   and (_ts(j.created_at) or 0) <= created for r in j.items if r.state == "pending")

    # ------------------------------------------------------------- run 1 item
    async def _reasons(self, job: Job, req: JobRequest, item):
        if item.reasons is not None:
            return item.reasons
        lock = self._prepared.setdefault(job.job_id, asyncio.Lock())
        async with lock:
            if item.reasons is None:
                need = [i for i, r in zip(req.items, job.items) if i.reasons is None and r.state not in TERMINAL_ITEM_STATES]
                info: dict = {}
                try:
                    info = await fetch_reject_info(req.advertiser_id, [i.smart_ad_id for i in need])
                except Exception as exc:
                    self.event(job, None, "tiktok_error", str(exc))
                    raise NetworkError(f"TikTok review_info lỗi: {exc}") from exc
                for i in need:
                    vids = {v for v in (i.video_id, i.preview_video_id) if v}
                    i.reasons = reasons_for_video(info.get(i.smart_ad_id, []), vids)
                self.save(job, req)
        return item.reasons or []

    def _stage_update(self, job: Job, res: ItemResult, stage: str, timings: dict) -> None:
        if res.state != "running":
            return
        res.stage, res.timings = stage, timings
        self.save(job)

    async def _run_item(self, job: Job, idx: int, ctx: RunContext) -> None:
        req, res = self.requests[job.job_id], job.items[idx]
        item = req.items[idx]
        work_res = ItemResult(smart_ad_id=res.smart_ad_id, smart_ad_name=res.smart_ad_name, video_id=res.video_id)
        loop = asyncio.get_running_loop()
        ctx.on_stage = lambda st: loop.call_soon_threadsafe(self._stage_update, job, res, st, dict(work_res.timings))
        res.state, res.stage, res.next_retry_at = "running", None, None
        res.attempt += 1
        res.started_at = res.started_at or _now()
        if job.state == "queued":
            job.state, job.started_at = "running", _now()
        self.save(job)
        self.event(job, res.smart_ad_id, "attempt", f"lần chạy thứ {res.attempt}")
        out = self.outputs_dir / job.job_id / f"{item.smart_ad_id}.mp4"
        err = None
        t0 = time.perf_counter()
        try:
            reasons = await self._reasons(job, req, item)
            item.product = item.product or req.product
            item.visual_method = item.visual_method or req.visual_method or settings.visual_method
            cache = self._caches.setdefault(job.job_id, JobCache())
            await asyncio.to_thread(self._call, ctx, process_item, item, work_res, reasons,
                                    self.work_dir(job.job_id, item.smart_ad_id), out, req.campaign_name, cache)
        except Exception as exc:
            err = exc
        finally:
            self._running.pop((job.job_id, idx), None)
            self._wake.set()
        for stage, sec in work_res.timings.items():
            self.stage_seconds[stage] += sec
            self.stage_count[stage] += 1

        if res.state == "cancelled" or ctx.reason in ("cancelled", "shutdown"):
            if ctx.reason == "shutdown":  # server tắt giữa chừng: lần start sau chạy tiếp
                res.state = "pending"
                self.save(job)
                return
            self._finish_item(job, res, "cancelled")
            return
        if err is None:
            for f in RESULT_FIELDS:
                setattr(res, f, getattr(work_res, f))
            res.stage, res.error_code, res.retryable = None, None, None
            if res.state in ("done", "partial", "needs_review") and out.exists():
                res.output_video_url = settings.public_base_url.rstrip("/") + sign_file_path(f"/files/{job.job_id}/{out.name}")
            if res.state == "needs_review":
                res.error_code = "needs_review"
            elif res.state == "no_changes":
                decision = res.plan.get("decision")
                res.error_code = {"unfixable": "unfixable", "outside_only": "outside_video"}.get(decision, "no_changes")
            self._finish_item(job, res, res.state, time.perf_counter() - t0)
            return

        c = classify(err)
        log.log(logging.WARNING if c.retryable else logging.ERROR, "item %s attempt %d failed [%s]: %s",
                item.smart_ad_id, res.attempt, c.code, c.message, exc_info=c.code == "internal")
        self.recent_errors.append((time.time(), c.code))
        self.counters[f"error:{c.code}"] += 1
        for f in ("language", "reject_reasons", "usage", "timings"):
            setattr(res, f, getattr(work_res, f) or getattr(res, f))
        res.error, res.error_code = c.message[:1000], c.code
        if c.code == "gpu_oom":
            self._on_oom()
        delay = next_delay(c, res.attempt)
        if delay is not None:
            res.state, res.retryable = "retrying", True
            res.next_retry_at = datetime.fromtimestamp(time.time() + delay, timezone.utc)
            self.event(job, res.smart_ad_id, "retry_scheduled", f"[{c.code}] {c.message[:300]}",
                       {"delay": round(delay, 1), "attempt": res.attempt})
            self.save(job)
            return
        res.retryable = c.code in MANUAL_RETRY_CODES
        self._finish_item(job, res, "failed", time.perf_counter() - t0, f"[{c.code}] {c.message[:300]}")

    @staticmethod
    def _call(ctx: RunContext, fn, *args):
        token = runctx.bind(ctx)
        try:
            return fn(*args)
        finally:
            runctx._current.reset(token)

    def _on_oom(self) -> None:
        self._oom_level = min(self._oom_level + 1, 4)
        self._oom_until = time.time() + settings.oom_cooldown_minutes * 60
        log.warning("GPU OOM: concurrency giảm còn %d trong %.0f phút", self.concurrency(), settings.oom_cooldown_minutes)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _finish_item(self, job: Job, res: ItemResult, state: str, seconds: float | None = None, message: str = "") -> None:
        res.state, res.stage, res.next_retry_at, res.finished_at = state, None, None, _now()
        if state == "cancelled":
            res.error_code, res.retryable, res.error = "cancelled", True, "đã huỷ theo yêu cầu"
        self.counters[f"items:{state}"] += 1
        self.event(job, res.smart_ad_id, state, message or (res.error or "")[:300],
                   {"seconds": round(seconds, 2)} if seconds else None)
        self._enqueue_webhook(job, "item.done", res)
        self._maybe_finish(job)
        self.save(job)

    def _maybe_finish(self, job: Job) -> None:
        if job.state not in ("queued", "running") or any(r.state not in TERMINAL_ITEM_STATES for r in job.items):
            return
        if any((job.job_id, i) in self._running for i in range(len(job.items))):
            return
        job.cache_hits = self._caches.pop(job.job_id, JobCache()).hits or job.cache_hits
        job.state = "cancelled" if job.job_id in self._cancel_requested else "done"
        job.finished_at = _now()
        self._cancel_requested.discard(job.job_id)
        self._prepared.pop(job.job_id, None)
        self.counters[f"jobs:{job.state}"] += 1
        self.event(job, None, f"job_{job.state}", json.dumps(self.progress(job)))
        self.save(job)
        self._enqueue_webhook(job, "job.done")

    # ------------------------------------------------------- cancel / retry
    def cancel(self, job: Job, smart_ad_ids: list[str] | None) -> list[str]:
        """Huỷ item chưa xong. Item đang chạy bị kill (ffmpeg kill ngay; bước đang gọi OpenAI/Demucs dừng khi trả về)."""
        done = []
        if smart_ad_ids is None:
            self._cancel_requested.add(job.job_id)
        for idx, res in enumerate(job.items):
            if smart_ad_ids is not None and res.smart_ad_id not in smart_ad_ids:
                continue
            if res.state in TERMINAL_ITEM_STATES:
                continue
            if ctx := self._running.get((job.job_id, idx)):
                ctx.kill("cancelled")
            done.append(res.smart_ad_id)
            res.error = "đã huỷ theo yêu cầu"
            res.error_code = "cancelled"
            if (job.job_id, idx) in self._running:
                res.state, res.stage = "cancelled", "killing"
                self.event(job, res.smart_ad_id, "kill", "đang dừng item đang chạy")
            else:
                self._finish_item(job, res, "cancelled")
        self._maybe_finish(job)
        self.save(job)
        return done

    def retry(self, job: Job, body: RetryRequest) -> list[str]:
        req = self.requests.get(job.job_id)
        if req is None:
            row = self.store._all("SELECT request FROM jobs WHERE job_id=?", (job.job_id,))
            if not row or not row[0]["request"]:
                raise ValueError("job không còn request gốc, không retry được")
            req = self.requests[job.job_id] = JobRequest.model_validate_json(row[0]["request"])
        targets = []
        for idx, (item, res) in enumerate(zip(req.items, job.items)):
            if (job.job_id, idx) in self._running or res.state in ("pending", "running", "retrying"):
                continue
            if body.smart_ad_ids is None:
                # retry-all: mặc định chỉ item failed/cancelled; force=true thì cả item đã xong (done/partial/...)
                if res.state not in ("failed", "cancelled") and not body.force:
                    continue
            elif res.smart_ad_id not in body.smart_ad_ids:
                continue
            elif res.state in ("done", "partial", "needs_review", "no_changes", "skipped") and not body.force:
                continue
            work = self.work_dir(job.job_id, item.smart_ad_id)
            if url := body.video_urls.get(res.smart_ad_id):
                item.video_url = url
                shutil.rmtree(work, ignore_errors=True)  # video khác => mọi file trung gian không còn đúng
            if res.smart_ad_id in body.reasons:
                item.reasons = body.reasons[res.smart_ad_id]
            if body.product:
                item.product = body.product
            if body.visual_method:
                item.visual_method = body.visual_method
            if body.ad_text is not None:
                item.ad_text = body.ad_text
            if body.landing_page_url is not None:
                item.landing_page_url = body.landing_page_url
            if res.smart_ad_id in body.reasons or body.product or res.state in ("done", "partial", "needs_review",
                                                                                 "no_changes"):
                for f in ("triage.json", "plan_raw.json", "rewrite.json", "policy_groups.json"):
                    (work / f).unlink(missing_ok=True)
                shutil.rmtree(work / "pass2", ignore_errors=True)
                for f in work.glob("tts_*.wav") if work.exists() else []:
                    f.unlink(missing_ok=True)
            if res.state == "skipped" and item.status not in req.process_statuses:
                req.process_statuses = [*req.process_statuses, item.status]
            job.items[idx] = ItemResult(smart_ad_id=res.smart_ad_id, smart_ad_name=res.smart_ad_name,
                                        video_id=res.video_id, state="pending")
            targets.append(res.smart_ad_id)
            self.event(job, res.smart_ad_id, "retry_requested", "", {"new_video_url": bool(body.video_urls.get(res.smart_ad_id))})
        if targets:
            job.state = "running" if job.started_at else "queued"
            job.finished_at = None
            self._cancel_requested.discard(job.job_id)
            self._caches[job.job_id] = JobCache()
            self.save(job, req)
            self._wake.set()
        return targets

    # ---------------------------------------------------------- status views
    def progress(self, job: Job) -> dict[str, int]:
        c = Counter(r.state for r in job.items)
        return {"total": len(job.items), "finished": sum(c[s] for s in TERMINAL_ITEM_STATES), **c}

    def view(self, job: Job) -> Job:
        v = job.model_copy(deep=True)
        v.progress = self.progress(job)
        v.queue_position = self.queue_position(job) if job.state in ("queued", "running") else None
        # Ký lại link video mỗi lần xem: link ký lúc item xong hết hạn sau file_url_ttl_hours,
        # dashboard cần link còn dùng được khi xem lại job cũ
        for res in v.items:
            if res.output_video_url and (out := self.outputs_dir / job.job_id / f"{res.smart_ad_id}.mp4").exists():
                res.output_video_url = settings.public_base_url.rstrip("/") + sign_file_path(f"/files/{job.job_id}/{out.name}")
        return v

    # ---------------------------------------------------------------- webhook
    def _enqueue_webhook(self, job: Job, kind: str, res: ItemResult | None = None) -> None:
        if not job.callback_url or (kind == "item.done" and not settings.webhook_item_events):
            return
        event_id = uuid.uuid4().hex
        if kind == "job.done":
            payload = json.loads(self.view(job).model_dump_json(exclude={"client_id"}))
        else:
            payload = {"job_id": job.job_id, "advertiser_id": job.advertiser_id, "campaign_id": job.campaign_id,
                       "campaign_name": job.campaign_name, "progress": self.progress(job),
                       "item": json.loads(res.model_dump_json())}
        payload = {"event": kind, "event_id": event_id, **payload}
        self.store.add_webhook(event_id, job.job_id, kind, job.callback_url,
                               json.dumps(payload, ensure_ascii=False).encode())
        if kind == "job.done":
            job.callback_status = "pending"

    def resend_webhook(self, job: Job) -> int:
        n = self.store.requeue_webhooks(job.job_id, "job.done")
        if not n:
            self._enqueue_webhook(job, "job.done")
            n = 1
        return n

    async def _webhook_sender(self) -> None:
        delays = settings.webhook_delays()
        async with httpx.AsyncClient(timeout=settings.webhook_timeout, follow_redirects=False) as client:
            while True:
                await asyncio.sleep(2)
                for row in self.store.due_webhooks(time.time()):
                    await self._deliver(client, row, delays)

    async def _deliver(self, client: httpx.AsyncClient, row, delays: list[float]) -> None:
        attempts = row["attempts"] + 1
        job = self.jobs.get(row["job_id"])
        try:
            # Check lại lúc gửi: DNS của callback có thể đã đổi sang IP nội bộ sau khi submit
            await asyncio.to_thread(check_public_url, row["url"], settings.callback_url_allowlist)
        except ValueError as exc:
            self.store.update_webhook(row["event_id"], attempts, "failed", 0, f"blocked: {exc}")
            if job and row["kind"] == "job.done":
                job.callback_status = f"blocked: {exc}"
                self.save(job)
            return
        body = bytes(row["body"])
        headers = {"Content-Type": "application/json", "X-Magician-Event": row["kind"],
                   "X-Magician-Event-Id": row["event_id"], "X-Magician-Delivery-Attempt": str(attempts),
                   **webhook_headers(body)}
        try:
            r = await client.post(row["url"], content=body, headers=headers)
            status, ok = f"HTTP {r.status_code}", r.is_success
        except httpx.HTTPError as exc:
            status, ok = f"error: {exc.__class__.__name__}: {exc}"[:300], False
        if ok:
            state, next_at = "delivered", 0
        elif attempts > len(delays):
            state, next_at = "failed", 0
        else:
            state, next_at = "pending", time.time() + delays[attempts - 1]
        self.store.update_webhook(row["event_id"], attempts, state, next_at, status)
        self.counters[f"webhook:{state}"] += 1
        if not ok:
            log.warning("webhook %s %s attempt %d: %s", row["kind"], row["job_id"], attempts, status)
        if job and row["kind"] == "job.done":
            job.callback_status = status if state != "pending" else f"{status}, retry {attempts}/{len(delays)}"
            self.save(job)

    # ------------------------------------------------------------ maintenance
    def disk_free_gb(self) -> float:
        return shutil.disk_usage(settings.data_dir).free / 1e9

    async def _maintenance(self) -> None:
        from .tts import pool

        tick = 0
        while True:
            await asyncio.sleep(2)
            tick += 1
            # stage timeout: kill subprocess ngay cả khi thread đang kẹt ở chỗ không gọi check()
            for ctx in list(self._running.values()):
                if not ctx.event.is_set() and ctx.expired():
                    log.warning("item %s: stage %s timeout, killing", ctx.smart_ad_id, ctx.stage)
                    ctx.kill(f"timeout:{ctx.stage}")
            if tick % 15 == 0:
                free = self.disk_free_gb()
                paused = f"ổ đĩa còn {free:.1f}GB < {settings.min_free_gb}GB" if free < settings.min_free_gb else None
                if paused != self.paused:
                    (log.error if paused else log.info)("scheduler %s", paused or "resumed (disk ok)")
                    self.paused = paused
                    self._wake.set()
                if settings.tts_warmup and (self._health_future is None or self._health_future.done()):
                    self._health_future = asyncio.get_running_loop().run_in_executor(None, pool.check_health)
            if tick % 1800 == 1:
                try:
                    await asyncio.to_thread(self.cleanup)
                except Exception:
                    log.exception("cleanup failed")

    def cleanup(self) -> dict[str, int]:
        now = time.time()
        stats = Counter()
        for job in list(self.jobs.values()):
            fin = _ts(job.finished_at)
            if not fin:
                continue
            work = self.jobs_dir / job.job_id
            if now - fin > settings.work_retention_hours * 3600 and work.exists():
                shutil.rmtree(work, ignore_errors=True)
                stats["work_dirs"] += 1
            out = self.outputs_dir / job.job_id
            if now - fin > settings.output_retention_days * 86400 and out.exists():
                shutil.rmtree(out, ignore_errors=True)
                stats["outputs"] += 1
        for jid in self.store.purge_jobs(now - settings.job_retention_days * 86400):
            self.jobs.pop(jid, None)
            self.requests.pop(jid, None)
            stats["jobs"] += 1
        if stats:
            log.info("cleanup: %s", dict(stats))
        return dict(stats)

    # ---------------------------------------------------------------- metrics
    def stats(self) -> dict:
        hour = time.time() - 3600
        states = Counter(r.state for j in self.jobs.values() if j.state in ("queued", "running") for r in j.items)
        return {
            "uptime_s": round(time.time() - self.started_at),
            "paused": self.paused,
            "concurrency": self.concurrency(),
            "running": len(self._running),
            "pending": states["pending"],
            "retrying": states["retrying"],
            "active_jobs": sum(1 for j in self.jobs.values() if j.state in ("queued", "running")),
            "errors_last_hour": dict(Counter(code for ts, code in self.recent_errors if ts > hour)),
            "webhooks": self.store.webhook_stats(),
            "disk_free_gb": round(self.disk_free_gb(), 1),
        }

    def metrics(self) -> str:
        from .tts import pool

        s = self.stats()
        lines = [
            f"magician_up 1", f"magician_uptime_seconds {s['uptime_s']}",
            f"magician_paused {int(bool(s['paused']))}", f"magician_concurrency {s['concurrency']}",
            f"magician_items_running {s['running']}", f"magician_items_pending {s['pending']}",
            f"magician_items_retrying {s['retrying']}", f"magician_jobs_active {s['active_jobs']}",
            f"magician_disk_free_gb {s['disk_free_gb']}", f"magician_tts_workers_alive {pool.alive_count()}",
        ]
        for k, v in sorted(self.counters.items()):
            kind, _, label = k.partition(":")
            if label:
                name = {"items": "items_finished_total", "jobs": "jobs_finished_total", "error": "item_errors_total",
                        "webhook": "webhook_deliveries_total"}[kind]
                key = {"items": "state", "jobs": "state", "error": "code", "webhook": "state"}[kind]
                lines.append(f'magician_{name}{{{key}="{label}"}} {v}')
            else:
                lines.append(f"magician_{kind}_total {v}")
        for st in sorted(self.stage_seconds):
            lines.append(f'magician_stage_seconds_sum{{stage="{st}"}} {self.stage_seconds[st]:.2f}')
            lines.append(f'magician_stage_seconds_count{{stage="{st}"}} {self.stage_count[st]}')
        for state, n in s["webhooks"].items():
            lines.append(f'magician_webhooks{{state="{state}"}} {n}')
        return "\n".join(lines) + "\n"


manager = JobManager()
