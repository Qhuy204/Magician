import asyncio
import json
import logging
import uuid

import httpx

from .config import settings
from .hardware import detect
from .pipeline import JobCache, process_item
from .security import check_public_url, sign_file_path, webhook_headers
from .schemas import Job, JobRequest, ItemResult
from .tiktok import fetch_reject_info, reasons_for_video

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.requests: dict[str, JobRequest] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    @property
    def jobs_dir(self):
        return settings.data_dir / "jobs"

    @property
    def outputs_dir(self):
        return settings.data_dir / "outputs"

    def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        for f in self.jobs_dir.glob("*/job.json"):
            job = Job.model_validate_json(f.read_text(encoding="utf-8"))
            if job.state in ("queued", "running"):
                job.state, job.callback_status = "failed", "interrupted by server restart"
            self.jobs[job.job_id] = job
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()

    def save(self, job: Job) -> None:
        d = self.jobs_dir / job.job_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def active_jobs(self, client_id: str) -> int:
        return sum(1 for j in self.jobs.values() if j.client_id == client_id and j.state in ("queued", "running"))

    def submit(self, req: JobRequest, client_id: str = "anonymous") -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            client_id=client_id,
            advertiser_id=req.advertiser_id,
            campaign_id=req.campaign_id,
            campaign_name=req.campaign_name,
            callback_url=req.callback_url,
            items=[
                ItemResult(
                    smart_ad_id=i.smart_ad_id, smart_ad_name=i.smart_ad_name, video_id=i.video_id,
                    state="pending" if i.status in req.process_statuses else "skipped",
                )
                for i in req.items
            ],
        )
        self.jobs[job.job_id] = job
        self.requests[job.job_id] = req
        self.save(job)
        self.queue.put_nowait(job.job_id)
        return job

    async def _run(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._process(self.jobs[job_id], self.requests.pop(job_id))
            except Exception as exc:
                log.exception("job %s failed", job_id)
                self.jobs[job_id].state = "failed"
                self.jobs[job_id].callback_status = f"job error: {exc}"
            self.save(self.jobs[job_id])
            await self._callback(self.jobs[job_id])

    async def _process(self, job: Job, req: JobRequest) -> None:
        job.state = "running"
        self.save(job)
        todo = [(item, res) for item, res in zip(req.items, job.items) if res.state == "pending"]

        need_api = [item.smart_ad_id for item, _ in todo if item.reasons is None]
        reject_info: dict[str, list[dict]] = {}
        if need_api:
            try:
                reject_info = await fetch_reject_info(req.advertiser_id, need_api)
            except Exception as exc:
                log.warning("TikTok review_info failed: %s", exc)

        sem = asyncio.Semaphore(detect().item_concurrency)
        cache = JobCache()

        async def one(item, res: ItemResult) -> None:
            async with sem:
                res.state = "running"
                self.save(job)
                reasons = item.reasons
                if reasons is None:
                    vids = {v for v in (item.video_id, item.preview_video_id) if v}
                    reasons = reasons_for_video(reject_info.get(item.smart_ad_id, []), vids)
                out = self.outputs_dir / job.job_id / f"{item.smart_ad_id}.mp4"
                try:
                    await asyncio.to_thread(
                        process_item, item, res, reasons, self.jobs_dir / job.job_id / item.smart_ad_id, out, req.campaign_name, cache
                    )
                    if res.state == "done":
                        res.output_video_url = settings.public_base_url.rstrip("/") + sign_file_path(f"/files/{job.job_id}/{out.name}")
                except Exception as exc:
                    log.exception("item %s failed", item.smart_ad_id)
                    res.state, res.error = "failed", str(exc)
                self.save(job)

        await asyncio.gather(*(one(i, r) for i, r in todo))
        job.cache_hits = cache.hits
        job.state = "done"

    async def _callback(self, job: Job) -> None:
        if not job.callback_url:
            return
        body = job.model_dump_json(exclude={"client_id"}).encode()
        try:
            # Check lại lúc gửi: DNS của callback có thể đã đổi sang IP nội bộ sau khi submit
            await asyncio.to_thread(check_public_url, job.callback_url, settings.callback_url_allowlist)
        except ValueError as exc:
            job.callback_status = f"blocked: {exc}"
            self.save(job)
            return
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            for attempt in range(3):
                try:
                    headers = {"Content-Type": "application/json", **webhook_headers(body)}
                    r = await client.post(job.callback_url, content=body, headers=headers)
                    job.callback_status = f"HTTP {r.status_code}"
                    if r.is_success:
                        break
                except httpx.HTTPError as exc:
                    job.callback_status = f"error: {exc}"
                if attempt < 2:
                    await asyncio.sleep(2**attempt * 5)
        self.save(job)


manager = JobManager()
