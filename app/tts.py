"""Pool worker TTS (OmniVoice hoặc XTTS-v2). Mỗi worker là 1 subprocess riêng nói chuyện qua JSON-lines,
nên có thể chạy bằng Python env khác server (vd XTTS cần transformers<5, OmniVoice cần transformers>=5)."""
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import runctx
from .config import settings
from .errors import WorkerCrash
from .hardware import detect, has_room_for_worker

log = logging.getLogger(__name__)

WORKER_DIR = Path(__file__).parent / "tts_workers"
XTTS_LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"}
# Số giây giọng mẫu tối ưu cho từng engine
REF_SECONDS = {"omnivoice": 10.0, "xtts": 20.0}


def engine() -> str:
    return settings.tts_engine.lower()


def supports_duration() -> bool:
    """OmniVoice sinh được audio đúng độ dài yêu cầu; XTTS thì phải atempo."""
    return engine() == "omnivoice"


def engine_lang(lang: str) -> str:
    code = lang.lower()
    if engine() == "omnivoice":
        overrides = dict(x.split(":", 1) for x in settings.omnivoice_lang_map.split(",") if ":" in x)
        return overrides.get(code, code.split("-")[0])
    code = code if code in XTTS_LANGS else code.split("-")[0]
    if code not in XTTS_LANGS:
        raise ValueError(f"XTTS không hỗ trợ ngôn ngữ: {lang}")
    return code


class Worker:
    def __init__(self, device: str, threads: int) -> None:
        eng = engine()
        if eng == "xtts":
            if not settings.coqui_tos_agreed:
                raise RuntimeError("XTTS-v2 dùng license CPML: đặt COQUI_TOS_AGREED=1 sau khi đã đọc license")
            python, args = settings.xtts_python or sys.executable, ["--model", settings.xtts_model]
        else:
            python = settings.omnivoice_python or sys.executable
            args = ["--model", settings.omnivoice_model, "--num-step", str(settings.omnivoice_num_step)]
        cmd = [python, str(WORKER_DIR / f"{eng}_worker.py"), "--device", device, "--threads", str(threads), *args]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        self.device = device
        self.calls = 0
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env)
        # Đọc stdout ở thread riêng để call() có timeout và kiểm tra được lệnh huỷ
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._read(600)
        if not ready.get("ready"):
            raise WorkerCrash(f"TTS worker không khởi động được: {ready}")

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._lines.get(timeout=0.5)
                break
            except queue.Empty:
                if time.monotonic() > deadline:
                    self.kill()
                    raise WorkerCrash(f"TTS worker trên {self.device} không phản hồi sau {timeout:.0f}s, đã kill")
        if line is None:
            raise WorkerCrash(f"TTS worker trên {self.device} đã thoát (exit code {self.proc.poll()})")
        return json.loads(line)

    def call(self, req: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise WorkerCrash(f"TTS worker trên {self.device} đã thoát: {e}") from e
        self.calls += 1
        return self._read(settings.tts_call_timeout)

    def kill(self) -> None:
        try:
            self.proc.kill()
        except OSError:
            pass

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.alive:
            self.proc.stdin.close()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class TTSPool:
    """Tạo thêm worker khi tất cả đang bận, tối đa theo `tts_slots` và chỉ khi VRAM/RAM lúc đó còn đủ."""

    def __init__(self) -> None:
        self._free: queue.Queue[Worker] = queue.Queue()
        self._lock = threading.Lock()
        # Load model tuần tự để RAM không bị spike
        self._spawn_lock = threading.Lock()
        self.loaded: list[str] = []
        self._workers: list[Worker] = []

    def _reserve_slot(self) -> str | None:
        with self._lock:
            slots = detect().tts_slots
            if len(self.loaded) >= len(slots):
                return None
            dev = slots[len(self.loaded)]
            if self.loaded and not has_room_for_worker(dev):
                return None
            self.loaded.append(dev)
            return dev

    def _spawn(self, device: str) -> Worker:
        hw = detect()
        threads = max(1, hw.cpu_cores // max(1, len(hw.tts_slots)))
        with self._spawn_lock:
            log.info("starting %s worker on %s", engine(), device)
            worker = Worker(device, threads)
        with self._lock:
            self._workers.append(worker)
        return worker

    def _release_slot(self, device: str) -> None:
        with self._lock:
            self.loaded.remove(device)

    @contextmanager
    def acquire(self):
        try:
            worker = self._free.get_nowait()
        except queue.Empty:
            dev = self._reserve_slot()
            if dev is None:
                worker = self._free.get()
            else:
                try:
                    worker = self._spawn(dev)
                except Exception:
                    self._release_slot(dev)
                    raise
        try:
            yield worker
        finally:
            if worker.alive:
                self._free.put(worker)
            else:
                log.error("TTS worker on %s died, slot freed for respawn", worker.device)
                with self._lock:
                    if worker in self._workers:
                        self._workers.remove(worker)
                self._release_slot(worker.device)

    def warmup(self) -> None:
        """Khởi động sẵn toàn bộ worker lúc startup."""
        while (dev := self._reserve_slot()) is not None:
            try:
                self._free.put(self._spawn(dev))
            except Exception:
                log.exception("TTS worker on %s failed to start", dev)
                self._release_slot(dev)
                break

    def shutdown(self) -> None:
        for w in self._workers:
            w.close()

    def check_health(self) -> int:
        """Watchdog: bỏ worker rảnh đã chết khỏi pool, giải phóng slot rồi spawn lại. Trả số worker đã thay."""
        dead = []
        alive = []
        while True:
            try:
                w = self._free.get_nowait()
            except queue.Empty:
                break
            (alive if w.alive else dead).append(w)
        for w in alive:
            self._free.put(w)
        for w in dead:
            log.error("watchdog: TTS worker on %s died (exit %s), respawning", w.device, w.proc.poll())
            with self._lock:
                if w in self._workers:
                    self._workers.remove(w)
            self._release_slot(w.device)
        if dead or not self.loaded:
            self.warmup()
        return len(dead)

    def alive_count(self) -> int:
        with self._lock:
            return sum(w.alive for w in self._workers)

    def stats(self) -> dict:
        with self._lock:
            workers = [{"device": w.device, "alive": w.alive, "calls": w.calls} for w in self._workers]
        return {"engine": engine(), "loaded": list(self.loaded), "idle": self._free.qsize(), "workers": workers}


pool = TTSPool()


def synthesize(
    text: str, language: str, speaker_wavs: list[Path], out: Path, ref_text: str | None = None, duration: float | None = None
) -> Path:
    """Clone giọng từ speaker_wavs và đọc `text`. `duration` (giây) chỉ có tác dụng với OmniVoice."""
    if out.exists():  # retry: câu này đã sinh xong ở lần chạy trước
        return out
    part = out.with_name(out.stem + ".part" + out.suffix)
    req = {"text": text, "language": engine_lang(language), "ref_wav": [str(p) for p in speaker_wavs],
           "ref_text": ref_text, "out": str(part), "duration": duration if supports_duration() else None}
    runctx.check()
    with pool.acquire() as worker:
        res = worker.call(req)
    if not res.get("ok"):
        raise RuntimeError(f"TTS lỗi: {res.get('error')}")
    part.replace(out)
    return out
