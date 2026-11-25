"""Pool worker TTS (OmniVoice hoặc XTTS-v2). Mỗi worker là 1 subprocess riêng nói chuyện qua JSON-lines,
nên có thể chạy bằng Python env khác server (vd XTTS cần transformers<5, OmniVoice cần transformers>=5)."""
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from .config import settings
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
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env)
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"TTS worker không khởi động được: {ready}")

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"TTS worker trên {self.device} đã thoát (exit code {self.proc.poll()})")
        return json.loads(line)

    def call(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._read()

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

    def stats(self) -> dict:
        return {"engine": engine(), "loaded": list(self.loaded), "idle": self._free.qsize()}


pool = TTSPool()


def synthesize(
    text: str, language: str, speaker_wavs: list[Path], out: Path, ref_text: str | None = None, duration: float | None = None
) -> Path:
    """Clone giọng từ speaker_wavs và đọc `text`. `duration` (giây) chỉ có tác dụng với OmniVoice."""
    req = {"text": text, "language": engine_lang(language), "ref_wav": [str(p) for p in speaker_wavs],
           "ref_text": ref_text, "out": str(out), "duration": duration if supports_duration() else None}
    with pool.acquire() as worker:
        res = worker.call(req)
    if not res.get("ok"):
        raise RuntimeError(f"TTS lỗi: {res.get('error')}")
    return out
