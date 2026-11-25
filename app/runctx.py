"""Ngữ cảnh chạy của 1 item: theo dõi stage hiện tại, timeout từng stage và lệnh kill (cancel).

Pipeline chạy trong thread nên không kill cứng được; thay vào đó mọi chỗ chạy lâu (tải video, ffmpeg, OCR, TTS)
kiểm tra `check()` thường xuyên, còn subprocess (ffmpeg) đăng ký vào ctx để bị kill ngay khi cancel / timeout.
"""
import contextvars
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Callable

from .errors import Cancelled, StageTimeout


class RunContext:
    def __init__(self, timeouts: dict[str, float] | None = None, on_stage: Callable[[str], None] | None = None) -> None:
        self.timeouts = timeouts or {}
        self.on_stage = on_stage
        self.stage: str | None = None
        self.deadline: float | None = None
        self.reason: str | None = None  # "cancelled" | "timeout:<stage>"
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen] = set()
        self._hooks: list[Callable[[], None]] = []

    # --- điều khiển từ bên ngoài (scheduler / API) ---
    def kill(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
            self.event.set()
            procs, hooks = list(self._procs), list(self._hooks)
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass
        for h in hooks:
            try:
                h()
            except Exception:
                pass

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    # --- dùng bên trong pipeline ---
    def check(self) -> None:
        if not self.event.is_set() and self.expired():
            self.kill(f"timeout:{self.stage}")
        if self.event.is_set():
            if self.reason and self.reason.startswith("timeout:"):
                raise StageTimeout(f"stage {self.reason[8:]} quá {self.timeouts.get(self.reason[8:], '?')}s")
            raise Cancelled("đã bị huỷ")

    @contextmanager
    def stage_scope(self, name: str):
        """Stage lồng nhau (vd tts trong rewrite) giữ deadline của stage ngoài nếu ngắn hơn."""
        prev_stage, prev_deadline = self.stage, self.deadline
        self.check()
        self.stage = name
        if (limit := self.timeouts.get(name)) and limit > 0:
            d = time.monotonic() + limit
            self.deadline = d if prev_deadline is None else min(d, prev_deadline)
        if self.on_stage:
            self.on_stage(name)
        try:
            yield
            self.check()
        finally:
            self.stage, self.deadline = prev_stage, prev_deadline

    def on_kill(self, hook: Callable[[], None]) -> None:
        with self._lock:
            self._hooks.append(hook)
            fire = self.event.is_set()
        if fire:
            hook()

    def register(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.add(proc)
            killed = self.event.is_set()
        if killed:
            proc.kill()

    def unregister(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(proc)


_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("magician_run_ctx", default=None)


def current() -> RunContext | None:
    return _current.get()


def bind(ctx: RunContext | None):
    return _current.set(ctx)


def check() -> None:
    if ctx := current():
        ctx.check()


@contextmanager
def stage(name: str):
    ctx = current()
    if ctx is None:
        yield
        return
    with ctx.stage_scope(name):
        yield


@contextmanager
def tracked(proc: subprocess.Popen):
    """Đăng ký subprocess để bị kill khi item bị huỷ / quá timeout."""
    ctx = current()
    if ctx:
        ctx.register(proc)
    try:
        yield proc
    finally:
        if ctx:
            ctx.unregister(proc)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """subprocess.run nhưng kill được qua ctx. Nhận capture_output/text/check như subprocess.run."""
    check_rc = kw.pop("check", False)
    if kw.pop("capture_output", False):
        kw["stdout"], kw["stderr"] = subprocess.PIPE, subprocess.PIPE
    proc = subprocess.Popen(cmd, **kw)
    with tracked(proc):
        out, err = proc.communicate()
    check()
    res = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check_rc:
        res.check_returncode()
    return res
