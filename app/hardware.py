"""Nhận diện GPU/VRAM/RAM và tính số instance model chạy song song."""
import logging
import os
from dataclasses import asdict, dataclass, field
from functools import cache

import psutil
import torch

from .config import settings

log = logging.getLogger(__name__)

GB = 1024**3
DEMUCS_VRAM_GB = 1.0
VRAM_MARGIN_GB = 0.8
RAM_RESERVE_GB = 3.0


@dataclass(frozen=True)
class WorkerProfile:
    vram_gb: float  # VRAM mỗi worker (weights + peak lúc sinh + CUDA context)
    ram_gb: float  # RSS host khi weights trên GPU
    cpu_ram_gb: float  # RAM khi chạy CPU / unified memory (MPS)
    load_ram_gb: float  # RAM tạm lúc load checkpoint (worker load tuần tự)
    win_commit_gb: float  # Windows commit charge mỗi worker (CUDA DLL + workspace), giới hạn bởi RAM + pagefile


# Đo trên RTX 3060 / Windows 11:
# OmniVoice fp16: weights 1.95GB, peak +0.15–0.6GB, RSS 2.2GB, commit 8.8GB
# XTTS-v2: weights 1.8GB, peak +0.2GB, RSS 0.8GB, commit 3.7GB, spike ~4GB lúc load
PROFILES = {
    "omnivoice": WorkerProfile(vram_gb=3.0, ram_gb=2.5, cpu_ram_gb=4.0, load_ram_gb=2.0, win_commit_gb=9.0),
    "xtts": WorkerProfile(vram_gb=2.6, ram_gb=1.0, cpu_ram_gb=3.0, load_ram_gb=4.5, win_commit_gb=4.0),
}


def profile() -> WorkerProfile:
    return PROFILES.get(settings.tts_engine.lower(), PROFILES["omnivoice"])


def host_available_gb() -> float:
    """RAM khả dụng cho worker mới. Windows giới hạn theo commit (RAM + pagefile) chứ không chỉ RAM vật lý."""
    phys = psutil.virtual_memory().available / GB
    if os.name != "nt":
        return phys
    import ctypes

    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_commit", "avail_commit",
                                              "total_virtual", "avail_virtual", "avail_ext_virtual")]

    st = _MemStatus()
    st.dwLength = ctypes.sizeof(st)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.avail_commit / GB


def _ram_per_worker(on_gpu: bool) -> tuple[float, float]:
    """(RAM cần mỗi worker, RAM tạm lúc load) theo cách hệ điều hành tính."""
    p = profile()
    if os.name == "nt":
        return p.win_commit_gb, 0.0
    return (p.ram_gb if on_gpu else p.cpu_ram_gb), p.load_ram_gb


@dataclass
class Hardware:
    backend: str
    devices: list[str]
    gpu_names: list[str] = field(default_factory=list)
    vram_total_gb: list[float] = field(default_factory=list)
    vram_free_gb: list[float] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    cpu_cores: int = 1
    demucs_device: str = "cpu"
    tts_engine: str = ""
    tts_slots: list[str] = field(default_factory=list)
    item_concurrency: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


def _backend(requested: str) -> tuple[str, list[int] | None]:
    req = requested.lower()
    base, _, idx = req.partition(":")
    only = [int(idx)] if idx else None
    if base == "auto":
        if torch.cuda.is_available():
            return ("rocm" if torch.version.hip else "cuda"), None
        if torch.backends.mps.is_available():
            return "mps", None
        return "cpu", None
    if base in ("cuda", "rocm") and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={requested} nhưng torch không thấy GPU (torch build: {torch.__version__})")
    if base == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DEVICE=mps nhưng MPS không khả dụng")
    return base, only


def has_room_for_worker(device: str) -> bool:
    """Còn đủ VRAM + RAM để bật thêm 1 worker TTS trên `device` lúc này không."""
    on_gpu = device.startswith("cuda")
    per, load = _ram_per_worker(on_gpu)
    if host_available_gb() - RAM_RESERVE_GB < per + load:
        return False
    if on_gpu:
        return torch.cuda.mem_get_info(torch.device(device).index or 0)[0] / GB >= profile().vram_gb + VRAM_MARGIN_GB
    return True


def _round_robin(per_device: dict[str, int], total: int) -> list[str]:
    slots, left = [], dict(per_device)
    while len(slots) < total and any(left.values()):
        for dev, n in left.items():
            if n and len(slots) < total:
                slots.append(dev)
                left[dev] -= 1
    return slots


@cache
def detect() -> Hardware:
    backend, only = _backend(settings.device)
    vm = psutil.virtual_memory()
    ram_avail = host_available_gb()
    prof = profile()
    hw = Hardware(
        backend=backend, devices=[], ram_total_gb=round(vm.total / GB, 1), ram_available_gb=round(ram_avail, 1),
        cpu_cores=os.cpu_count() or 1, tts_engine=settings.tts_engine.lower(),
    )

    def ram_cap(on_gpu: bool) -> int:
        per, load = _ram_per_worker(on_gpu)
        return max(1, int((ram_avail - RAM_RESERVE_GB - load) // per))

    if backend in ("cuda", "rocm"):
        ids = only or list(range(torch.cuda.device_count()))
        per_device = {}
        for i in ids:
            free, total = (x / GB for x in torch.cuda.mem_get_info(i))
            dev = f"cuda:{i}"
            hw.devices.append(dev)
            hw.gpu_names.append(torch.cuda.get_device_name(i))
            hw.vram_total_gb.append(round(total, 1))
            hw.vram_free_gb.append(round(free, 1))
            budget = free - VRAM_MARGIN_GB - (DEMUCS_VRAM_GB if i == ids[0] else 0)
            per_device[dev] = max(0, min(settings.max_tts_per_gpu, int(budget // prof.vram_gb)))
        hw.demucs_device = hw.devices[0]
        auto_total = min(sum(per_device.values()), ram_cap(True))
        if not auto_total:
            log.warning("VRAM trống không đủ cho TTS worker (%s GB), vẫn thử 1 worker", hw.vram_free_gb)
            per_device[hw.devices[0]], auto_total = 1, 1
        total = settings.tts_workers or auto_total
        hw.tts_slots = _round_robin(per_device if not settings.tts_workers else {d: total for d in hw.devices}, total)
    elif backend == "mps":
        # Unified memory: GPU dùng chung RAM; MPS chỉ có 1 GPU nên 2 instance là đủ bão hoà
        rec = getattr(torch.mps, "recommended_max_memory", lambda: vm.total * 0.65)() / GB
        hw.devices, hw.gpu_names = ["mps"], ["Apple Silicon (MPS)"]
        hw.vram_total_gb, hw.vram_free_gb = [round(rec, 1)], [round(min(rec, ram_avail), 1)]
        hw.demucs_device = "mps"
        budget = min(rec, ram_avail) - RAM_RESERVE_GB - DEMUCS_VRAM_GB
        n = settings.tts_workers or max(1, min(2, int(budget // prof.cpu_ram_gb), ram_cap(False)))
        hw.tts_slots = ["mps"] * n
    else:
        hw.devices = ["cpu"]
        n = settings.tts_workers or max(1, min(3, hw.cpu_cores // 6, ram_cap(False)))
        hw.tts_slots = ["cpu"] * n
        torch.set_num_threads(max(1, hw.cpu_cores // n))

    hw.item_concurrency = settings.item_concurrency or max(2, min(16, len(hw.tts_slots) * 3 + 2))
    log.info("hardware: %s", hw.as_dict())
    return hw


# --------------------------------------------------------------- live stats
_live_cache: dict = {"ts": 0.0, "data": None}


def live_stats() -> dict:
    """CPU/RAM/GPU usage hiện tại (cache 4s — nvidia-smi/pynvml tốn tới ~100ms).
    GPU chỉ có khi backend NVIDIA; lỗi thì trả phần đọc được."""
    import time

    now = time.time()
    if _live_cache["data"] and now - _live_cache["ts"] < 4:
        return _live_cache["data"]
    out: dict = {"cpu_percent": round(psutil.cpu_percent(), 1)}
    vm = psutil.virtual_memory()
    out["ram_used_gb"] = round((vm.total - vm.available) / GB, 1)
    out["ram_total_gb"] = round(vm.total / GB, 1)
    out["ram_percent"] = round(vm.percent, 1)
    gpus = []
    try:
        import pynvml

        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpus.append({"name": pynvml.nvmlDeviceGetName(h), "util": util.gpu,
                         "vram_used_gb": round(mem.used / GB, 1), "vram_total_gb": round(mem.total / GB, 1)})
    except Exception:
        try:
            import csv
            import subprocess

            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            for row in csv.reader(r.stdout.strip().splitlines()):
                util, used, total, name = row[0].strip(), float(row[1]), float(row[2]), row[3].strip()
                gpus.append({"name": name, "util": float(util), "vram_used_gb": round(used / 1024, 1),
                             "vram_total_gb": round(total / 1024, 1)})
        except Exception as exc:
            log.debug("gpu stats unavailable: %s", exc)
    out["gpus"] = gpus
    _live_cache.update(ts=now, data=out)
    return out
