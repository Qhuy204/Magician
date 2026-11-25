"""Phân tích hình: lấy mẫu frame, OCR text trên màn hình và gom thành các "text track" theo thời gian."""
import base64
import difflib
import io
import json
import logging
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import runctx
from .config import settings
from .hardware import detect

log = logging.getLogger(__name__)

# EasyOCR chỉ ghép được các ngôn ngữ cùng hệ chữ; luôn kèm "en" để đọc số, SAR, $...
OCR_LANGS = {"ar": ["ar", "en"], "hi": ["hi", "en"], "pt": ["pt", "en"], "es": ["es", "en"], "fr": ["fr", "en"],
             "de": ["de", "en"], "it": ["it", "en"], "tr": ["tr", "en"], "ru": ["ru", "en"], "ja": ["ja", "en"],
             "ko": ["ko", "en"], "zh-cn": ["ch_sim", "en"]}
_ocr_lock = threading.Lock()
_readers: dict[tuple, object] = {}


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    frames: int


@dataclass
class TextTrack:
    id: int
    text: str
    bbox: list[float]  # [x0, y0, x1, y1] chuẩn hoá 0–1
    start: float
    end: float
    confidence: float
    boxes: list[list[float]] = field(default_factory=list, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("boxes")
        d["bbox"] = [round(x, 3) for x in self.bbox]
        return d


def video_info(path: Path) -> VideoInfo:
    out = runctx.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_read_packets:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    st = d["streams"][0]
    num, _, den = st["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    return VideoInfo(int(st["width"]), int(st["height"]), fps, float(d["format"]["duration"]), int(st["nb_read_packets"]))


def sample_frames(path: Path, fps: float, max_side: int) -> list[tuple[float, np.ndarray]]:
    """Frame RGB lấy mẫu đều theo `fps`, thu nhỏ cạnh dài về `max_side`."""
    info = video_info(path)
    scale = min(1.0, max_side / max(info.width, info.height))
    w, h = int(info.width * scale) // 2 * 2, int(info.height * scale) // 2 * 2
    raw = runctx.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return [((i + 0.5) / fps, f) for i, f in enumerate(frames)]


def _reader(lang: str):
    langs = tuple(OCR_LANGS.get(lang, ["en"]))
    if langs not in _readers:
        import easyocr

        dev = detect().demucs_device
        _readers[langs] = easyocr.Reader(list(langs), gpu=dev.startswith("cuda"), verbose=False,
                                         model_storage_directory=str(settings.models_dir / "easyocr"))
    return _readers[langs]


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def ocr_tracks(frames: list[tuple[float, np.ndarray]], lang: str, sample_fps: float,
               stop: threading.Event | None = None) -> list[TextTrack]:
    """OCR từng frame mẫu rồi nối các box cùng vị trí + cùng nội dung ở các frame liên tiếp thành 1 track."""
    with _ocr_lock:
        reader = _reader(lang)
        detections = []
        last_small, last_found = None, []
        for t, img in frames:
            if stop is not None and stop.is_set():
                return []
            h, w = img.shape[:2]
            small = img[::8, ::8].mean(axis=2)
            # frame gần như y hệt frame trước (màn hình tĩnh): dùng lại kết quả OCR, đỡ ~0.4s/frame
            if last_small is not None and np.abs(small - last_small).mean() < 1.5:
                found = last_found
            else:
                found = []
                for quad, text, conf in reader.readtext(img, paragraph=False):
                    if conf < 0.3 or not text.strip():
                        continue
                    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
                    found.append(([min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h], text.strip(), float(conf)))
                last_small, last_found = small, found
            detections += [(t, box, text, conf) for box, text, conf in found]

    gap = 1.5 / sample_fps
    tracks: list[dict] = []
    for t, box, text, conf in detections:
        best = None
        for tr in tracks:
            if t - tr["last"] > gap + 1e-6 or tr["last"] == t:
                continue
            if _iou(tr["boxes"][-1], box) > 0.4 and _similar(tr["texts"][-1], text) > 0.5:
                best = tr
                break
        if best is None:
            best = {"first": t, "last": t, "boxes": [], "texts": [], "confs": []}
            tracks.append(best)
        best["last"] = t
        best["boxes"].append(box)
        best["texts"].append(text)
        best["confs"].append(conf)

    half = 0.5 / sample_fps
    out = []
    for i, tr in enumerate(sorted(tracks, key=lambda x: x["first"])):
        b = np.array(tr["boxes"])
        out.append(TextTrack(
            id=i, text=max(tr["texts"], key=tr["texts"].count), confidence=round(float(np.mean(tr["confs"])), 2),
            bbox=[float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())],
            start=round(max(0.0, tr["first"] - half), 2), end=round(tr["last"] + half, 2), boxes=tr["boxes"],
        ))
    return out


def jpeg_data_url(img: np.ndarray, quality: int = 70) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def frame_at(path: Path, t: float) -> np.ndarray:
    """1 frame RGB độ phân giải gốc tại thời điểm t."""
    info = video_info(path)
    raw = runctx.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw[: info.width * info.height * 3], np.uint8).reshape(info.height, info.width, 3)
