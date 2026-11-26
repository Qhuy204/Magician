"""Thực thi các thao tác sửa hình: xoá vùng (LaMa inpaint / blur), vẽ text mới, cắt đoạn, rồi encode lại video."""
import logging
import subprocess
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import runctx
from .config import settings
from .hardware import detect
from .vision import TextTrack, VideoInfo

log = logging.getLogger(__name__)

FONTS = Path(__file__).parent / "assets" / "fonts"
_lama = None
_lama_lock = threading.Lock()


@dataclass
class Region:
    """Vùng cần xử lý trên khung hình trong [start, end), toạ độ pixel."""
    start: float
    end: float
    box: tuple[int, int, int, int]
    method: str  # inpaint | blur | overlay (chỉ vẽ đè, không xoá gì: disclaimer)
    overlay: np.ndarray | None = None  # RGBA vẽ đè sau khi xoá (text mới)
    overlay_xy: tuple[int, int] = (0, 0)
    ref: np.ndarray | None = None  # frame sạch ngay trước/sau lúc vùng xuất hiện, dùng khi nền đứng yên


RING = 12
STATIC_DIFF = 8.0


def _ring(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    a = img[max(0, y0 - RING):min(H, y1 + RING), max(0, x0 - RING):min(W, x1 + RING)].astype(np.float32)
    m = np.ones(a.shape[:2], bool)
    m[y0 - max(0, y0 - RING):y1 - max(0, y0 - RING), x0 - max(0, x0 - RING):x1 - max(0, x0 - RING)] = False
    return a[m]


def _with_ref(regions: list[Region], info: VideoInfo, frame_at) -> None:
    """Gắn frame sạch (trước khi vùng xuất hiện hoặc sau khi biến mất) cho từng vùng xoá."""
    for r in regions:
        if r.method == "overlay":
            continue
        t = r.start - 0.1 if r.start >= 0.1 else (r.end + 0.1 if r.end + 0.1 < info.duration else None)
        if t is not None:
            r.ref = frame_at(t)


def _load_lama():
    global _lama
    if _lama is None:
        import torch

        path = settings.models_dir / "big-lama.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading LaMa model to %s", path)
            urllib.request.urlretrieve(settings.lama_url, path)
        dev = detect().demucs_device
        _lama = (torch.jit.load(str(path), map_location=dev).eval(), dev)
    return _lama


def inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa trên vùng crop quanh mask (đủ ngữ cảnh, cạnh dài <= 512) rồi dán lại phần bị mask."""
    import cv2
    import torch

    ys, xs = np.where(mask)
    if not len(xs):
        return img
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mw, mh = x1 - x0, y1 - y0
    cx0, cy0 = max(0, x0 - mw), max(0, y0 - mh)
    cx1, cy1 = min(w, x1 + mw), min(h, y1 + mh)
    crop, cmask = img[cy0:cy1, cx0:cx1], mask[cy0:cy1, cx0:cx1]
    scale = min(1.0, 512 / max(crop.shape[:2]))
    sw, sh = max(8, int(crop.shape[1] * scale) // 8 * 8), max(8, int(crop.shape[0] * scale) // 8 * 8)
    small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    smask = cv2.resize(cmask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    with _lama_lock:
        model, dev = _load_lama()
        # Tắt profiling executor của TorchScript: mỗi kích thước crop mới sẽ không bị tối ưu lại (~1.5s)
        with torch.no_grad(), torch.jit.optimized_execution(False):
            t_img = torch.from_numpy(small).permute(2, 0, 1)[None].float().div(255).to(dev)
            t_mask = torch.from_numpy(smask)[None, None].float().to(dev)
            out = model(t_img, t_mask)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.GaussianBlur(cmask.astype(np.float32), (0, 0), 1.5)[..., None]
    res = img.copy()
    res[cy0:cy1, cx0:cx1] = (out * alpha + crop * (1 - alpha)).astype(np.uint8)
    return res


def blur(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    import cv2

    x0, y0, x1, y1 = box
    res = img.copy()
    region = res[y0:y1, x0:x1]
    k = max(15, (max(region.shape[:2]) // 4) | 1)
    res[y0:y1, x0:x1] = cv2.GaussianBlur(region, (k, k), 0)
    return res


def _font_for(text: str) -> Path:
    if any("\u0600" <= c <= "\u06ff" or "\u0750" <= c <= "\u077f" for c in text):
        return FONTS / "NotoSansArabic.ttf"
    if any("\u0900" <= c <= "\u097f" for c in text):
        return FONTS / "NotoSansDevanagari.ttf"
    return FONTS / "NotoSans.ttf"


def _kmeans(px: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    _, labels, centers = cv2.kmeans(px.astype(np.float32), k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centers


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in np.clip(c, 0, 255))


def _contrast(c: np.ndarray) -> np.ndarray:
    return np.array([0, 0, 0]) if c.mean() > 127 else np.array([255, 255, 255])


def _text_colors_diff(region: np.ndarray, clean: np.ndarray) -> tuple[str, str] | None:
    """Dùng frame sạch (lúc chưa có chữ) để tách pixel chữ khỏi nền rồi đoán màu thân chữ + màu viền."""
    import cv2

    changed = (np.abs(region.astype(np.int16) - clean.astype(np.int16)).max(axis=2) > 50).astype(np.uint8)
    if changed.sum() < 30:
        return None
    if changed.mean() > 0.7:
        # chữ nằm trên khung nền riêng (pop-up, banner): cụm lớn nhất là khung, cụm còn lại là chữ
        labels, centers = _kmeans(region.reshape(-1, 3), 2)
        fill = centers[np.argmin(np.bincount(labels, minlength=2))]
        return _hex(fill), _hex(_contrast(fill))
    # chữ trắng trên nền trắng thì chỉ có viền đổi màu: đóng kín nét chữ để lấy cả phần thân
    closed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    dist = cv2.distanceTransform(closed.astype(np.uint8), cv2.DIST_L2, 3)
    labels, centers = _kmeans(region[closed], 2)
    near_edge = dist[closed] <= 1.5
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 0.1 * counts.sum():
        fill = centers[np.argmax(counts)]
        return _hex(fill), _hex(_contrast(fill))
    # cụm nằm sát mép nhiều hơn là viền
    edge_ratio = [near_edge[labels == i].mean() for i in range(2)]
    fill, border = centers[int(np.argmin(edge_ratio))], centers[int(np.argmax(edge_ratio))]
    if np.abs(fill - border).mean() < 60:
        border = _contrast(fill)
    return _hex(fill), _hex(border)


def _text_colors(region: np.ndarray) -> tuple[str, str]:
    """Đoán màu chữ + màu viền: k-means 3 cụm trên vùng chữ, cụm lớn nhất là nền, cụm lớn kế tiếp là thân chữ,
    cụm còn lại là viền (nếu đủ tương phản với thân chữ)."""
    import cv2

    if region.size == 0:
        return "white", "black"
    px = region.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(px, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=3))
    fill, edge = centers[order[1]], centers[order[2]]
    hexc = lambda c: "#%02x%02x%02x" % tuple(int(v) for v in np.clip(c, 0, 255))
    if np.abs(fill - edge).mean() < 60:
        edge = np.array([0, 0, 0]) if fill.mean() > 127 else np.array([255, 255, 255])
    return hexc(fill), hexc(edge)


def _draw_lines(lines: list[str], font: Path, size: int, W: int, H: int, color: str, border: str) -> np.ndarray:
    """ffmpeg drawtext từng dòng (HarfBuzz + FriBiDi nên nối chữ Arab / Devanagari đúng), căn giữa, ra RGBA."""
    from PIL import Image

    line_h = int(size * 1.35)
    top = (H - line_h * len(lines)) // 2
    esc = lambda p: str(p).replace("\\", "/").replace(":", r"\:")
    with tempfile.TemporaryDirectory() as td:
        draws = []
        for i, line in enumerate(lines):
            tf = Path(td) / f"t{i}.txt"
            tf.write_text(line, encoding="utf-8")
            draws.append(f"drawtext=fontfile='{esc(font)}':textfile='{esc(tf)}':fontsize={size}:fontcolor={color}"
                         f":borderw={max(1, size // 12)}:bordercolor={border}:x=(w-text_w)/2:y={top + i * line_h}+({line_h}-th)/2")
        out = Path(td) / "t.png"
        runctx.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black@0.0:s={W}x{H},format=rgba",
                        "-vf", ",".join(draws), "-frames:v", "1", str(out)], check=True, capture_output=True)
        return np.array(Image.open(out).convert("RGBA"))


def _ink_width(img: np.ndarray) -> int:
    cols = np.where(img[..., 3].max(axis=0) > 0)[0]
    return int(cols[-1] - cols[0] + 1) if len(cols) else 0


def render_text(text: str, box_w: int, box_h: int, color: str, border: str) -> np.ndarray:
    """Vẽ text mới vừa khung chữ cũ: đo bề rộng thật sau khi shape (vẽ thử 1 lần) rồi chỉnh cỡ chữ;
    câu quá dài so với khung thì xuống 2 dòng (khung cao thêm)."""
    font = _font_for(text)
    base = max(12, int(box_h * 0.72))

    def fit(lines: list[str]) -> int:
        probe = _draw_lines(lines, font, base, box_w * 4, int(base * 1.35 * len(lines)) + 8, color, border)
        width = _ink_width(probe) or 1
        return max(12, min(base, int(base * box_w * 0.98 / width)))

    lines, size = [text], fit([text])
    words = text.split()
    if size < base * 0.5 and len(words) > 3:
        mid = len(words) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
        size = fit(lines)
    pad = max(4, size // 4)
    W = int(box_w + 2 * pad) // 2 * 2
    H = int(max(box_h, size * 1.35 * len(lines)) + 2 * pad) // 2 * 2
    return _draw_lines(lines, font, size, W, H, color, border)


# Vùng an toàn để đặt disclaimer: tránh 20% dưới cùng (caption + nút của TikTok) và cột nút bên phải
DISCLAIMER_BOX = (0.1, 0.67, 0.82, 0.71)
AIGC_BOX = (0.25, 0.085, 0.75, 0.115)


def overlay_label(text: str, info: VideoInfo, box: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[int, int]]:
    """Dòng chữ trắng viền đen trên nền tối mờ, vừa khung box (toạ độ chuẩn hoá), dùng cho disclaimer / nhãn AI."""
    x0, y0, x1, y1 = (int(v * d) for v, d in zip(box, (info.width, info.height, info.width, info.height)))
    img = render_text(text, x1 - x0, y1 - y0, "white", "black")
    h, w = img.shape[:2]
    pad = max(4, h // 6)
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)
    out[..., 3] = 110  # nền đen mờ để chữ đọc được trên mọi nền
    tile = out[pad:pad + h, pad:pad + w]
    a = img[..., 3:4].astype(np.float32) / 255
    tile[..., :3] = (img[..., :3] * a + tile[..., :3] * (1 - a)).astype(np.uint8)
    tile[..., 3] = np.maximum(tile[..., 3], img[..., 3])
    xy = (max(0, x0 + (x1 - x0 - out.shape[1]) // 2), max(0, y0 + (y1 - y0 - out.shape[0]) // 2))
    return out, xy


def _px_box(bbox: list[float], info: VideoInfo, pad: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    ph = (y1 - y0) * pad
    return (max(0, int((x0 - ph * info.height / info.width) * info.width)), max(0, int((y0 - ph) * info.height)),
            min(info.width, int((x1 + ph * info.height / info.width) * info.width) + 1), min(info.height, int((y1 + ph) * info.height) + 1))


TIME_PAD = 0.25
MAX_GAP = 2.0


def _expand_tracks(selected: list[TextTrack], all_tracks: dict[int, TextTrack]) -> list[list[TextTrack]]:
    """OCR hay bỏ sót vài frame nên 1 dòng chữ bị tách thành nhiều track. Gom thêm các track cùng vị trí + gần giống
    nội dung, rồi chia theo nhóm vị trí; mỗi nhóm phủ liên tục từ lúc xuất hiện đầu tiên tới lúc biến mất."""
    from .vision import _iou, _similar

    pool = {t.id: t for t in selected}
    for t in all_tracks.values():
        if t.id not in pool and any(_iou(t.bbox, s.bbox) > 0.5 and _similar(t.text, s.text) > 0.5 for s in selected):
            pool[t.id] = t
    def same_row(a: TextTrack, b: TextTrack) -> bool:
        oy = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
        ox = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
        return ox > 0 and oy >= 0.5 * min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])

    groups: list[list[TextTrack]] = []
    for t in sorted(pool.values(), key=lambda x: x.start):
        for g in groups:
            if any(same_row(t, x) for x in g) and t.start - max(x.end for x in g) <= MAX_GAP:
                g.append(t)
                break
        else:
            groups.append([t])
    return groups


def _union(ts: list[TextTrack]) -> list[float]:
    return [min(t.bbox[0] for t in ts), min(t.bbox[1] for t in ts), max(t.bbox[2] for t in ts), max(t.bbox[3] for t in ts)]


def _refine_region(a: dict, info: VideoInfo, frame_at) -> tuple[int, int, int, int]:
    """Bbox của vision LLM thường hụt vài %. Nới thêm lề, rồi so frame trước khi graphic xuất hiện với frame lúc đang
    hiện: vùng pixel đổi mạnh quanh bbox chính là graphic, lấy hợp của 2 vùng."""
    import cv2

    x0, y0, x1, y1 = a["bbox"]
    mx, my = max(0.015, (x1 - x0) * 0.08), max(0.012, (y1 - y0) * 0.15)
    box = [max(0.0, x0 - mx), max(0.0, y0 - my), min(1.0, x1 + mx), min(1.0, y1 + my)]
    px = [int(box[0] * info.width), int(box[1] * info.height), int(box[2] * info.width) + 1, int(box[3] * info.height) + 1]
    ref_t = a["start"] - 0.3 if a["start"] >= 0.3 else (a["end"] + 0.3 if a["end"] + 0.3 < info.duration else None)
    if ref_t is None:
        return tuple(px)
    before, during = frame_at(ref_t).astype(np.int16), frame_at((a["start"] + a["end"]) / 2).astype(np.int16)
    # chỉ tìm trong vùng gấp đôi bbox để không bắt nhầm chuyển động ở chỗ khác
    w, h = px[2] - px[0], px[3] - px[1]
    sx0, sy0 = max(0, px[0] - w // 2), max(0, px[1] - h // 2)
    sx1, sy1 = min(info.width, px[2] + w // 2), min(info.height, px[3] + h // 2)
    diff = np.abs(during[sy0:sy1, sx0:sx1] - before[sy0:sy1, sx0:sx1]).max(axis=2) > 40
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(diff)
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        gx0, gy0, gx1, gy1 = cx + sx0, cy + sy0, cx + sx0 + cw, cy + sy0 + ch
        # chỉ nhận thành phần chạm vào bbox gốc và không phủ gần hết vùng tìm (tức là cả cảnh đổi)
        touches = gx0 < px[2] and gx1 > px[0] and gy0 < px[3] and gy1 > px[1]
        if touches and area > 0.05 * w * h and cw * ch < 0.9 * (sx1 - sx0) * (sy1 - sy0):
            px = [min(px[0], gx0 - 4), min(px[1], gy0 - 4), max(px[2], gx1 + 4), max(px[3], gy1 + 4)]
    return (max(0, px[0]), max(0, px[1]), min(info.width, px[2]), min(info.height, px[3]))


def build_regions(actions: list[dict], tracks: dict[int, TextTrack], info: VideoInfo, frame_at,
                  blur_only: bool = False) -> list[Region]:
    """Chuyển action của planner thành vùng pixel; replace_text thì chuẩn bị sẵn ảnh text mới.
    blur_only: mọi vùng đều làm mờ (replace_text cũng chỉ làm mờ text cũ, không vẽ text mới)."""
    regions = []
    for a in actions:
        if a["type"] in ("remove_text", "replace_text"):
            groups = _expand_tracks([tracks[i] for i in a["track_ids"]], tracks)
            for gi, g in enumerate(groups):
                start, end = max(0.0, min(t.start for t in g) - TIME_PAD), min(info.duration, max(t.end for t in g) + TIME_PAD)
                box = _px_box(_union(g), info, 0.25)
                reg = Region(start, end, box, "blur" if blur_only else "inpaint")
                if a["type"] == "replace_text" and gi == 0 and not blur_only:
                    tb = _px_box(_union(g), info, 0.0)
                    first = min(g, key=lambda t: t.start)
                    fx0, fy0, fx1, fy1 = _px_box(first.bbox, info, 0.0)
                    region = frame_at((first.start + first.end) / 2)[fy0:fy1, fx0:fx1]
                    ref_t = first.start - 0.15 if first.start >= 0.15 else max(t.end for t in g) + 0.15
                    colors = _text_colors_diff(region, frame_at(ref_t)[fy0:fy1, fx0:fx1]) if ref_t < info.duration else None
                    color, border = colors or _text_colors(region)
                    img = render_text(a["new_text"], tb[2] - tb[0], tb[3] - tb[1], color, border)
                    reg.overlay = img
                    reg.overlay_xy = (tb[0] + (tb[2] - tb[0] - img.shape[1]) // 2, tb[1] + (tb[3] - tb[1] - img.shape[0]) // 2)
                regions.append(reg)
        elif a["type"] in ("add_disclaimer", "aigc_label"):
            img, xy = overlay_label(a["text"], info, AIGC_BOX if a["type"] == "aigc_label" else DISCLAIMER_BOX)
            reg = Region(a["start"], a["end"], (xy[0], xy[1], xy[0] + img.shape[1], xy[1] + img.shape[0]), "overlay")
            reg.overlay, reg.overlay_xy = img, xy
            regions.append(reg)
        elif a["type"] == "remove_region":
            box = _refine_region(a, info, frame_at)
            x0, y0, x1, y1 = box
            # Làm mờ 1 khối màu đặc vẫn để lại khối màu, nên mặc định inpaint; blur khi vùng quá lớn để inpaint,
            # khi request ép blur, hoặc khi planner chọn blur (watermark/handle trong suốt không inpaint sạch được)
            big = (x1 - x0) * (y1 - y0) > 0.2 * info.width * info.height
            method = "blur" if blur_only or big or a.get("method") == "blur" else "inpaint"
            regions.append(Region(max(0.0, a["start"] - TIME_PAD), min(info.duration, a["end"] + TIME_PAD), box, method))
    _with_ref(regions, info, frame_at)
    return regions


def _composite(img: np.ndarray, overlay: np.ndarray, xy: tuple[int, int]) -> np.ndarray:
    x, y = xy
    h, w = overlay.shape[:2]
    H, W = img.shape[:2]
    sx0, sy0, dx0, dy0 = max(0, -x), max(0, -y), max(0, x), max(0, y)
    dx1, dy1 = min(W, x + w), min(H, y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return img
    ov = overlay[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0].astype(np.float32)
    a = ov[..., 3:4] / 255
    img[dy0:dy1, dx0:dx1] = (ov[..., :3] * a + img[dy0:dy1, dx0:dx1] * (1 - a)).astype(np.uint8)
    return img


def _apply(frame: np.ndarray, t: float, regions: list[Region]) -> np.ndarray:
    active = [r for r in regions if r.start <= t < r.end]
    if not active:
        return frame
    inp = []
    for r in active:
        if r.method != "inpaint":
            continue
        # Nền đứng yên (vd quay màn hình): chép lại đúng pixel nền từ frame sạch, đẹp hơn inpaint
        if r.ref is not None and np.abs(_ring(frame, r.box) - _ring(r.ref, r.box)).mean() < STATIC_DIFF:
            x0, y0, x1, y1 = r.box
            frame[y0:y1, x0:x1] = r.ref[y0:y1, x0:x1]
        else:
            inp.append(r)
    if inp:
        mask = np.zeros(frame.shape[:2], bool)
        for r in inp:
            x0, y0, x1, y1 = r.box
            mask[y0:y1, x0:x1] = True
        frame = inpaint(frame, mask)
    for r in active:
        if r.method == "blur":
            frame = blur(frame, r.box)
    for r in active:
        if r.overlay is not None:
            frame = _composite(frame, r.overlay, r.overlay_xy)
    return frame


def _gray_frames(video: Path, info: VideoInfo, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Frame xám 64px trong [t0, t1] kèm thời điểm của từng frame."""
    t0, t1 = max(0.0, t0), min(info.duration, t1)
    w = 64
    h = max(2, int(round(info.height * w / info.width / 2)) * 2)
    raw = runctx.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(video),
                      "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    frames = np.frombuffer(raw[: len(raw) // (w * h) * w * h], np.uint8).reshape(-1, h, w).astype(np.float32)
    return t0 + (np.arange(len(frames)) + 0.5) / info.fps, frames


def _transitions(times: np.ndarray, frames: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """Các lần chuyển cảnh (cắt cứng hoặc mờ dần): so frame i với frame i-k (~0.25s trước) để bắt cả fade,
    gom các frame vượt ngưỡng liền nhau thành 1 khoảng [bắt đầu, kết thúc] của chuyển cảnh."""
    k = max(1, int(round(fps * 0.25)))
    if len(frames) <= k + 1:
        return []
    d = np.abs(frames[k:] - frames[:-k]).mean(axis=(1, 2))
    thr = max(12.0, 3 * float(np.median(d)))
    out, cur = [], None
    for idx in np.where(d > thr)[0]:
        a, b = times[idx], times[idx + k]  # khoảng [frame i-k, frame i] chứa chuyển cảnh
        if cur and a <= cur[1] + 1.5 / fps:
            cur = (cur[0], b)
        else:
            if cur:
                out.append(cur)
            cur = (a, b)
    if cur:
        out.append(cur)
    # thu hẹp mỗi khoảng về đúng frame thay đổi (so từng cặp frame liền nhau)
    tight = []
    step = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    for a, b in out:
        sel = [n for n in range(len(step)) if a - 1e-6 <= times[n] and times[n + 1] <= b + 1e-6]
        moving = [n for n in sel if step[n] > max(1.0, 0.1 * max(step[m] for m in sel))] if sel else []
        tight.append((times[moving[0]], times[moving[-1] + 1]) if moving else (a, b))
    return tight


def snap_cut(video: Path, info: VideoInfo, start: float, end: float, window: float = 1.0) -> tuple[float, float]:
    """Planner chỉ xem frame mỗi giây nên mốc cắt lệch ranh giới cảnh. Dời start về đầu lần chuyển cảnh GẦN start nhất
    và end về cuối lần chuyển cảnh gần end nhất (trong ±window), bắt được cả chuyển cảnh mờ dần, để không sót
    đầu/đuôi của shot vi phạm mà cũng không cắt lẹm sang shot hợp lệ bên cạnh."""
    def edge(t: float, at_end: bool) -> float:
        times, frames = _gray_frames(video, info, t - window, t + window)
        trans = _transitions(times, frames, info.fps)
        if not trans:
            return t
        a, b = min(trans, key=lambda r: 0.0 if r[0] <= t <= r[1] else min(abs(r[0] - t), abs(r[1] - t)))
        return float(b + 0.5 / info.fps) if at_end else float(a - 0.5 / info.fps)

    s = 0.0 if start < 0.3 else max(0.0, edge(start, False))
    e = info.duration if info.duration - end < 0.3 else min(info.duration, edge(end, True))
    if e - s < 0.2:
        return start, end
    return round(s, 3), round(e, 3)


def cut_audio(audio: np.ndarray, sr: int, cuts: list[tuple[float, float]], fade: float = 0.015) -> np.ndarray:
    """Bỏ các khoảng cắt khỏi audio, crossfade ngắn ở chỗ nối để không bị click."""
    keep, cursor = [], 0
    for s, e in sorted(cuts):
        keep.append(audio[cursor:int(s * sr)])
        cursor = int(e * sr)
    keep.append(audio[cursor:])
    n = int(fade * sr)
    out = keep[0]
    for part in keep[1:]:
        if len(out) > n and len(part) > n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
            out = np.concatenate([out[:-n], out[-n:] * (1 - ramp) + part[:n] * ramp, part[n:]])
        else:
            out = np.concatenate([out, part])
    return out


def render(video: Path, audio_wav: Path | None, out: Path, info: VideoInfo, regions: list[Region],
           cuts: list[tuple[float, float]], audio_is_cut: bool = False) -> None:
    """Decode từng frame, sửa frame trong các vùng, bỏ frame bị cắt, encode lại kèm audio đã cắt tương ứng."""
    cut_wav = out.with_suffix(".cut.wav")
    audio_args = ["-an"]
    if audio_wav:
        audio, sr = sf.read(audio_wav, dtype="float32", always_2d=True)
        sf.write(cut_wav, cut_audio(audio, sr, cuts) if cuts and not audio_is_cut else audio, sr, subtype="FLOAT")
        audio_args = ["-i", str(cut_wav), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    w, h, size = info.width, info.height, info.width * info.height * 3
    dec = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{info.fps:.6f}", "-i", "-",
         *audio_args, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(settings.video_crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    if ctx := runctx.current():
        ctx.register(dec)
        ctx.register(enc)
    try:
        i = 0
        while len(buf := dec.stdout.read(size)) == size:
            t = (i + 0.5) / info.fps
            i += 1
            if i % 30 == 0:
                runctx.check()
            if any(s <= t < e for s, e in cuts):
                continue
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            enc.stdin.write(_apply(frame.copy(), t, regions).tobytes() if regions else buf)
    finally:
        # exception giữa chừng (vd enc chết làm gãy pipe) thì dec vẫn đang ghi vào pipe không ai đọc:
        # chờ wait() sẽ treo vĩnh viễn, phải giết process không thoát kịp
        try:
            enc.stdin.close()
        except Exception:
            pass
        for proc in (dec, enc):
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
                proc.wait()
        cut_wav.unlink(missing_ok=True)
        if ctx := runctx.current():
            ctx.unregister(dec)
            ctx.unregister(enc)
    runctx.check()
    if enc.returncode:
        raise RuntimeError(f"ffmpeg encode failed ({enc.returncode})")
