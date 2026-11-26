import hashlib
import json
import logging
import re
import shutil
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import soundfile as sf
from openai import OpenAI

from . import media, policies, recheck, runctx, video_edit, vision
from .errors import InvalidInput
from .config import settings
from .rewrite import lengthen, plan_edits, shorten
from .planner import plan_fix, triage
from .schemas import AdItem, Edit, ItemResult, ReviewReason
from .separation import separate
from .transcribe import transcribe
from .tts import REF_SECONDS, engine, supports_duration, synthesize

log = logging.getLogger(__name__)

FADE = 0.02


class JobCache:
    """Cache dùng chung giữa các item trong 1 job: video trùng audio/lời thoại thì chỉ xử lý 1 lần."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key_locks: dict = {}
        self._values: dict = {}
        self.hits: dict[str, int] = {}

    def get(self, key: tuple, compute):
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if key in self._values:
                with self._lock:
                    self.hits[key[0]] = self.hits.get(key[0], 0) + 1
                return self._values[key]
            value = self._values[key] = compute()
            return value


def _digest(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _openai_client(usage: dict[str, float]) -> OpenAI:
    """OpenAI client cộng dồn số call / token / giây audio vào `usage`."""
    lock = threading.Lock()

    def hook(resp: httpx.Response) -> None:
        resp.read()
        try:
            u = resp.json().get("usage") or {}
        except ValueError:
            u = {}
        kind = "transcribe" if "/audio/" in resp.request.url.path else "chat"
        with lock:
            usage[f"{kind}_calls"] = usage.get(f"{kind}_calls", 0) + 1
            for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "seconds"):
                if isinstance(u.get(k), (int, float)):
                    usage[f"{kind}_{k}"] = usage.get(f"{kind}_{k}", 0) + u[k]

    return OpenAI(api_key=settings.openai_api_key, http_client=httpx.Client(timeout=180, event_hooks={"response": [hook]}))


@contextmanager
def _timed(timings: dict[str, float], stage: str):
    """Đo thời gian + báo stage hiện tại / áp timeout của stage qua runctx."""
    t = time.perf_counter()
    try:
        with runctx.stage(stage):
            yield
    finally:
        timings[stage] = round(timings.get(stage, 0) + time.perf_counter() - t, 2)


def _persisted(path: Path, compute):
    """Kết quả JSON của 1 bước (gọi OpenAI) được lưu ra work dir; retry item thì đọc lại, không gọi lại API."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            path.unlink(missing_ok=True)
    value = compute()
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return value


def _separate_cached(mix_wav: Path, work: Path) -> tuple[np.ndarray, np.ndarray, int]:
    vp, bp = work / "vocals.wav", work / "bgm.wav"
    if vp.exists() and bp.exists():
        vocals, sr = sf.read(vp, dtype="float32", always_2d=True)
        bgm, _ = sf.read(bp, dtype="float32", always_2d=True)
        return vocals, bgm, sr
    vocals, bgm, sr = separate(mix_wav)
    for arr, path in ((bgm, bp), (vocals, vp)):
        tmp = path.with_name(path.stem + ".part.wav")
        sf.write(tmp, arr, sr, subtype="FLOAT")
        tmp.replace(path)
    return vocals, bgm, sr


def _fade_mute(track: np.ndarray, s: int, e: int, fade: int) -> None:
    """Tắt tiếng track[s:e] với fade-out/fade-in ở hai mép."""
    s, e = max(0, s), min(len(track), e)
    if e <= s:
        return
    track[s:e] = 0
    if fade and s > 0:
        a = max(0, s - fade)
        track[a:s] *= np.linspace(1, 0, s - a, dtype=np.float32)[:, None]
    if fade and e < len(track):
        b = min(len(track), e + fade)
        track[e:b] *= np.linspace(0, 1, b - e, dtype=np.float32)[:, None]


def _rms(x: np.ndarray) -> float:
    x = x[np.abs(x).max(axis=1) > 1e-3] if x.size else x
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def _speaker_refs(vocals: np.ndarray, sr: int, segments: list[dict], work: Path) -> tuple[list[Path], str]:
    """Ghép các đoạn vocal dài nhất làm mẫu clone giọng; trả kèm transcript của mẫu (OmniVoice cần)."""
    limit = REF_SECONDS.get(engine(), 15.0)
    chosen, total = [], 0.0
    for seg in sorted(segments, key=lambda s: s["end"] - s["start"], reverse=True):
        if chosen and total + seg["end"] - seg["start"] > limit * 1.3:
            continue
        chosen.append(seg)
        total += seg["end"] - seg["start"]
        if total >= limit:
            break
    chosen.sort(key=lambda s: s["start"])
    ref = work / "speaker_ref.wav"
    sf.write(ref, np.concatenate([vocals[int(s["start"] * sr) : int(s["end"] * sr)] for s in chosen]).mean(axis=1), sr)
    return [ref], " ".join(s["text"] for s in chosen)


# Câu mới ngắn hơn tỉ lệ này của khung thì viết dài ra / đọc chậm lại (người nói vẫn mấp máy môi trên hình)
MIN_FILL = 0.75
TARGET_FILL = 0.92
MAX_SLOWDOWN = 1.2


def _render_edit(
    client: OpenAI, edit: Edit, language: str, refs: list[Path], ref_text: str, slot: float,
    work: Path, idx: int, cache: JobCache, voice_key: str, product: str | None = None,
) -> np.ndarray:
    fitted = work / f"tts_{idx}.wav"

    def synth(tag: str, duration: float | None = None) -> float:
        """Sinh giọng, trim silence, trả về tỉ lệ độ dài so với khung (>1 = dài hơn khung)."""
        text = edit.new_text
        out = work / f"tts_{idx}_{tag}_{_digest(json.dumps([text, language, duration, voice_key]))[:10]}.wav"
        raw = cache.get(("tts", voice_key, language, text, duration),
                        lambda: synthesize(text, language, refs, out, ref_text, duration))
        media.condition_tts(raw, fitted)
        nonlocal last_raw
        last_raw = raw
        return sf.info(fitted).duration / slot if slot > 0 else 1.0

    last_raw: Path | None = None
    need = synth("raw0")
    if need > settings.max_tempo:
        max_chars = max(5, int(len(edit.new_text) * settings.max_tempo / need * 0.95))
        text = edit.new_text
        edit.new_text = cache.get(("shorten", voice_key, text), lambda: shorten(client, text, language, max_chars))
        need = synth("raw1")
    elif need < MIN_FILL:
        text, limit = edit.new_text, len(edit.original_text)
        min_chars = min(limit, int(len(text) * TARGET_FILL / max(need, 0.05)))
        edit.new_text = cache.get(("lengthen", voice_key, text),
                                  lambda: lengthen(client, text, language, min_chars, limit, product))
        if edit.new_text != text:
            need = synth("raw1")
    if need > 1.0 and supports_duration():
        # OmniVoice đọc nhanh tự nhiên cho vừa khung thay vì tăng tốc audio
        need = synth("fit", round(slot, 2))
        edit.fit = "duration"
    elif need < TARGET_FILL and supports_duration():
        # vẫn ngắn: đọc chậm lại một chút (tối đa MAX_SLOWDOWN) cho gần kín khung
        need = synth("fill", round(min(slot * 0.97, need * slot * MAX_SLOWDOWN), 2))
        edit.fit = "duration"
    edit.tempo = round(min(max(need, 1.0), settings.max_tempo), 3)
    if edit.tempo > 1.0:
        media.condition_tts(last_raw, fitted, tempo=edit.tempo)
        edit.fit = "atempo" if edit.fit == "natural" else "duration+atempo"
    audio, _ = sf.read(fitted, dtype="float32", always_2d=True)
    return audio


def _download_images(urls: list[str], work: Path) -> list[str]:
    """Tải ảnh frame vi phạm TikTok gửi kèm (đã check SSRF) thành data URL cho planner."""
    import base64

    out = []
    for i, url in enumerate(urls[:6]):
        try:
            path = media.download(url, work / f"violation_{i}.img")
            out.append("data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode())
        except Exception as exc:
            log.warning("violation frame %s: %s", url[:80], exc)
    return out


def _norm_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c for c in text if not unicodedata.category(c).startswith(("M", "P", "S")))
    return re.findall(r"\w+", text)


def _caption_actions(edits: list[Edit], tracks: list[vision.TextTrack], taken: set[int]) -> list[dict]:
    """Phụ đề in sẵn lặp lại voice-over: track nào trùng thời gian + nội dung với câu bị sửa thì thay bằng câu mới,
    chia câu mới theo các khung phụ đề xuất hiện lần lượt."""
    import difflib

    def known(word: str, vocab: set[str]) -> bool:
        # OCR hay dính thêm/thiếu 1 ký tự (vd "امضمون"), nên so gần đúng
        return word in vocab or any(difflib.SequenceMatcher(None, word, v).ratio() >= 0.75 for v in vocab)

    actions = []
    for e in edits:
        words = set(_norm_words(e.original_text))
        cands = []
        for t in tracks:
            tw = _norm_words(t.text)
            if t.id in taken or not tw or t.end <= e.start or t.start >= e.end:
                continue
            if sum(known(w, words) for w in tw) / len(tw) >= 0.6:
                cands.append(t)
        if not cands:
            continue
        # gom các track xuất hiện cùng lúc (phụ đề nhiều dòng) thành 1 khung
        groups: list[list[vision.TextTrack]] = []
        for t in sorted(cands, key=lambda x: x.start):
            if groups and t.start < max(x.end for x in groups[-1]) - 0.2:
                groups[-1].append(t)
            else:
                groups.append([t])
        new_words = e.new_text.split()
        if len(groups) == 1 or len(new_words) <= 8:
            # câu mới ngắn: hiện nguyên câu suốt thời gian của các khung phụ đề cũ
            ids = [x.id for g in groups for x in g]
            taken.update(ids)
            actions.append({"type": "replace_text", "track_ids": ids, "new_text": e.new_text})
            continue
        span = lambda g: max(x.end for x in g) - min(x.start for x in g)
        total = sum(span(g) for g in groups) or 1.0
        cursor = 0
        for i, g in enumerate(groups):
            n = len(new_words) - cursor if i == len(groups) - 1 else max(1, round(span(g) / total * len(new_words)))
            chunk = " ".join(new_words[cursor:cursor + n])
            cursor += n
            ids = [x.id for x in g]
            taken.update(ids)
            actions.append({"type": "replace_text", "track_ids": ids, "new_text": chunk} if chunk
                           else {"type": "remove_text", "track_ids": ids})
    return actions


def _to_cut(t: float, cuts: list[tuple[float, float]]) -> float:
    """Thời điểm trên video gốc -> thời điểm trên video đã cắt."""
    return t - sum(max(0.0, min(t, e) - s) for s, e in cuts)


def _from_cut(t: float, cuts: list[tuple[float, float]]) -> float:
    """Thời điểm trên video đã cắt -> thời điểm trên video gốc."""
    for s, e in sorted(cuts):
        if t >= s:
            t += e - s
    return t


def _remap_segments(segments: list[dict], cuts: list[tuple[float, float]]) -> list[dict]:
    out = []
    for seg in segments:
        s, e = _to_cut(seg["start"], cuts), _to_cut(seg["end"], cuts)
        if e - s >= 0.2:
            out.append({**seg, "start": round(s, 3), "end": round(e, 3)})
    return out


def _even(frames: list, limit: int) -> list:
    if len(frames) <= limit:
        return frames
    step = len(frames) / limit
    return [frames[int(i * step)] for i in range(limit)]


def _rewrite_voiceover(client, result, segments, language, vocals, sr, work, cache, hint, timed,
                       product: str | None = None, policy: dict | None = None) -> np.ndarray:
    """Viết lại câu thoại vi phạm, đọc lại bằng giọng clone và ghép vào vocal stem. Trả về vocal mới."""
    reason_lines = result.reject_reasons
    new_vocals = vocals.copy()
    with timed("rewrite"):
        script_key = _digest(json.dumps([language, reason_lines, [x["text"] for x in segments], product], ensure_ascii=False))
        audio_plan = cache.get(("rewrite", script_key), lambda: _persisted(
            work / "rewrite.json", lambda: plan_edits(client, segments, language, reason_lines, hint, product, policy)))
    if not audio_plan["edits"]:
        return new_vocals
    by_id = {s["id"]: s for s in segments}
    refs, ref_text = _speaker_refs(vocals, sr, segments, work)
    fade = int(FADE * sr)
    for idx, e in enumerate(audio_plan["edits"]):
        first, last = by_id[e["segment_ids"][0]], by_id[e["segment_ids"][-1]]
        prev_end = max((s["end"] for s in segments if s["end"] <= first["start"]), default=0.0)
        next_start = min((s["start"] for s in segments if s["start"] >= last["end"]), default=len(vocals) / sr)
        start = max(prev_end, first["start"] - 0.05)
        end = min(next_start, last["end"] + 0.1)
        edit = Edit(
            segment_ids=e["segment_ids"], start=round(start, 3), end=round(end, 3),
            original_text=" ".join(by_id[i]["text"] for i in e["segment_ids"]),
            new_text=e["new_text"], reason=e["reason"],
        )
        with timed("tts"):
            tts = _render_edit(client, edit, language, refs, ref_text, end - start, work, idx, cache, script_key,
                               product)
        s, e_ = int(start * sr), int(end * sr)
        target = _rms(vocals[s:e_])
        if target and (cur := _rms(tts)):
            tts *= target / cur
        n = min(len(tts), len(new_vocals) - s)
        _fade_mute(new_vocals, s, max(e_, s + n), fade)
        new_vocals[s : s + n] += tts[:n]
        result.edits.append(edit)
    return new_vocals


def _ocr_and_triage(ctx: dict, video: Path, work: Path, segments, language, planner_frames, video_key):
    """OCR chạy nền (cần ngôn ngữ để chọn bộ chữ), song song với lượt triage. Trả (quick, tracks)."""
    result, cache, client, timed = ctx["result"], ctx["cache"], ctx["client"], ctx["timed"]
    stop_ocr = threading.Event()
    if rc := runctx.current():
        rc.on_kill(stop_ocr.set)
    tracks_file = work / "tracks.pkl"

    def run_ocr():
        import pickle

        if tracks_file.exists():
            return pickle.loads(tracks_file.read_bytes())
        t0 = time.perf_counter()
        tracks = vision.ocr_tracks(vision.sample_frames(video, settings.ocr_fps, 960), language or "en", settings.ocr_fps, stop_ocr)
        result.timings["ocr"] = round(result.timings.get("ocr", 0) + time.perf_counter() - t0, 2)
        if not stop_ocr.is_set():
            tracks_file.write_bytes(pickle.dumps(tracks))
        return tracks

    with ThreadPoolExecutor(1) as bg:
        ocr_future = bg.submit(lambda: cache.get(("ocr", video_key), run_ocr))
        quick = {"decision": "fix"}
        if ctx["pass_no"] == 1:
            with timed("triage"):
                quick = cache.get(("triage", video_key, tuple(result.reject_reasons), ctx["declared"]), lambda: _persisted(
                    work / "triage.json", lambda: triage(client, ctx["reasons"], segments, planner_frames,
                                                         ctx["violation_images"], ctx["hint"], ctx["declared"],
                                                         ctx["policy"])))
        if quick["decision"] == "unfixable":
            stop_ocr.set()
            return quick, []
        with runctx.stage("ocr"):
            while True:
                try:
                    return quick, ocr_future.result(timeout=1)
                except TimeoutError:
                    runctx.check()


def _merge_overlays(actions: list[dict], language: str | None) -> list[dict]:
    """Điền text disclaimer theo ngôn ngữ, gộp nhiều disclaimer thành 1 dòng (cùng vị trí trên khung)."""
    disc = [a for a in actions if a["type"] == "add_disclaimer"]
    rest = [a for a in actions if a["type"] != "add_disclaimer"]
    texts = [t for t in dict.fromkeys(policies.disclaimer(a["key"], language) for a in disc) if t]
    if texts:
        rest.append({"type": "add_disclaimer", "keys": [a["key"] for a in disc], "text": " · ".join(texts),
                     "start": min(a["start"] for a in disc), "end": max(a["end"] for a in disc)})
    return rest


def _fix_pass(ctx: dict, video: Path, work: Path, out_path: Path, feedback: list[dict] | None = None) -> bool:
    """1 lượt phân tích + sửa + render. True = đã render video mới vào out_path."""
    result, cache, client, timed, item = ctx["result"], ctx["cache"], ctx["client"], ctx["timed"], ctx["item"]
    reasons, declared, hint = ctx["reasons"], ctx["declared"], ctx["hint"]
    work.mkdir(parents=True, exist_ok=True)
    video_key = _digest(video.read_bytes())
    info = vision.video_info(video)
    if info.duration > settings.max_video_seconds:
        raise InvalidInput(f"video dài {info.duration:.0f}s, vượt giới hạn {settings.max_video_seconds}s")
    has_audio = media.has_audio(video)

    # --- phân tích audio: tách giọng + transcript ---
    segments, language, vocals, bgm, sr = [], None, None, None, 0
    if has_audio:
        with timed("separate"):
            mix_wav = media.extract_audio(video, work / "mix.wav")
            audio_key = _digest(mix_wav.read_bytes())
            vocals, bgm, sr = cache.get(("separate", audio_key), lambda: _separate_cached(mix_wav, work))
        vocals_path = work / "vocals.wav"
        if not vocals_path.exists():  # item trùng audio lấy stem từ cache của item khác
            sf.write(vocals_path, vocals, sr, subtype="FLOAT")
        with timed("transcribe"):
            tr = cache.get(("transcribe", audio_key), lambda: _persisted(
                work / "transcript.json",
                lambda: transcribe(client, media.to_mono_16k(vocals_path, work / "vocals_16k.wav"))))
        segments, language = tr["segments"], tr["language"]
        result.language = result.language or language

    planner_frames = _even(vision.sample_frames(video, settings.planner_fps, 384), settings.max_planner_frames)
    quick, tracks = _ocr_and_triage(ctx, video, work, segments, language, planner_frames, video_key)
    if quick["decision"] == "unfixable":
        result.plan = {"decision": "unfixable", "reason": quick["reason"], "product": quick.get("product", ""),
                       "stage": "triage"}
        result.policy_findings = [{"reason_index": i, "group": (gs or [None])[0], "verdict": "unfixable",
                                   "fix": quick["reason"]} for i, gs in enumerate(ctx["groups"])]
        result.state, result.error = "no_changes", quick["reason"] or "không sửa được bằng cách chỉnh video"
        return False
    track_map = {t.id: t for t in tracks}

    with timed("plan"):
        plan_key = _digest(json.dumps([video_key, [r.model_dump() for r in reasons], declared, feedback],
                                      ensure_ascii=False))
        plan = cache.get(("plan", plan_key), lambda: _persisted(work / "plan_raw.json", lambda: plan_fix(
            client, reasons, info.duration, segments, language, tracks, planner_frames, ctx["violation_images"], hint,
            declared, ctx["policy"], feedback)))
    if ctx["pass_no"] == 1:
        result.plan = plan
        result.policy_findings = plan.get("findings") or []
    else:
        result.plan = {**result.plan, "pass2": plan}
    ctx["actions_required"] += plan.get("action_required") or []
    (work / "plan.json").write_text(json.dumps({"plan": plan, "tracks": [t.public() for t in tracks]},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("ad %s plan (pass %d): %s", item.smart_ad_id, ctx["pass_no"], json.dumps(plan, ensure_ascii=False)[:500])
    if plan["decision"] != "fix":
        if ctx["pass_no"] == 1:
            result.state, result.error = "no_changes", plan["reason"] or "không sửa được bằng cách chỉnh video"
        return False

    for a in plan["visual_actions"]:
        if a["type"] == "cut" and "planned" not in a:
            a["planned"] = [a["start"], a["end"]]
            a["start"], a["end"] = video_edit.snap_cut(video, info, a["start"], a["end"])
    cuts = sorted((a["start"], a["end"]) for a in plan["visual_actions"] if a["type"] == "cut")
    audio_is_cut = False
    if cuts and has_audio and plan["fix_audio"] and segments and language:
        # Cắt stem trước rồi mới viết lại + TTS: câu mới được sinh vừa khung trên timeline đã cắt,
        # không bị cắt mất giữa câu như khi cắt audio sau cùng
        vocals, bgm = video_edit.cut_audio(vocals, sr, cuts), video_edit.cut_audio(bgm, sr, cuts)
        segments = _remap_segments(segments, cuts)
        audio_is_cut = True

    new_vocals = vocals
    n_edits = len(result.edits)
    if plan["fix_audio"] and segments and language:
        new_vocals = _rewrite_voiceover(client, result, segments, language, vocals, sr, work, cache, hint, timed,
                                        declared or plan.get("product"), ctx["policy"])
    pass_edits = result.edits[n_edits:]

    lang = language or result.language or ctx["market_language"]
    actions = _merge_overlays(list(plan["visual_actions"]), lang)
    if plan["sync_captions"] and pass_edits:
        taken = {i for a in actions for i in a.get("track_ids", [])}
        # text track theo timeline gốc; edit theo timeline đã cắt (nếu đã cắt audio trước)
        edits = [e.model_copy(update={"start": _from_cut(e.start, cuts), "end": _from_cut(e.end, cuts)})
                 for e in pass_edits] if audio_is_cut else pass_edits
        actions += _caption_actions(edits, tracks, taken)
    if pass_edits and settings.aigc_overlay and not ctx["aigc_added"]:
        # Policy "Edited media and AIGC": giọng clone nói câu người thật chưa nói => phải công bố
        actions.append({"type": "aigc_label", "text": policies.disclaimer("aigc_voice", lang),
                        "start": 0.0, "end": info.duration})
        ctx["aigc_added"] = True
    if not pass_edits and not actions:
        if ctx["pass_no"] == 1:
            result.state, result.error = "no_changes", plan["notes"] or "planner không đưa ra thao tác sửa nào"
        return False

    mixed = None
    if has_audio:
        mix = new_vocals + bgm
        peak = float(np.abs(mix).max())
        if peak > 0.99:
            mix *= 0.99 / peak
        mixed = work / "final_mix.wav"
        sf.write(mixed, mix, sr, subtype="FLOAT")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visual = [a for a in actions if a["type"] != "cut"]
    if visual or cuts:
        with timed("render"):
            regions = video_edit.build_regions(visual, track_map, info, lambda t: vision.frame_at(video, t))
            video_edit.render(video, mixed, out_path, info, regions, cuts, audio_is_cut)
    else:
        with timed("mux"):
            media.mux(video, mixed, out_path)
    result.visual_edits += [
        {**a, "pass": ctx["pass_no"],
         **({"texts": [track_map[i].text for i in a["track_ids"]]} if "track_ids" in a else {})}
        for a in actions]
    return True


def _ad_text(client, item: AdItem, reasons: list[ReviewReason], policy: dict, product: str | None) -> dict | None:
    """Caption: chạy sau lượt sửa video để biết sản phẩm thật (declared hoặc planner xác định)."""
    from . import adcopy

    if not item.ad_text:
        return None
    try:
        return adcopy.rewrite_ad_text(client, item.ad_text, reasons, policy, product, item.call_to_action)
    except Exception as exc:
        log.warning("ad text rewrite failed: %s", exc)
        return {"original": item.ad_text, "error": str(exc)[:300]}


def _side_checks(item: AdItem, reasons: list[ReviewReason], policy: dict, declared: str | None, usage: dict) -> dict:
    """Landing page, chạy song song với phần video (không cần GPU)."""
    from . import landing

    out: dict = {}
    if item.landing_page_url:
        try:
            # so khớp nội dung làm sau, khi đã biết sản phẩm thật + caption mới
            out["landing_page"] = landing.check_landing(None, item.landing_page_url, declared, None)
        except Exception as exc:
            log.warning("landing page check failed: %s", exc)
            out["landing_page"] = {"url": item.landing_page_url, "ok": False, "issues": [f"không kiểm tra được: {exc}"]}
    return out


def process_item(
    item: AdItem, result: ItemResult, reasons: list[ReviewReason], work: Path, out_path: Path,
    campaign: str | None, cache: JobCache | None = None,
) -> None:
    cache = cache or JobCache()
    client = _openai_client(result.usage)
    timed = lambda stage: _timed(result.timings, stage)
    work.mkdir(parents=True, exist_ok=True)
    result.reject_reasons = [r.summary() for r in reasons]
    if not reasons:
        result.state, result.error = "no_changes", "không có reject reason"
        return
    if not item.video_url:
        raise ValueError("thiếu video_url")

    with timed("download"):
        video = work / "source.mp4"
        if not video.exists():
            media.download(item.video_url, video)

    declared = (item.product or "").strip() or None
    with timed("policy"):
        groups = _persisted(work / "policy_groups.json",
                            lambda: policies.classify_reasons(client, reasons, settings.vision_model))
    policy = policies.policy_context(reasons, groups)
    result.targeting_fix = policies.targeting_fix(reasons, groups)
    regions = [x for r in reasons for x in r.affected_regions]
    ctx = {
        "item": item, "result": result, "reasons": reasons, "cache": cache, "client": client, "timed": timed,
        "declared": declared, "groups": groups, "policy": policy, "pass_no": 1, "aigc_added": False,
        "actions_required": [], "market_language": policies.language_for_regions(regions) or "en",
        "hint": " | ".join(x for x in (campaign, item.smart_ad_name, Path(urlparse(item.video_url).path).name) if x),
        "violation_images": _download_images([u for r in reasons for u in r.violation_frames] + item.violation_frames,
                                             work),
    }
    side_usage: dict[str, float] = {}
    with ThreadPoolExecutor(1) as side_pool:
        side = side_pool.submit(_side_checks, item, reasons, policy, declared, side_usage)
        rendered = _fix_pass(ctx, video, work, out_path)
        if rendered and settings.recheck:
            with timed("recheck"):
                chk = recheck.recheck(client, out_path, reasons, policy, [e.model_dump() for e in result.edits],
                                      result.visual_edits, ctx["hint"], declared, work)
            passes = 1
            if not chk["passed"] and settings.recheck_retry:
                # Lượt 2: sửa tiếp trên chính video đã sửa, kèm danh sách vấn đề recheck tìm thấy
                work2 = work / "pass2"
                work2.mkdir(parents=True, exist_ok=True)
                src2, out2 = work2 / "source.mp4", work2 / "out.mp4"
                if not src2.exists():
                    shutil.copyfile(out_path, src2)
                ctx["pass_no"] = passes = 2
                if _fix_pass(ctx, src2, work2, out2, chk["issues"]):
                    shutil.copyfile(out2, out_path)
                    with timed("recheck"):
                        chk = recheck.recheck(client, out_path, reasons, policy, [e.model_dump() for e in result.edits],
                                              result.visual_edits, ctx["hint"], declared, work2)
            result.compliance_check = {"passed": chk["passed"], "issues": chk["issues"], "passes": passes}
        side_out = side.result()
    for k, v in side_usage.items():
        result.usage[k] = result.usage.get(k, 0) + v
    product = declared or (result.plan or {}).get("product") or None
    result.ad_text_fix = _ad_text(client, item, reasons, policy, product)
    result.landing_page = side_out.get("landing_page")
    if result.landing_page and result.landing_page.get("final_url"):
        from . import landing

        new_text = (result.ad_text_fix or {}).get("new") or item.ad_text
        try:
            result.landing_page = landing.check_consistency(client, result.landing_page, product, new_text)
        except Exception as exc:
            log.warning("landing consistency check failed: %s", exc)
        result.landing_page.pop("_text", None)

    # --- việc ngoài video ---
    required = list(ctx["actions_required"])
    if result.edits:
        required.append({"type": "aigc_label", "detail": "Voice-over được clone bằng AI: bật nhãn AIGC khi đăng ad "
                                                          "(policy Edited media and AIGC)."})
    if result.targeting_fix and (result.targeting_fix["exclude_regions"] or result.targeting_fix["exclude_placements"]):
        tf = result.targeting_fix
        required.append({"type": "targeting", "detail": "Loại khỏi targeting: " + ", ".join(
            tf["exclude_regions"] + tf["exclude_placements"]) + (f"; tuổi tối thiểu {tf['min_age']}+" if tf["min_age"] else "")})
    if (fix := result.ad_text_fix) and fix.get("changed"):
        required.append({"type": "ad_text", "detail": f"Thay caption bằng: {fix['new']}"})
    elif fix and fix.get("compliant") is False:
        required.append({"type": "ad_text", "detail": "Caption vi phạm, cần viết caption mới: " + "; ".join(fix["issues"])})
    if (lp := result.landing_page) and lp.get("ok") is False:
        required.append({"type": "landing_page", "detail": "; ".join(lp.get("issues") or [])[:500]})

    if rendered:
        if result.compliance_check and not result.compliance_check["passed"]:
            result.state = "needs_review"
            result.error = "; ".join(i["problem"] for i in result.compliance_check["issues"])[:1000]
        else:
            outside = [a for a in result.action_required if a["type"] not in ("aigc_label",)]
            result.state = "partial" if outside else "done"
