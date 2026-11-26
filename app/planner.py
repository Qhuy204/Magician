"""Planner: đọc lý do reject + transcript + text trên màn hình + frame video, chọn công cụ sửa cho từng vi phạm."""
import json
import logging

from openai import OpenAI

from .config import settings
from .schemas import ReviewReason
from .vision import TextTrack, jpeg_data_url

log = logging.getLogger(__name__)

# remove_region lớn hơn tỉ lệ này của khung hình thì đổi thành cut
MAX_INPAINT_AREA = 0.45

SYSTEM = """You are the compliance editor for TikTok video ads. Given TikTok's rejection reasons, the voice-over transcript,
the on-screen text detected by OCR (text tracks with ids, time ranges and normalized bounding boxes [x0,y0,x1,y1]),
and video frames (one per second, each labelled with its timestamp), decide how to fix the ad with these tools:

- fix_audio: true when the spoken voice-over itself is part of a violation. A separate step will rewrite the violating
  spoken lines; you do not write them here.
- sync_captions: true when on-screen text is burned-in captions/subtitles that repeat the voice-over, so captions get
  updated automatically after the voice-over is rewritten.
- replace_text {track_ids, new_text}: erase on-screen text and draw compliant text in the same place, same language.
- remove_text {track_ids}: erase on-screen text with nothing in its place.
- remove_region {start, end, bbox, what}: erase a non-text graphic (pop-up, banner, cash counter, badge, logo) inside
  bbox during [start, end]. The bbox must enclose the whole graphic including its text, shadow and border.
- cut {start, end, what}: drop a time range from the video and audio, for shots whose whole content violates policy
  (e.g. a full-screen payout screen). Cuts must not split a spoken sentence and should follow shot boundaries.

Rules:
- Fix the violation for real: remove the violating claim or visual. Never keep it by rewording it vaguely,
  misspelling it, partially hiding it, or moving it elsewhere.
- Do not invent product claims. replace_text must describe only what the ad already legitimately shows.
- Use the given timestamps to locate violations, but check neighbouring frames: the same element often appears longer.
  Cover the whole time the element is visible.
- Prefer the least destructive tool: replace_text/remove_text/remove_region before cut. But when the violating element
  fills most of the frame or the whole phone screen (e.g. a full-screen form or payout screen), use cut: inpainting
  such a large area only produces a blur. A voice-over line that runs across the cut is fine when fix_audio is true
  (the new line is generated for the shortened timeline).
- First identify the product actually advertised from the visuals (end card, app name/logo, store badges, app UI) and
  what it legitimately does. Report it in "product".
- Distinguish a FALSE CLAIM about a legitimate product from a PROHIBITED PRODUCT:
  * False claim (fixable): the product shown is a legitimate app (e.g. phone-to-phone data transfer/backup, call
    filtering, caller ID), but the voice-over or on-screen text claims a prohibited capability it does not provide
    (e.g. "see other people's WhatsApp chats by entering their number"). Fix it: fix_audio so the voice-over describes
    only the product's real, legitimate function, and remove/cut every visual that depicts the false capability
    (e.g. a screen where someone else's phone number is typed in to unlock their chats, captions repeating the claim).
    The voice-over being the violating part throughout is NOT a reason for "unfixable" in this case: it is rewritten.
    Judge the visuals as they will read AFTER the rewrite: ordinary app screens (a chat list, a conversation, photos)
    are not violations by themselves when the new voice-over presents them as the user's own data (e.g. chats moved
    from the old phone to the new one). Only shots that specifically show the false capability need remove/cut
    (typing another person's number/name to unlock their data, "spy"/"hack" overlays, captions repeating the claim).
    Count only those shots against the cut limit.
  * Prohibited product (unfixable): the product itself provides the prohibited capability (spyware/stalkerware,
    account hacking), or the business is prohibited (gambling, chance-based rewards, apps whose core offer is paying
    users cash for watching/playing, prohibited products, impersonation).
- Return "unfixable" also when the violating VISUALS run through most of the video, or the required cuts exceed
  {max_cut_pct}% of the duration.
- Violations outside the video (landing page, targeting, app store page) cannot be fixed here: list them in
  out_of_scope, and still fix what is in the video.
Respond with JSON only:
{"decision": "fix" | "unfixable", "reason": str, "product": str, "fix_audio": bool, "sync_captions": bool,
 "visual_actions": [{"type": "replace_text"|"remove_text"|"remove_region"|"cut", ...}],
 "out_of_scope": [str], "notes": str}"""


TRIAGE = """You review rejected TikTok video ads. Decide quickly, before any detailed editing plan, whether the ad can be
made compliant by editing the creative (rewriting voice-over, replacing/removing on-screen text or graphics, cutting
short shots).
It cannot ("unfixable") when the violation is about the product or business itself (gambling, chance-based rewards,
apps whose core offer is paying users cash for watching/playing, prohibited products such as spyware/stalkerware or
account hacking tools, impersonation), or when violating VISUALS run through most of the video.
Identify the product from the visuals (end card, app name/logo, store badges). If it is a legitimate app and the
violation is a false claim in the voice-over or text about a capability the product does not provide (e.g. claiming a
data-transfer app lets you read other people's chats), it is fixable: the voice-over can be fully rewritten to the
product's real function and the shots depicting the false capability removed, even if the claim is in every spoken line.
A single misleading pop-up, badge, caption or exaggerated claim on an otherwise legitimate product is fixable.
Answer "unfixable" only when you are certain; when in doubt answer "fix" (a detailed planner checks again later).
Respond with JSON only: {"decision": "fix" | "unfixable", "reason": str, "product": str}"""


def _frame_content(frames: list, violation_images: list[str]) -> list[dict]:
    content: list[dict] = []
    for url in violation_images[:6]:
        content += [{"type": "text", "text": "TikTok violation screenshot:"},
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}}]
    for t, img in frames:
        content += [{"type": "text", "text": f"frame t={t:.1f}s"},
                    {"type": "image_url", "image_url": {"url": jpeg_data_url(img), "detail": "low"}}]
    return content


def triage(client: OpenAI, reasons: list[ReviewReason], segments: list[dict], frames: list,
           violation_images: list[str], hint: str | None) -> dict:
    """Lượt lọc nhanh (chạy song song với OCR): ad hỏng ở bản chất sản phẩm thì dừng sớm, khỏi chờ OCR."""
    payload = {"ad": hint, "rejection_reasons": [r.model_dump(exclude={"violation_frames"}, exclude_defaults=True) for r in reasons],
               "voice_over": " ".join(s["text"] for s in segments)}
    r = client.chat.completions.create(
        model=settings.vision_model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": TRIAGE},
                  {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                                               *_frame_content(frames, violation_images)]}],
    )
    out = json.loads(r.choices[0].message.content or "{}")
    return {"decision": "unfixable" if out.get("decision") == "unfixable" else "fix", "reason": out.get("reason", ""),
            "product": out.get("product", "")}


def _track_digest(tracks: list[TextTrack], limit: int = 150) -> list[dict]:
    """Rút gọn danh sách track để đưa vào prompt: bỏ track nhiễu, ưu tiên track rõ + lâu + to."""
    useful = [t for t in tracks if t.confidence >= 0.45 or t.end - t.start >= 1.0]
    useful.sort(key=lambda t: -(t.confidence * (t.end - t.start) * (t.bbox[2] - t.bbox[0])))
    return sorted((t.public() for t in useful[:limit]), key=lambda d: d["start"])


def plan_fix(
    client: OpenAI, reasons: list[ReviewReason], duration: float, segments: list[dict], language: str | None,
    tracks: list[TextTrack], frames: list[tuple[float, "object"]], violation_images: list[str], hint: str | None,
) -> dict:
    payload = {
        "ad": hint,
        "video_duration_s": round(duration, 2),
        "language": language,
        "rejection_reasons": [r.model_dump(exclude={"violation_frames"}, exclude_defaults=True) for r in reasons],
        "voice_over_segments": segments,
        "on_screen_text_tracks": _track_digest(tracks),
    }
    content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}, *_frame_content(frames, violation_images)]
    r = client.chat.completions.create(
        model=settings.vision_model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM.replace("{max_cut_pct}", str(int(settings.max_cut_ratio * 100)))},
                  {"role": "user", "content": content}],
    )
    plan = json.loads(r.choices[0].message.content or "{}")
    return _validate(plan, tracks, duration)


def _validate(plan: dict, tracks: list[TextTrack], duration: float) -> dict:
    """Bỏ action sai định dạng, kẹp toạ độ/thời gian, áp giới hạn tổng thời lượng cắt."""
    ids = {t.id for t in tracks}
    actions = []
    for a in plan.get("visual_actions") or []:
        kind = a.get("type")
        if kind in ("replace_text", "remove_text"):
            tids = [int(i) for i in a.get("track_ids") or [] if int(i) in ids]
            if not tids or (kind == "replace_text" and not (a.get("new_text") or "").strip()):
                continue
            actions.append({"type": kind, "track_ids": tids, **({"new_text": a["new_text"].strip()} if kind == "replace_text" else {})})
        elif kind in ("remove_region", "cut"):
            try:
                start, end = max(0.0, float(a["start"])), min(duration, float(a["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end - start < 0.05:
                continue
            act = {"type": kind, "start": round(start, 2), "end": round(end, 2), "what": a.get("what", "")}
            if kind == "remove_region":
                try:
                    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in a["bbox"])
                except (KeyError, TypeError, ValueError):
                    continue
                if x1 - x0 < 0.01 or y1 - y0 < 0.01:
                    continue
                if (x1 - x0) * (y1 - y0) > MAX_INPAINT_AREA:
                    # inpaint vùng gần cả khung chỉ ra 1 mảng nhoè => cắt cả đoạn
                    act |= {"type": "cut", "what": f"{act['what']} (vùng quá lớn để xoá, chuyển sang cắt)"}
                    actions.append(act)
                    continue
                act |= {"bbox": [x0, y0, x1, y1], "method": "blur" if a.get("method") == "blur" else "inpaint"}
            actions.append(act)
    cut_total = sum(a["end"] - a["start"] for a in actions if a["type"] == "cut")
    decision = plan.get("decision") if plan.get("decision") in ("fix", "unfixable") else "unfixable"
    reason = plan.get("reason", "")
    if decision == "fix" and cut_total > settings.max_cut_ratio * duration:
        decision, reason = "unfixable", f"cần cắt {cut_total:.1f}s / {duration:.1f}s, vượt giới hạn {settings.max_cut_ratio:.0%}"
    if decision == "fix" and not actions and not plan.get("fix_audio"):
        decision, reason = "unfixable", reason or "không tìm thấy vi phạm trong video để sửa"
    return {
        "decision": decision, "reason": reason, "product": plan.get("product", ""),
        "fix_audio": bool(plan.get("fix_audio")) and decision == "fix",
        "sync_captions": bool(plan.get("sync_captions")) and decision == "fix",
        "visual_actions": actions if decision == "fix" else [], "out_of_scope": plan.get("out_of_scope") or [],
        "notes": plan.get("notes", ""),
    }
