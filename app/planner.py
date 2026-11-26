"""Planner: đọc lý do reject + playbook policy theo nhóm + transcript + text trên màn hình + frame video,
chọn công cụ sửa cho từng vi phạm và báo việc cần làm ngoài video."""
import json
import logging

from openai import OpenAI

from . import policies
from .config import settings
from .schemas import ReviewReason
from .vision import TextTrack, jpeg_data_url

log = logging.getLogger(__name__)

VERDICTS = ("fixed_in_video", "outside_video", "unfixable", "not_found")
# Tăng khi đổi SYSTEM: cache key của plan gồm version này để prompt mới không dùng lại plan cũ
PLAN_PROMPT_VERSION = 2


SYSTEM = """You are the compliance editor for TikTok video ads. Given TikTok's rejection reasons, the TikTok policy
playbooks for the policy groups those reasons belong to ("policy"), the voice-over transcript, the on-screen text
detected by OCR (text tracks with ids, time ranges and normalized bounding boxes [x0,y0,x1,y1]), and video frames (one
per second, each labelled with its timestamp), decide how to fix the ad with these tools:

- fix_audio: true when the spoken voice-over itself is part of a violation. A separate step will rewrite the violating
  spoken lines; you do not write them here.
- sync_captions: true when on-screen text is burned-in captions/subtitles that repeat the voice-over, so captions get
  updated automatically after the voice-over is rewritten.
- replace_text {track_ids, new_text}: erase on-screen text and draw compliant text in the same place, same language.
  Also used to correct spelling / excessive capitalization / symbols among letters.
- remove_text {track_ids}: erase on-screen text with nothing in its place.
- remove_region {start, end, bbox, what}: blur a non-text graphic or violating visual content inside bbox during
  [start, end]. The bbox must enclose the whole violating content including its text, shadow and border. Use it for
  pop-ups, banners, cash counters, badges, watermarks, creator handles, TikTok logos, unauthorized brand logos,
  QR codes, fake buttons, personal data, suggestive/gory imagery - and also for whole shots whose content violates
  policy (then use a full-frame or near-full-frame bbox). Blurring is the ONLY visual removal method: the video is
  never cut and nothing is ever inpainted, so pick the bbox generously enough that nothing violating stays readable
  or recognizable. Watermarks/handles/third-party logos must be blurred even when no rejection reason mentions them.
- add_disclaimer {key, start, end}: overlay a required disclaimer line. key is one of the playbook "disclaimers"
  (results_vary, terms_apply, financial_risk, responsible_drinking, gambling_responsible, healthy_lifestyle,
  consult_professional). Default start/end = whole video. The text is filled in the ad's language automatically.
  (The AI-voice disclosure is added automatically when the voice-over is rewritten; do not add it.)

Your role:
- You are a media editor, not a product judge. NEVER refuse to fix because of what the product or business is: you
  do not decide whether the advertiser may sell it. Even when the rejection is about a prohibited or restricted
  product category, you still clean up the video: rewrite voice-over and text so they no longer state or depict the
  prohibited capability, and blur every visual that depicts it.
- First identify the product actually advertised from the visuals (end card, app name/logo, store badges, app UI) and
  what it legitimately does. Report it in "product". "declared_product", when given, is the advertiser's statement of
  the real product and is authoritative over what the footage suggests.
- Rewrite direction for fix_audio: the new voice-over describes only the product's real, legitimate function (a
  phone-to-phone transfer app moves YOUR OWN chats/photos to a new phone; a caller-ID app filters spam calls). It
  must never restate or hint at the prohibited capability.
- Judge the visuals as they will read AFTER the rewrite: ordinary app screens (a chat list, a conversation, photos)
  are not violations by themselves when the new voice-over presents them as the user's own data. Blur only shots that
  specifically depict the prohibited capability (typing another person's number/name to unlock their data, "spy" or
  "hack" overlays, cash/payout screens, suggestive or gory imagery, captions repeating the claim).

Policy:
- Follow "policy.global_rules" and every playbook in "policy.playbooks". "policy.reason_groups" maps each rejection
  reason (by index) to its policy groups.
- For each rejection reason output one finding:
  {"reason_index", "group", "verdict", "fix"}, verdict one of:
  * "fixed_in_video": the actions you list remove this violation from the video,
  * "outside_video": the violation is only about targeting, landing page, certification, rights, ad caption or
    fulfillment, with nothing violating in the video itself; list what must be done in action_required,
  * "not_found": nothing in the video matches the reason.
- action_required items: {"type": one of targeting | certificate | landing_page | ad_text | rights | new_creative |
  fulfillment | account | review, "detail": str}. These are informational for the advertiser; they never replace
  fixing the video. Only include them when they are clearly relevant.

Rules:
- Fix the violation for real: remove the violating claim or blur the violating visual so nothing of it stays
  readable or recognizable. Never keep it by rewording it vaguely, misspelling it, or moving it elsewhere.
- Do not invent product claims. replace_text must describe only what the ad already legitimately shows.
- Use the given timestamps to locate violations, but check neighbouring frames: the same element often appears
  longer. Cover the whole time the element is visible.
- decision "fix" whenever anything in the video needs changing. decision "outside_only" only when nothing in the
  video needs changing at all (every finding is outside_video / not_found). Never output "unfixable".
- "review_feedback", when present, means this video was ALREADY edited once and a compliance check found these
  remaining problems: fix only them. Disclaimers / the AI-voice label already visible are intended: do not add them
  again or remove them. Blur patches from the earlier edit are intentional: do not blur them again, only enlarge or
  extend them when the feedback says the violation is still visible.
Respond with JSON only:
{"decision": "fix" | "outside_only", "reason": str, "product": str, "fix_audio": bool,
 "sync_captions": bool,
 "visual_actions": [{"type": "replace_text"|"remove_text"|"remove_region"|"add_disclaimer", ...}],
 "findings": [{"reason_index": int, "group": str, "verdict": str, "fix": str}],
 "action_required": [{"type": str, "detail": str}], "notes": str}"""


def _frame_content(frames: list, violation_images: list[str]) -> list[dict]:
    content: list[dict] = []
    for url in violation_images[:6]:
        content += [{"type": "text", "text": "TikTok violation screenshot:"},
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}}]
    for t, img in frames:
        content += [{"type": "text", "text": f"frame t={t:.1f}s"},
                    {"type": "image_url", "image_url": {"url": jpeg_data_url(img), "detail": "low"}}]
    return content


def _reasons_payload(reasons: list[ReviewReason]) -> list[dict]:
    return [{"index": i, **r.model_dump(exclude={"violation_frames"}, exclude_defaults=True)} for i, r in enumerate(reasons)]


def _track_digest(tracks: list[TextTrack], limit: int = 150) -> list[dict]:
    """Rút gọn danh sách track để đưa vào prompt: bỏ track nhiễu, ưu tiên track rõ + lâu + to."""
    useful = [t for t in tracks if t.confidence >= 0.45 or t.end - t.start >= 1.0]
    useful.sort(key=lambda t: -(t.confidence * (t.end - t.start) * (t.bbox[2] - t.bbox[0])))
    return sorted((t.public() for t in useful[:limit]), key=lambda d: d["start"])


def plan_fix(
    client: OpenAI, reasons: list[ReviewReason], duration: float, segments: list[dict], language: str | None,
    tracks: list[TextTrack], frames: list[tuple[float, "object"]], violation_images: list[str], hint: str | None,
    product: str | None = None, policy: dict | None = None, review_feedback: list[dict] | None = None,
) -> dict:
    payload = {
        "ad": hint,
        "declared_product": product,
        "video_duration_s": round(duration, 2),
        "language": language,
        "policy": policy,
        "rejection_reasons": _reasons_payload(reasons),
        "voice_over_segments": segments,
        "on_screen_text_tracks": _track_digest(tracks),
    }
    if review_feedback:
        payload["review_feedback"] = review_feedback
    content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}, *_frame_content(frames, violation_images)]
    r = client.chat.completions.create(
        model=settings.vision_model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": content}],
    )
    plan = json.loads(r.choices[0].message.content or "{}")
    return _validate(plan, tracks, duration, len(reasons))


def _findings(plan: dict, n_reasons: int) -> list[dict]:
    groups = {g for g in policies.catalog().groups}
    out = []
    for f in plan.get("findings") or []:
        try:
            idx = int(f.get("reason_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < max(n_reasons, 1):
            continue
        out.append({"reason_index": idx, "group": f.get("group") if f.get("group") in groups else None,
                    "verdict": f.get("verdict") if f.get("verdict") in VERDICTS else "not_found",
                    "fix": str(f.get("fix") or "")[:500]})
    return out


def _actions_required(plan: dict) -> list[dict]:
    out = []
    for a in plan.get("action_required") or []:
        kind = a.get("type") if isinstance(a, dict) else None
        if kind in policies.ACTION_TYPES and (a.get("detail") or "").strip():
            out.append({"type": kind, "detail": a["detail"].strip()[:500]})
    return out


def _validate(plan: dict, tracks: list[TextTrack], duration: float, n_reasons: int = 0) -> dict:
    """Bỏ action sai định dạng, kẹp toạ độ/thời gian. Mọi xoá hình đều là blur; cut do model hallucinate
    thì đổi thành blur toàn khung (video không bao giờ bị cắt)."""
    ids = {t.id for t in tracks}
    disclaimer_keys = set(policies.catalog().disclaimers) - {"aigc_voice"}
    actions = []
    for a in plan.get("visual_actions") or []:
        kind = a.get("type")
        if kind in ("replace_text", "remove_text"):
            tids = [int(i) for i in a.get("track_ids") or [] if str(i).lstrip("-").isdigit() and int(i) in ids]
            if not tids or (kind == "replace_text" and not (a.get("new_text") or "").strip()):
                continue
            actions.append({"type": kind, "track_ids": tids, **({"new_text": a["new_text"].strip()} if kind == "replace_text" else {})})
        elif kind == "add_disclaimer":
            if a.get("key") not in disclaimer_keys:
                continue
            try:
                start = max(0.0, float(a.get("start", 0.0)))
                end = min(duration, float(a.get("end", duration)))
            except (TypeError, ValueError):
                start, end = 0.0, duration
            if end - start < 1.0:
                start, end = 0.0, duration
            if not any(x["type"] == "add_disclaimer" and x["key"] == a["key"] for x in actions):
                actions.append({"type": "add_disclaimer", "key": a["key"], "start": round(start, 2), "end": round(end, 2)})
        elif kind in ("remove_region", "cut"):
            try:
                start, end = max(0.0, float(a["start"])), min(duration, float(a["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end - start < 0.05:
                continue
            what = a.get("what", "")
            if kind == "cut":
                # model không còn được dùng cut; nếu vẫn sinh ra thì blur toàn khung đoạn đó
                actions.append({"type": "remove_region", "start": round(start, 2), "end": round(end, 2),
                                "bbox": [0.0, 0.0, 1.0, 1.0], "method": "blur",
                                "what": f"{what} (toàn cảnh)"})
                continue
            try:
                x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in a["bbox"])
            except (KeyError, TypeError, ValueError):
                continue
            if x1 - x0 < 0.01 or y1 - y0 < 0.01:
                continue
            actions.append({"type": "remove_region", "start": round(start, 2), "end": round(end, 2),
                            "bbox": [x0, y0, x1, y1], "method": "blur", "what": what})
    # quyết định derive từ nội dung: có gì để sửa thì fix, không có gì thì outside_only
    decision = "fix" if (plan.get("fix_audio") or actions) else "outside_only"
    reason = plan.get("reason", "")
    findings, required = _findings(plan, n_reasons), _actions_required(plan)
    fixing = decision == "fix"
    return {
        "decision": decision, "reason": reason, "product": plan.get("product", ""),
        "fix_audio": bool(plan.get("fix_audio")) and fixing,
        "sync_captions": bool(plan.get("sync_captions")) and fixing,
        "visual_actions": actions if fixing else [],
        "findings": findings, "action_required": required,
        "out_of_scope": [f"{a['type']}: {a['detail']}" for a in required],
        "notes": plan.get("notes", ""),
    }
