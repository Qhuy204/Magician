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

# remove_region lớn hơn tỉ lệ này của khung hình thì đổi thành cut
MAX_INPAINT_AREA = 0.45
# remove_region lớn hơn tỉ lệ này tính vào giới hạn thời lượng như cut
LARGE_AREA = 0.3
VERDICTS = ("fixed_in_video", "outside_video", "unfixable", "not_found")


def _area(b: list[float]) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


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
- remove_region {start, end, bbox, what, method}: erase a non-text graphic (pop-up, banner, cash counter, badge,
  watermark, creator handle, TikTok logo, unauthorized brand logo, QR code, fake button, someone's personal data)
  inside bbox during [start, end]. The bbox must enclose the whole graphic including its text, shadow and border.
  method is "inpaint" (default) or "blur": use "blur" for ALL watermarks, creator handles and third-party logos
  (brand logos, TikTok logo), whether semi-transparent or solid. Watermarks/handles/logos must be blurred even
  when no rejection reason mentions them.
- cut {start, end, what}: drop a time range from the video and audio, for shots whose whole content violates policy
  (e.g. a full-screen payout screen, a suggestive or gory shot). Cuts must not split a spoken sentence and should
  follow shot boundaries.
- add_disclaimer {key, start, end}: overlay a required disclaimer line. key is one of the playbook "disclaimers"
  (results_vary, terms_apply, financial_risk, responsible_drinking, gambling_responsible, healthy_lifestyle,
  consult_professional). Default start/end = whole video. The text is filled in the ad's language automatically.
  (The AI-voice disclosure is added automatically when the voice-over is rewritten; do not add it.)

Policy:
- Follow "policy.global_rules" and every playbook in "policy.playbooks". "policy.reason_groups" maps each rejection
  reason (by index) to its policy groups.
- Never blur or pixelate to hide something: covering images with blur/pixels is itself a TikTok violation. Remove with
  remove_region (inpaint) or cut.
- For each rejection reason output one finding:
  {"reason_index", "group", "verdict", "fix"}, verdict one of:
  * "fixed_in_video": the actions you list remove this violation from the video,
  * "outside_video": the violation is (also) about targeting, landing page, certification, rights, ad caption or
    fulfillment; list what must be done in action_required,
  * "unfixable": the product/business itself is prohibited for this policy (see playbook unfixable_if_product_is),
  * "not_found": nothing in the video matches the reason.
- action_required items: {"type": one of targeting | certificate | landing_page | ad_text | rights | new_creative |
  fulfillment | account | review, "detail": str}. Market/age restricted industries (dating, alcohol, gambling,
  finance, healthcare, weight management...) need targeting/certificate actions; do not try to make the video pass in
  a market where the industry is not allowed.

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
- "declared_product", when given, is the advertiser's statement of the product actually being advertised. It is
  authoritative over what the footage suggests: creatives are sometimes third-party footage, so shots showing a
  different product or category (e.g. dating profiles and "date now" buttons, partner-monitoring screens) are false
  depictions to remove (remove_region / remove_text / cut), and fix_audio rewrites the voice-over to the declared
  product. It is still "unfixable" when the declared product itself is prohibited, or when such footage makes up
  most of the video so nothing would remain that actually shows the declared product.
- The total duration of shots that are cut or where a large part of the frame (> 30%) must be erased must stay within
  {max_cut_pct}% of the video; otherwise return "unfixable".
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
- decision "outside_only" when nothing in the video needs changing and every finding is outside_video / not_found.
- "review_feedback", when present, means this video was ALREADY edited once and a compliance check found these
  remaining problems: fix only them. Disclaimers / the AI-voice label already visible are intended: do not add them
  again or remove them.
Respond with JSON only:
{"decision": "fix" | "unfixable" | "outside_only", "reason": str, "product": str, "fix_audio": bool,
 "sync_captions": bool,
 "visual_actions": [{"type": "replace_text"|"remove_text"|"remove_region"|"cut"|"add_disclaimer", ...}],
 "findings": [{"reason_index": int, "group": str, "verdict": str, "fix": str}],
 "action_required": [{"type": str, "detail": str}], "notes": str}"""


TRIAGE = """You review rejected TikTok video ads. Decide quickly, before any detailed editing plan, whether the ad can be
made compliant by editing the creative (rewriting voice-over, replacing/removing on-screen text or graphics, cutting
short shots, adding required disclaimers).
"policy" lists the TikTok policy groups of the rejection reasons. "unfixable_if_product_is" applies only when the
PRODUCT ITSELF is one of those (e.g. it really is spyware), not when the ad merely claims or depicts such a capability
for a product that does not have it.
It cannot ("unfixable") when the violation is about the product or business itself (gambling, chance-based rewards,
apps whose core offer is paying users cash for watching/playing, prohibited products such as spyware/stalkerware or
account hacking tools, impersonation), or when violating VISUALS run through most of the video.
Industries that are allowed only in some markets / ages (dating, alcohol, finance, healthcare, weight management) are
not unfixable by themselves: the creative is cleaned up and targeting is changed outside the video.
Identify the product from the visuals (end card, app name/logo, store badges). If it is a legitimate app and the
violation is a false claim in the voice-over or text about a capability the product does not provide (e.g. claiming a
data-transfer app lets you read other people's chats), it is fixable: the voice-over can be fully rewritten to the
product's real function and the shots depicting the false capability removed, even if the claim is in every spoken line.
Judge the visuals as they will read AFTER the voice-over is rewritten: ordinary app screens (chat lists, conversations,
photos) are not violations by themselves when they can be presented as the user's own data; only shots that
specifically show the false capability (typing another person's number to unlock their data, "spy"/"hack" overlays,
captions repeating the claim) count as violating visuals.
If "declared_product" is given it is the advertiser's statement of the real product and overrides what the footage
suggests (the creative may be third-party footage of another app); footage of another product or category is then a
false depiction that can be removed, unless it makes up most of the video or the declared product itself is prohibited.
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


def _reasons_payload(reasons: list[ReviewReason]) -> list[dict]:
    return [{"index": i, **r.model_dump(exclude={"violation_frames"}, exclude_defaults=True)} for i, r in enumerate(reasons)]


def _triage_policy(policy: dict | None) -> dict | None:
    if not policy:
        return None
    return {"reason_groups": policy["reason_groups"],
            "groups": [{k: p[k] for k in ("group", "name", "kind", "unfixable_if_product_is") if k in p}
                       for p in policy["playbooks"]]}


def triage(client: OpenAI, reasons: list[ReviewReason], segments: list[dict], frames: list,
           violation_images: list[str], hint: str | None, product: str | None = None, policy: dict | None = None) -> dict:
    """Lượt lọc nhanh (chạy song song với OCR): ad hỏng ở bản chất sản phẩm thì dừng sớm, khỏi chờ OCR."""
    payload = {"ad": hint, "declared_product": product, "policy": _triage_policy(policy),
               "rejection_reasons": _reasons_payload(reasons),
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
        messages=[{"role": "system", "content": SYSTEM.replace("{max_cut_pct}", str(int(settings.max_cut_ratio * 100)))},
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
    """Bỏ action sai định dạng, kẹp toạ độ/thời gian, áp giới hạn tổng thời lượng cắt."""
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
            act = {"type": kind, "start": round(start, 2), "end": round(end, 2), "what": a.get("what", "")}
            if kind == "remove_region":
                try:
                    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in a["bbox"])
                except (KeyError, TypeError, ValueError):
                    continue
                if x1 - x0 < 0.01 or y1 - y0 < 0.01:
                    continue
                method = "blur" if a.get("method") == "blur" else "inpaint"
                if method == "inpaint" and (x1 - x0) * (y1 - y0) > MAX_INPAINT_AREA:
                    # inpaint vùng gần cả khung chỉ ra 1 mảng nhoè => cắt cả đoạn
                    # (vùng blur thì không bị vấn đề này, nhưng vẫn tính vào giới hạn cắt ở dưới)
                    act |= {"type": "cut", "what": f"{act['what']} (vùng quá lớn để xoá, chuyển sang cắt)"}
                    actions.append(act)
                    continue
                act |= {"bbox": [x0, y0, x1, y1], "method": method}
            actions.append(act)
    cut_total = sum(a["end"] - a["start"] for a in actions
                    if a["type"] == "cut" or (a["type"] == "remove_region" and _area(a["bbox"]) > LARGE_AREA))
    decision = plan.get("decision") if plan.get("decision") in ("fix", "unfixable", "outside_only") else "unfixable"
    reason = plan.get("reason", "")
    findings, required = _findings(plan, n_reasons), _actions_required(plan)
    if decision == "fix" and cut_total > settings.max_cut_ratio * duration:
        decision, reason = "unfixable", (f"cần cắt / xoá phần lớn khung hình trong {cut_total:.1f}s / {duration:.1f}s, "
                                         f"vượt giới hạn {settings.max_cut_ratio:.0%}")
    if decision == "fix" and not actions and not plan.get("fix_audio"):
        decision = "outside_only" if required else "unfixable"
        reason = reason or "không tìm thấy vi phạm trong video để sửa"
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
