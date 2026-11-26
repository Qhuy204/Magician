"""Kiểm tra lại video sau khi sửa: transcribe audio mới + xem frame, đối chiếu với lý do reject và playbook policy.
Còn vi phạm thì pipeline sửa thêm 1 lượt (review_feedback), vẫn còn thì item = needs_review."""
import json
import logging
from pathlib import Path

from openai import OpenAI

from . import media, vision
from .config import settings
from .planner import _frame_content, _reasons_payload
from .schemas import ReviewReason

log = logging.getLogger(__name__)

PROMPT = """You are a TikTok ad reviewer re-checking an EDITED video ad that was previously rejected.
You get the original rejection reasons, the TikTok policy playbooks ("policy"), the voice-over transcript of the edited
video, the edits that were applied, and frames of the edited video (one per second, labelled with timestamps).
Decide whether the edited VIDEO still violates any rejection reason or global rule. Ignore what is outside the video
(targeting, landing page, licenses, AIGC label setting): those are handled separately.
Alpha-blended meme cover layers are the intended treatment of this editor for 18+ imagery: during the segments
listed in "applied_visual_edits" the frame intentionally shows a semi-transparent meme video blended over the
original video (both videos ghost-visible at once). Do NOT flag the double-image blended look, and do NOT flag the
covered imagery being faintly visible underneath - that is by design. DO flag 18+ imagery appearing OUTSIDE any
covered segment, covered segments missing their blend layer (clean unblended violating footage), text half-erased
or left garbled, when
a disclaimer is unreadable, when the voice-over still states the violating claim, or when captions contradict the new
voice-over.
Judge the video as a whole with its NEW voice-over. Visuals of apps, phones and product demonstrations are NEVER
flagged - not even when they depict capabilities the rewritten voice-over no longer claims (entering a number,
viewing chats, spy-like screens): this editor deliberately leaves footage untouched and fixes only the voice-over
and on-screen text. Third-party logos, watermarks, creator handles and brand UI (WhatsApp, TikTok, store badges)
visible in the footage are likewise accepted - do not flag them. In the VISUAL layer you may only flag: 18+ imagery
outside any covered segment, violating captions/text still readable, half-erased or garbled text. In the AUDIO layer
you may flag any violating claim the new voice-over still states.
Only flag (a) what still matches a rejection reason, (b) clear violations of the global rules, (c) defects the editing
introduced. Standard ad elements are legitimate: end cards with app name, logo, store badges and an animated
"search the app" bar, CTA buttons, the AI-voice label and disclaimers added by the edit.
If "declared_product" is given it is the real product. Do not flag legitimate content. Be specific about where the
problem is.
Respond with JSON only: {"passed": bool, "issues": [{"reason_index": int | null, "time": str | null,
"problem": str, "fix_hint": str}]}"""


def recheck(client: OpenAI, video: Path, reasons: list[ReviewReason], policy: dict | None, edits: list[dict],
            visual_edits: list[dict], hint: str | None, product: str | None, work: Path) -> dict:
    transcript = ""
    if media.has_audio(video):
        wav = media.to_mono_16k(video, work / "recheck_16k.wav")
        with open(wav, "rb") as f:
            r = client.audio.transcriptions.create(model=settings.transcribe_model, file=f, response_format="json")
        transcript = getattr(r, "text", "") or ""
    frames = vision.sample_frames(video, 1.0, 384)
    if len(frames) > settings.max_planner_frames:
        step = len(frames) / settings.max_planner_frames
        frames = [frames[int(i * step)] for i in range(settings.max_planner_frames)]
    payload = {"ad": hint, "declared_product": product, "policy": policy,
               "rejection_reasons": _reasons_payload(reasons), "new_voice_over": transcript,
               "applied_audio_edits": [{"original": e.get("original_text"), "new": e.get("new_text")} for e in edits],
               "applied_visual_edits": [{k: v for k, v in a.items() if k not in ("track_ids",)} for a in visual_edits]}
    resp = client.chat.completions.create(
        model=settings.vision_model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                                               *_frame_content(frames, [])]}])
    out = json.loads(resp.choices[0].message.content or "{}")
    issues = [{"reason_index": i.get("reason_index"), "time": i.get("time"), "problem": str(i.get("problem", ""))[:400],
               "fix_hint": str(i.get("fix_hint", ""))[:300]} for i in out.get("issues") or [] if isinstance(i, dict)]
    passed = bool(out.get("passed")) and not issues
    log.info("recheck %s: passed=%s issues=%d", video.name, passed, len(issues))
    return {"passed": passed, "issues": issues, "transcript": transcript[:2000]}
