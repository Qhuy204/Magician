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
Blurring is the intended removal method of this editor: every blur patch listed in "applied_visual_edits" is
intentional. Do NOT flag the existence of blur patches. DO flag when the violation is still readable or recognizable
through or outside the blur (blur too small, too short, or too weak), when text was half-erased or left garbled, when
a disclaimer is unreadable, when the voice-over still states the violating claim, or when captions contradict the new
voice-over.
Judge the video as a whole with its NEW voice-over: ordinary app screens (chat lists, conversations, photos) are not
violations by themselves when the new voice-over presents them as the user's own data; flag only what still states or
depicts the violation (e.g. typing another person's number to unlock their data, "spy"/"hack" text, the old claim).
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
