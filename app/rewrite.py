import json

from openai import OpenAI

from .config import settings

SYSTEM = """You are a senior compliance copywriter for TikTok video ads.
You receive the voice-over transcript of an ad (segments with ids and timings) and the reasons TikTok rejected it.
Your job:
1. Find the MINIMAL set of transcript segments whose spoken content causes the rejection
   (e.g. exaggerated/absolute claims, fear-mongering, fake system warnings, misleading guarantees,
   impersonation, prohibited wording). Leave compliant segments untouched.
2. Rewrite those segments so they genuinely comply with TikTok Advertising Policies, keeping the tone
   and call to action.
Rules:
- Follow "policy.global_rules" and the playbooks in "policy.playbooks" (TikTok policy groups of the rejection reasons).
- Remove the violating claim; never restate it with euphemisms, misspellings, or coded wording meant to slip past review.
- Never keep claims about accessing other people's accounts, messages, calls, or location without their consent,
  hacking, or capabilities the product does not plausibly have. Replace them with the product's legitimate
  function. "product" (identified from the video's end card / app UI) is the most reliable description of what the
  product is; the campaign / video name is a weaker hint. E.g. a phone-to-phone transfer app: moving / backing up
  YOUR OWN WhatsApp chats, photos and videos to a new phone; a call app: call filtering, caller ID, spam blocking.
- The rewrite must not hint at the removed capability (no "see anyone's chats", "check what they say", "no one will
  know", references to someone else's number or account).
- If the ad cannot be made compliant without changing what the product is, return no edits and explain in notes.
- Write new_text in the SAME language and dialect as the original (e.g. Gulf Arabic stays Gulf Arabic, Hindi stays Hindi in Devanagari, Brazilian Portuguese stays pt-BR).
- Each rewrite must be speakable in the same time as the original: new_text must be at most max_chars characters
  and at least min_chars characters. The presenter keeps talking on screen, so a much shorter line leaves visible
  silent lip movement: fill the time with the product's legitimate benefits and the call to action, never with new
  claims the video does not support.
- segment_ids in one edit must be consecutive. Never overlap edits.
- Plain spoken text only: no emojis, no hashtags, no stage directions.
- If the rejection is not caused by the spoken audio (e.g. visuals, landing page, targeting), return no edits.
Respond with JSON only: {"edits": [{"segment_ids": [int], "new_text": str, "reason": str}], "notes": str}"""


def plan_edits(client: OpenAI, segments: list[dict], language: str, reject_reasons: list[str], campaign: str | None,
               product: str | None = None, policy: dict | None = None) -> dict:
    payload = {
        "campaign": campaign,
        "product": product,
        "policy": policy,
        "language": language,
        "reject_reasons": reject_reasons,
        "segments": [
            {**s, "duration": round(s["end"] - s["start"], 2), "max_chars": len(s["text"]),
             "min_chars": int(len(s["text"]) * 0.85)} for s in segments
        ],
        "note": "When merging several segments in one edit, max_chars / min_chars are the sums over those segments.",
    }
    r = client.chat.completions.create(
        model=settings.rewrite_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    data = json.loads(r.choices[0].message.content or "{}")
    ids = {s["id"] for s in segments}
    used: set[int] = set()
    edits = []
    for e in data.get("edits") or []:
        seg_ids = sorted(int(i) for i in e.get("segment_ids") or [] if int(i) in ids)
        text = (e.get("new_text") or "").strip()
        if not seg_ids or not text or used & set(seg_ids) or seg_ids != list(range(seg_ids[0], seg_ids[-1] + 1)):
            continue
        used |= set(seg_ids)
        edits.append({"segment_ids": seg_ids, "new_text": text, "reason": e.get("reason", "")})
    return {"edits": edits, "notes": data.get("notes", "")}


def lengthen(client: OpenAI, text: str, language: str, min_chars: int, max_chars: int, product: str | None) -> str:
    """Câu mới ngắn hơn khung nhiều: viết dài ra bằng lợi ích hợp lệ / CTA, không thêm claim mới."""
    r = client.chat.completions.create(
        model=settings.rewrite_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "Lengthen an ad voice-over line to between min_chars and max_chars characters, same language and "
                "dialect. Only elaborate on what the line already says about the product's legitimate function and "
                "the call to action; never add new claims, guarantees, numbers, or anything about accessing other "
                "people's data. Plain spoken text. Respond with JSON {\"text\": str}.")},
            {"role": "user", "content": json.dumps({"language": language, "product": product, "min_chars": min_chars,
                                                    "max_chars": max_chars, "text": text}, ensure_ascii=False)},
        ],
    )
    out = (json.loads(r.choices[0].message.content or "{}").get("text") or text).strip()
    return out if len(text) < len(out) <= max_chars * 1.1 else text


def shorten(client: OpenAI, text: str, language: str, max_chars: int) -> str:
    r = client.chat.completions.create(
        model=settings.rewrite_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Shorten ad voice-over lines without changing language, meaning, or policy compliance. Respond with JSON {\"text\": str}."},
            {"role": "user", "content": json.dumps({"language": language, "max_chars": max_chars, "text": text}, ensure_ascii=False)},
        ],
    )
    return (json.loads(r.choices[0].message.content or "{}").get("text") or text).strip()
