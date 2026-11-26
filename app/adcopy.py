"""Viết lại caption (ad text) theo policy: bỏ claim vi phạm, sửa chính tả / viết hoa / ký hiệu (Ad format)."""
import json

from openai import OpenAI

from .config import settings
from .schemas import ReviewReason

PROMPT = """You fix the caption (ad text) of a rejected TikTok ad so it complies with TikTok advertising policies.
Rules:
- Follow "policy.global_rules" and the playbooks in "policy.playbooks".
- Ad format: no spelling or grammar mistakes, no excessive or informal capitalization, spacing, numbers, symbols or
  punctuation among letters; commonly accepted slang is fine.
- Remove violating claims (absolute terms, guarantees, exaggerated results, privacy-infringing capabilities, cash
  rewards, etc.). Never restate them with euphemisms, misspellings or coded words.
- Describe only the product's legitimate function ("declared_product" is authoritative when given). Do not invent claims.
- The caption must be consistent with the video and the call to action. Keep the same language and dialect.
- At most max_chars characters. If the caption is already compliant, return it unchanged.
Respond with JSON only: {"text": str, "changed": bool, "issues": [str]}"""


def rewrite_ad_text(client: OpenAI, text: str, reasons: list[ReviewReason], policy: dict, product: str | None,
                    cta: str | None) -> dict:
    max_chars = max(len(text), 100)
    payload = {"caption": text, "call_to_action": cta, "declared_product": product, "max_chars": max_chars,
               "policy": policy, "rejection_reasons": [r.summary() for r in reasons]}
    r = client.chat.completions.create(
        model=settings.rewrite_model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    out = json.loads(r.choices[0].message.content or "{}")
    new = (out.get("text") or text).strip()[:max_chars]
    changed = new != text.strip()
    issues = [str(i)[:200] for i in out.get("issues") or []][:10]
    # model nêu vi phạm mà không viết lại được (vd chỉ có claim vi phạm, không có gì hợp lệ để giữ)
    compliant = changed or not issues
    return {"original": text, "new": new, "changed": changed, "compliant": compliant, "issues": issues}
