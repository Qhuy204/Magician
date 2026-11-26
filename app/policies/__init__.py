"""Danh mục policy TikTok (catalog.yaml): phân loại reject reason → nhóm policy, playbook cho planner, disclaimer,
gợi ý targeting. Xem README mục "Policy"."""
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

CATALOG = Path(__file__).with_name("catalog.yaml")
# Loại việc ngoài video planner / service có thể trả về trong action_required
ACTION_TYPES = ("targeting", "certificate", "landing_page", "ad_text", "rights", "aigc_label", "new_creative",
                "fulfillment", "account", "review")


@dataclass
class Group:
    id: str
    name: str
    url: str
    kind: str
    keywords: list[str]
    rules: list[str]
    region_restricted: bool = False
    min_age: int | None = None
    disclaimers: list[str] = field(default_factory=list)
    product_unfixable: list[str] = field(default_factory=list)
    out_of_video: list[str] = field(default_factory=list)

    def brief(self) -> dict:
        """Phần đưa vào prompt."""
        d = {"group": self.id, "name": self.name, "kind": self.kind, "rules": self.rules}
        if self.product_unfixable:
            d["unfixable_if_product_is"] = self.product_unfixable
        if self.disclaimers:
            d["disclaimers"] = self.disclaimers
        if self.min_age:
            d["min_age"] = self.min_age
        return d


@dataclass
class Catalog:
    version: str
    groups: dict[str, Group]
    disclaimers: dict[str, dict[str, str]]
    market_languages: dict[str, str]
    global_rules: list[str]


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    groups = {g["id"]: Group(**g) for g in raw["groups"]}
    return Catalog(raw["version"], groups, raw["disclaimers"], raw["market_languages"], raw["global_rules"])


def _score(group: Group, category: str, detail: str) -> float:
    s = 0.0
    for kw in group.keywords:
        k = kw.lower()
        pattern = r"\b" + re.escape(k) if k[0].isalnum() else re.escape(k)
        if re.search(pattern, category):
            s += 3
        elif re.search(pattern, detail):
            s += 1
    if group.name.lower() in category:
        s += 5
    return s


def classify_reason(category: str, detail: str) -> list[tuple[str, float]]:
    """Khớp từ khoá: trả [(group_id, score)] giảm dần, tối đa 3 nhóm (đủ gần nhóm cao nhất)."""
    category, detail = (category or "").lower(), (detail or "").lower()
    scores = sorted(((g.id, _score(g, category, detail)) for g in catalog().groups.values()), key=lambda x: -x[1])
    top = scores[0][1] if scores else 0
    if top < 2:
        return []
    return [(gid, sc) for gid, sc in scores[:3] if sc >= max(2, top * 0.6)]


LLM_PROMPT = """Classify TikTok ad rejection reasons into TikTok advertising policy groups.
Groups: {groups}
For each reason return 1-3 group ids, most relevant first. Respond with JSON only:
{{"results": [{{"index": int, "groups": [str]}}]}}"""


def classify_reasons(client, reasons, model: str) -> list[list[str]]:
    """Mỗi reason → danh sách group id. Từ khoá trước; reason không khớp gì thì hỏi model (1 lần cho tất cả)."""
    out: list[list[str]] = []
    unknown = []
    for i, r in enumerate(reasons):
        hits = [gid for gid, _ in classify_reason(r.category, r.detail)]
        out.append(hits)
        if not hits:
            unknown.append(i)
    if unknown and client is not None:
        cat = catalog()
        try:
            resp = client.chat.completions.create(
                model=model, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": LLM_PROMPT.format(
                              groups=json.dumps({g.id: g.name for g in cat.groups.values()}))},
                          {"role": "user", "content": json.dumps(
                              [{"index": i, "category": reasons[i].category, "detail": reasons[i].detail}
                               for i in unknown], ensure_ascii=False)}])
            for res in json.loads(resp.choices[0].message.content or "{}").get("results") or []:
                i = int(res.get("index", -1))
                if i in unknown:
                    out[i] = [g for g in res.get("groups") or [] if g in cat.groups][:3]
        except Exception as exc:  # phân loại chỉ để chọn playbook, lỗi thì planner vẫn chạy với luật chung
            log.warning("policy classify fallback failed: %s", exc)
    return out


def policy_context(reasons, groups_per_reason: list[list[str]]) -> dict:
    """Payload playbook cho planner / rewrite / recheck: luật chung + luật của các nhóm vi phạm."""
    cat = catalog()
    used = {g for gs in groups_per_reason for g in gs}
    return {
        "catalog_version": cat.version,
        "global_rules": cat.global_rules,
        "reason_groups": [{"reason_index": i, "groups": gs} for i, gs in enumerate(groups_per_reason)],
        "playbooks": [cat.groups[g].brief() for g in sorted(used)],
    }


def disclaimer(key: str, language: str | None) -> str | None:
    texts = catalog().disclaimers.get(key) or {}
    lang = (language or "en").lower().split("-")[0]
    return texts.get(lang) or texts.get("en")


def language_for_regions(regions: list[str]) -> str | None:
    langs = [catalog().market_languages.get(r.upper()) for r in regions]
    langs = [x for x in langs if x]
    return max(set(langs), key=langs.count) if langs else None


def targeting_fix(reasons, groups_per_reason: list[list[str]]) -> dict | None:
    """Ngành bị hạn chế theo khu vực / tuổi: gợi ý loại các khu vực bị flag và giới hạn tuổi tối thiểu."""
    cat = catalog()
    regions: set[str] = set()
    placements: set[str] = set()
    min_age = None
    why = []
    for r, gs in zip(reasons, groups_per_reason):
        restricted = [cat.groups[g] for g in gs if cat.groups[g].region_restricted]
        ages = [cat.groups[g].min_age for g in gs if cat.groups[g].min_age]
        if ages:
            min_age = max(min_age or 0, *ages)
        if restricted and r.affected_regions:
            regions.update(x.upper() for x in r.affected_regions)
            why.append(f"{restricted[0].name}: {', '.join(r.affected_regions)}")
        # chỉ gợi ý bỏ placement khi reason chỉ dính một phần placement (vd chỉ Pangle)
        if r.affected_placements and len(r.affected_placements) == 1:
            placements.update(r.affected_placements)
    if not (regions or placements or min_age):
        return None
    return {"exclude_regions": sorted(regions), "exclude_placements": sorted(placements), "min_age": min_age,
            "why": why}
