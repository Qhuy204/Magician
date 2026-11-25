import json

import httpx

from .config import settings
from .schemas import ReviewReason


async def fetch_reject_info(advertiser_id: str, ad_ids: list[str]) -> dict[str, list[dict]]:
    """Gọi /smart_plus/ad/review_info/ lấy reject_info theo smart_plus_ad_id."""
    if not settings.tiktok_access_token or not ad_ids:
        return {}
    out: dict[str, list[dict]] = {}
    async with httpx.AsyncClient(base_url=settings.tiktok_api_base, timeout=30) as client:
        for i in range(0, len(ad_ids), 100):
            r = await client.get(
                "/smart_plus/ad/review_info/",
                headers={"Access-Token": settings.tiktok_access_token},
                params={
                    "advertiser_id": advertiser_id,
                    "smart_plus_ad_ids": json.dumps(ad_ids[i : i + 100]),
                    "extra_info_setting": json.dumps({"include_reject_info": True, "include_violation_frame": True}),
                    "lang": "en",
                },
            )
            r.raise_for_status()
            body = r.json()
            if body.get("code") != 0:
                raise RuntimeError(f"TikTok API error {body.get('code')}: {body.get('message')}")
            for info in body["data"].get("smart_plus_ad_review_infos") or []:
                out[str(info["smart_plus_ad_id"])] = info.get("reject_info") or []
    return out


def reasons_for_video(reject_info: list[dict], video_ids: set[str]) -> list[ReviewReason]:
    """Ưu tiên reject_info gắn đúng video, nếu không có thì lấy toàn bộ lý do cấp ad."""
    def video_of(entry: dict) -> str | None:
        return ((entry.get("content_info") or {}).get("video_content") or {}).get("video_id")

    matched = [e for e in reject_info if video_of(e) in video_ids] or [e for e in reject_info if not video_of(e)] or reject_info
    out = []
    for e in matched:
        detail = "; ".join(e.get("reasons") or [])
        if e.get("suggestion"):
            detail += f" | Suggestion: {e['suggestion']}"
        frames = [u for f in e.get("violation_frames") or [] for u in f.get("web_url_list") or []]
        if detail or frames:
            out.append(ReviewReason(detail=detail, affected_regions=e.get("forbidden_locations") or [],
                                    affected_placements=e.get("forbidden_placements") or [], violation_frames=frames))
    return out
