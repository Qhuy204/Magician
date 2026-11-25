import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_TS = re.compile(r"(\d+(?::\d+){0,2}(?:\.\d+)?)\s*[-–—~]\s*(\d+(?::\d+){0,2}(?:\.\d+)?)")


def _seconds(ts: str) -> float:
    out = 0.0
    for part in ts.split(":"):
        out = out * 60 + float(part)
    return out


class ReviewReason(BaseModel):
    """1 lý do reject. Nhận cả dạng có cấu trúc của dashboard lẫn chuỗi thuần."""

    model_config = ConfigDict(extra="allow")

    category: str = ""
    detail: str = ""
    timestamps: list[str] = []
    affected_regions: list[str] = []
    affected_placements: list[str] = []
    violation_frames: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, v: Any):
        return {"detail": v} if isinstance(v, str) else v

    def ranges(self) -> list[tuple[float, float]]:
        out = []
        for ts in self.timestamps:
            if m := _TS.search(ts):
                out.append((_seconds(m.group(1)), _seconds(m.group(2))))
        return out

    def summary(self) -> str:
        parts = [f"[{self.category}]" if self.category else "", self.detail]
        if self.timestamps:
            parts.append(f"(at {', '.join(self.timestamps)})")
        if self.affected_regions:
            parts.append(f"(regions: {', '.join(self.affected_regions)})")
        return " ".join(p for p in parts if p)


class AdItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    smart_ad_id: str
    smart_ad_name: str | None = None
    status: str | None = None
    video_id: str | None = None
    preview_video_id: str | None = None
    video_url: str | None = None
    video_url_source: str | None = None
    # Override thủ công, nếu có sẽ không gọi TikTok API. Chuỗi thuần hoặc object {category, detail, timestamps, ...}
    reasons: list[ReviewReason] | None = None
    # Ảnh frame vi phạm (URL) nếu dashboard có
    violation_frames: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, v: Any):
        """Map format dashboard mới: ad_name, review_status, rejected_video_id, reject_reasons (cũ)."""
        if not isinstance(v, dict):
            return v
        v = dict(v)
        v.setdefault("smart_ad_name", v.get("ad_name"))
        v.setdefault("status", v.get("review_status"))
        v.setdefault("video_id", v.get("rejected_video_id"))
        if v.get("reasons") is None and v.get("reject_reasons") is not None:
            v["reasons"] = v["reject_reasons"]
        return v


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    view: str | None = None
    advertiser_id: str
    campaign_id: str | None = None
    campaign_name: str | None = None
    total: int | None = None
    items: list[AdItem]

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, v: Any):
        """Format mới: {"campaign": {"name", "campaign_id"}, "ads": [...]}; format cũ: campaign_name + items."""
        if not isinstance(v, dict):
            return v
        v = dict(v)
        camp = v.get("campaign") if isinstance(v.get("campaign"), dict) else {}
        v.setdefault("campaign_name", camp.get("name") or camp.get("campaign_name"))
        v.setdefault("campaign_id", camp.get("campaign_id") or camp.get("id"))
        if "items" not in v and "ads" in v:
            v["items"] = v.pop("ads")
        return v
    callback_url: str | None = None
    # Mặc định chỉ xử lý ad bị reject (toàn phần hoặc một phần)
    process_statuses: list[str] = Field(default_factory=lambda: ["UNAVAILABLE", "PART_AVAILABLE"])


ItemState = Literal["pending", "running", "done", "no_changes", "skipped", "failed"]


class Edit(BaseModel):
    segment_ids: list[int]
    start: float
    end: float
    original_text: str
    new_text: str
    reason: str = ""
    tempo: float = 1.0
    # natural | duration (OmniVoice sinh đúng độ dài khung) | atempo (tăng tốc bằng ffmpeg)
    fit: str = "natural"


class ItemResult(BaseModel):
    smart_ad_id: str
    smart_ad_name: str | None = None
    video_id: str | None = None
    state: ItemState = "pending"
    language: str | None = None
    reject_reasons: list[str] = []
    edits: list[Edit] = []
    output_video_url: str | None = None
    error: str | None = None
    usage: dict[str, float] = {}
    timings: dict[str, float] = {}
    # Kế hoạch sửa do planner chọn: decision, fix_audio, visual_actions, notes...
    plan: dict[str, Any] = {}
    # Các thao tác trên hình đã áp dụng (replace_text / remove_text / remove_region / cut)
    visual_edits: list[dict[str, Any]] = []


class Job(BaseModel):
    job_id: str
    client_id: str = "anonymous"
    state: Literal["queued", "running", "done", "failed"] = "queued"
    advertiser_id: str
    campaign_id: str | None = None
    campaign_name: str | None = None
    callback_url: str | None = None
    items: list[ItemResult] = []
    callback_status: str | None = None
    cache_hits: dict[str, int] = {}
