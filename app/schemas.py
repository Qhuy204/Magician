import re
from datetime import datetime
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
    # Sản phẩm thật đang quảng cáo (vd "Shieldy - call app: caller ID, spam blocking"). Ghi đè JobRequest.product.
    # Planner coi đây là sự thật về sản phẩm, kể cả khi creative là footage lấy từ nơi khác.
    product: str | None = None
    # deprecated: pipeline chỉ còn làm mờ (không inpaint); giữ field để tương thích, bỏ qua giá trị
    visual_method: Literal["auto", "blur"] | None = None
    # Caption của ad và URL trang đích (nếu dashboard có): viết lại caption theo policy + kiểm tra landing page
    ad_text: str | None = None
    landing_page_url: str | None = None
    call_to_action: str | None = None

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
        for key in ("ad_text", "ad_texts", "caption", "ad_caption", "text"):
            if v.get("ad_text") is None and v.get(key):
                val = v[key]
                v["ad_text"] = val[0] if isinstance(val, list) else val
        for key in ("landing_page_url", "landing_url", "landing_page", "click_url", "store_url", "app_url", "deeplink"):
            if v.get("landing_page_url") is None and isinstance(v.get(key), str) and v[key].startswith("http"):
                v["landing_page_url"] = v[key]
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
    product: str | None = None
    visual_method: Literal["auto", "blur"] | None = None
    # Mặc định chỉ xử lý ad bị reject (toàn phần hoặc một phần)
    process_statuses: list[str] = Field(default_factory=lambda: ["UNAVAILABLE", "PART_AVAILABLE"])


# pending = đang chờ trong hàng đợi; retrying = lỗi tạm thời, chờ tới next_retry_at để chạy lại
# partial = đã có video sửa nhưng còn việc ngoài video (action_required); needs_review = recheck vẫn thấy vi phạm
ItemState = Literal["pending", "running", "retrying", "done", "partial", "needs_review", "no_changes", "skipped",
                    "failed", "cancelled"]
TERMINAL_ITEM_STATES = {"done", "partial", "needs_review", "no_changes", "skipped", "failed", "cancelled"}


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
    # --- policy ---
    # Mỗi reject reason: nhóm policy, kết quả xử lý (fixed_in_video | outside_video | unfixable | not_found)
    policy_findings: list[dict[str, Any]] = []
    # Việc phải làm ngoài video: [{type: targeting|certificate|landing_page|ad_text|rights|aigc_label|..., detail}]
    action_required: list[dict[str, Any]] = []
    # Gợi ý targeting cho ngành hạn chế theo khu vực / tuổi: {exclude_regions, exclude_placements, min_age, why}
    targeting_fix: dict[str, Any] | None = None
    # Kiểm tra lại video kết quả: {passed, issues: [...], passes}
    compliance_check: dict[str, Any] | None = None
    # Caption viết lại: {original, new, changed, notes}
    ad_text_fix: dict[str, Any] | None = None
    # Kiểm tra landing page: {url, final_url, status, ok, issues}
    landing_page: dict[str, Any] | None = None
    # --- vận hành ---
    stage: str | None = None
    attempt: int = 0
    error_code: str | None = None
    retryable: bool | None = None
    next_retry_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # item trùng với item đang chạy ở job khác (cùng ad + video + reasons)
    duplicate_of: str | None = None


class Job(BaseModel):
    job_id: str
    client_id: str = "anonymous"
    state: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    advertiser_id: str
    campaign_id: str | None = None
    campaign_name: str | None = None
    callback_url: str | None = None
    items: list[ItemResult] = []
    callback_status: str | None = None
    cache_hits: dict[str, int] = {}
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Tính lúc trả response: {"total", "finished", "done", "no_changes", "failed", ...}
    progress: dict[str, int] = {}
    # Số item của job khác đang xếp trước item đầu tiên còn chờ của job này (None = không còn item chờ)
    queue_position: int | None = None


class RetryRequest(BaseModel):
    """Body của POST /jobs/{id}/retry. Mặc định retry mọi item failed / cancelled."""

    smart_ad_ids: list[str] | None = None
    # Link video mới cho item bị source_expired: {smart_ad_id: url}
    video_urls: dict[str, str] = {}
    # Ghi đè reject reasons: {smart_ad_id: [reason, ...]}
    reasons: dict[str, list[ReviewReason]] = {}
    # Cho phép chạy lại cả item no_changes / done (khi chỉ định smart_ad_ids)
    force: bool = False
    # Ghi đè sản phẩm thật / cách sửa hình cho các item được retry
    product: str | None = None
    visual_method: Literal["auto", "blur"] | None = None
    ad_text: str | None = None
    landing_page_url: str | None = None


class CancelRequest(BaseModel):
    """Body của POST /jobs/{id}/cancel. Trống = huỷ toàn bộ item chưa xong của job."""

    smart_ad_ids: list[str] | None = None
