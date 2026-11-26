"""Hàm thuần của pipeline/planner: map timeline khi cắt, đổi remove_region quá lớn thành cut."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import _from_cut, _remap_segments, _to_cut  # noqa: E402
from app.planner import _validate  # noqa: E402

CUTS = [(5.0, 8.0), (12.0, 13.0)]


def test_cut_timeline_roundtrip():
    for t in (0.0, 4.9, 8.0, 10.0, 13.5, 19.0):
        assert abs(_from_cut(_to_cut(t, CUTS), CUTS) - t) < 1e-9
    assert _to_cut(6.0, CUTS) == 5.0  # trong đoạn cắt => dính vào điểm nối
    assert _to_cut(19.0, CUTS) == 15.0


def test_remap_segments_shrinks_and_drops():
    segs = [{"id": 0, "start": 0.0, "end": 18.6, "text": "a"}, {"id": 1, "start": 5.5, "end": 7.5, "text": "b"}]
    out = _remap_segments(segs, CUTS)
    assert [s["id"] for s in out] == [0]
    assert out[0]["end"] == 14.6


def test_large_remove_region_becomes_cut():
    plan = {"decision": "fix", "fix_audio": True, "visual_actions": [
        {"type": "remove_region", "start": 5, "end": 8, "bbox": [0.06, 0.04, 0.94, 0.92], "what": "form"},
        {"type": "remove_region", "start": 1, "end": 2, "bbox": [0.2, 0.4, 0.8, 0.55], "what": "popup"}]}
    out = _validate(plan, [], 19.0)
    assert [a["type"] for a in out["visual_actions"]] == ["cut", "remove_region"]
    assert out["decision"] == "fix"


def test_large_blur_regions_count_against_cut_limit():
    # 3 vùng lớn (> 30% khung) tổng 9s / 19s => vượt 35% => unfixable
    big = [0.1, 0.1, 0.7, 0.7]
    plan = {"decision": "fix", "fix_audio": True, "visual_actions": [
        {"type": "remove_region", "start": s, "end": s + 3, "bbox": big, "what": "dating profile"} for s in (0, 5, 10)]}
    out = _validate(plan, [], 19.0)
    assert out["decision"] == "unfixable"
    # vùng nhỏ thì không tính
    small = [0.2, 0.4, 0.6, 0.5]
    plan["visual_actions"] = [{"type": "remove_region", "start": 0, "end": 18, "bbox": small, "what": "badge"}]
    assert _validate(plan, [], 19.0)["decision"] == "fix"


def test_schema_product_and_visual_method():
    from app.schemas import JobRequest

    req = JobRequest.model_validate({"advertiser_id": "1", "product": "Shieldy call app", "visual_method": "blur",
                                     "ads": [{"smart_ad_id": "a"}, {"smart_ad_id": "b", "product": "x", "visual_method": "auto"}]})
    assert req.product == "Shieldy call app" and req.visual_method == "blur"
    assert req.items[0].product is None and req.items[1].visual_method == "auto"
