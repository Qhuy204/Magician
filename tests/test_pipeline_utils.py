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

def test_large_remove_region_stays_blur():
    # vùng lớn cũng blur: video không bao giờ bị cắt
    plan = {"decision": "fix", "fix_audio": True, "visual_actions": [
        {"type": "remove_region", "start": 5, "end": 8, "bbox": [0.06, 0.04, 0.94, 0.92], "what": "form"},
        {"type": "remove_region", "start": 1, "end": 2, "bbox": [0.2, 0.4, 0.8, 0.55], "what": "popup"}]}
    out = _validate(plan, [], 19.0)
    assert [a["type"] for a in out["visual_actions"]] == ["remove_region", "remove_region"]
    assert all(a["method"] == "blur" for a in out["visual_actions"])
    assert out["decision"] == "fix"

def test_cut_action_becomes_fullframe_blur():
    # planner không còn công cụ cut; nếu model vẫn sinh ra thì đổi thành blur toàn khung
    plan = {"decision": "fix", "fix_audio": True, "visual_actions": [
        {"type": "cut", "start": 5, "end": 8, "what": "spy footage"}]}
    out = _validate(plan, [], 19.0)
    a = out["visual_actions"][0]
    assert a["type"] == "remove_region" and a["method"] == "blur" and a["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert a["start"] == 5 and a["end"] == 8

def test_schema_product_and_visual_method():
    from app.schemas import JobRequest

    req = JobRequest.model_validate({"advertiser_id": "1", "product": "Shieldy call app", "visual_method": "blur",
                                     "ads": [{"smart_ad_id": "a"}, {"smart_ad_id": "b", "product": "x", "visual_method": "auto"}]})
    assert req.product == "Shieldy call app" and req.visual_method == "blur"
    assert req.items[0].product is None and req.items[1].visual_method == "auto"

def test_snap_cut_catches_crossfade(tmp_path):
    """Đỏ 0–2s, mờ dần sang xanh 1.75–2.25s, xanh tới 5s, cắt cứng sang trắng ở 4s. Planner bảo cắt 2.0–3.8
    (đoạn xanh) => start lùi về đầu crossfade, end tới sát lần cắt cứng ở 4s."""
    import subprocess

    from app import video_edit, vision

    v = tmp_path / "x.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=320x568:d=2:r=30",
                    "-f", "lavfi", "-i", "color=blue:s=320x568:d=2.25:r=30", "-f", "lavfi", "-i", "color=white:s=320x568:d=2:r=30",
                    "-filter_complex", "[0][1]xfade=transition=fade:duration=0.5:offset=1.75[a];[a][2]concat=n=2:v=1[v]",
                    "-map", "[v]", "-pix_fmt", "yuv420p", str(v)], check=True)
    info = vision.video_info(v)
    s, e = video_edit.snap_cut(v, info, 2.0, 3.8)
    assert 1.7 <= s <= 2.0, s
    assert 3.95 <= e <= 4.1, e

def test_build_regions_all_blur(tmp_path):
    """Mọi vùng xoá đều blur (không còn inpaint); replace_text vẫn có overlay text mới."""
    import subprocess

    from app import video_edit, vision

    v = tmp_path / "w.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x568:duration=2:rate=15",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    info = vision.video_info(v)
    actions = [
        {"type": "remove_region", "start": 0.0, "end": 2.0, "bbox": [0.0, 0.0, 0.3, 0.06]},
        {"type": "remove_region", "start": 0.0, "end": 2.0, "bbox": [0.3, 0.4, 0.5, 0.5]},
    ]
    regions = video_edit.build_regions(actions, {}, info, lambda t: vision.frame_at(v, t))
    assert all(r.method == "blur" for r in regions)

def test_blur_render_obscures_watermark(tmp_path):
    """Vẽ chữ giả watermark lên video, render với vùng blur: chữ không còn đọc được (năng lượng tần số cao giảm)."""
    import subprocess

    import numpy as np

    from app import video_edit, vision

    import cv2

    src = tmp_path / "src.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 568))
    for _ in range(30):
        frame = np.full((568, 320, 3), 128, np.uint8)
        cv2.putText(frame, "@creator_handle", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(frame)
    vw.release()
    info = vision.video_info(src)
    box = (0, 0, int(0.75 * info.width), int(0.06 * info.height))
    regions = [video_edit.Region(0.0, 2.0, box, "blur")]

    def highfreq(img):
        import cv2
        x0, y0, x1, y1 = box
        roi = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.abs(np.diff(roi, axis=1)).mean())

    import soundfile as sf

    silence = tmp_path / "silence.wav"
    sf.write(silence, np.zeros((15000 * 2, 1), np.float32), 15000)
    before = highfreq(vision.frame_at(src, 1.0))
    out = tmp_path / "out.mp4"
    video_edit.render(src, silence, out, info, regions, [], False)
    after = highfreq(vision.frame_at(out, 1.0))
    assert after < 0.4 * before, (before, after)
