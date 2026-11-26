# Magician

API service sửa voice-over của TikTok ad bị reject vì policy. Nhận JSON danh sách ad (dạng `historical_reviews`),
tải video, tìm câu thoại vi phạm, viết lại cho đúng policy, đọc lại bằng giọng clone của chính người nói trong video,
ghép vào video gốc và gửi kết quả về qua webhook. Có hàng đợi bền (SQLite), tự retry theo loại lỗi, huỷ / kill job,
chạy tiếp sau restart, chống gửi trùng, webhook có hàng đợi gửi lại, health / metrics.

- Production: `https://magician.qhuy204.id.vn`
- Ngôn ngữ giọng đọc: en, hi, pt, ar (UAE/Gulf), es, fr, de, it, pl, tr, ru, nl, cs, zh-cn, hu, ko, ja

## Mục lục
1. [Pipeline](#1-pipeline)
2. [Chạy service](#2-chạy-service)
3. [Xác thực](#3-xác-thực)
4. [API](#4-api)
5. [Webhook](#5-webhook)
6. [Bảo mật & giới hạn](#6-bảo-mật--giới-hạn)
7. [Cấu hình](#7-cấu-hình-env)
8. [Phần cứng & hiệu năng](#8-phần-cứng--hiệu-năng)
9. [Vận hành](#9-vận-hành)
10. [Hạn chế](#10-hạn-chế)

---

## 1. Pipeline

Với mỗi ad có `status` thuộc `process_statuses` (mặc định `UNAVAILABLE`, `PART_AVAILABLE`), service tự quyết định
dùng công cụ nào dựa trên lý do reject và danh mục policy TikTok (`app/policies/catalog.yaml`):

| # | Bước | Công cụ |
|---|---|---|
| 1 | Tải `video_url` (chỉ domain trong allowlist, tối đa 300MB) | httpx |
| 2 | **Phân loại policy**: mỗi lý do reject → nhóm policy TikTok (20 nhóm + Pangle), lấy playbook của nhóm đó | Từ khoá trong catalog, không chắc thì hỏi `VISION_MODEL` |
| 3 | Tách giọng nói khỏi nhạc nền, cắt câu theo khoảng lặng, transcribe từng câu | Demucs `htdemucs` + OpenAI `gpt-transcribe` |
| 4 | **Triage**: sản phẩm thuộc loại bị cấm thì dừng ngay (chạy song song với OCR) | `VISION_MODEL` + frame 1 giây/lần |
| 5 | OCR text trên màn hình, gom thành "text track" (nội dung, vị trí, thời gian) | EasyOCR (GPU) |
| 6 | **Planner** chọn công cụ cho từng vi phạm theo playbook, kết luận cho từng lý do reject, liệt kê việc ngoài video | `VISION_MODEL` |
| 7 | Sửa voice-over nếu cần: viết lại câu vi phạm, clone giọng, ghép lại | `gpt-6-luna` + OmniVoice / XTTS |
| 8 | Sửa hình: xoá text/pop-up/logo/QR, vẽ text mới, cắt đoạn (mốc cắt tự dời về ranh giới cảnh), chèn disclaimer, nhãn "AI-generated voice" | LaMa inpaint, ffmpeg drawtext |
| 9 | Encode lại video (hoặc chỉ thay audio nếu hình không đổi) | ffmpeg / libx264 |
| 10 | **Recheck**: transcribe audio mới + xem frame video kết quả, đối chiếu lý do reject. Còn vi phạm thì sửa thêm 1 lượt, vẫn còn thì `needs_review` | `gpt-transcribe` + `VISION_MODEL` |
| 11 | Viết lại caption (`ad_text`), kiểm tra landing page (`landing_page_url`) nếu request có | `REWRITE_MODEL`, httpx |

Công cụ planner có thể chọn:

| Công cụ | Khi nào | Cách làm |
|---|---|---|
| `fix_audio` | Lời thoại vi phạm | Viết lại câu vi phạm cùng ngôn ngữ, đọc bằng giọng clone, vừa khung thời gian cũ |
| `sync_captions` | Có phụ đề in sẵn lặp lại lời thoại | Phụ đề của câu bị sửa được thay bằng câu mới, cùng vị trí / màu / cỡ chữ |
| `replace_text` | Text vi phạm có cách nói hợp lệ, text sai chính tả / viết hoa / ký hiệu loạn | Xoá text cũ, vẽ text mới cùng vị trí, màu chữ + viền lấy từ text cũ |
| `remove_text` | Text vi phạm, không cần thay | Xoá text |
| `remove_region` | Pop-up, banner, số tiền, badge, logo TikTok / thương hiệu khác, QR code, nút giả, dữ liệu cá nhân | Inpaint; nền đứng yên thì chép lại nền từ frame sạch. Vùng > 45% khung hình tự đổi thành `cut` |
| `cut` | Cả shot vi phạm (màn hình rút tiền, cảnh gợi dục, máu me...) | Bỏ đoạn đó khỏi cả hình lẫn tiếng; tổng thời lượng cắt + xoá vùng lớn tối đa `MAX_CUT_RATIO` (35%) |
| `add_disclaimer` | Ngành bắt buộc disclaimer (tài chính, rượu, cờ bạc, sức khoẻ, giảm cân) hoặc promotion cần "Terms apply" | Dòng chữ trên nền mờ ở vùng an toàn, đúng ngôn ngữ của ad (en / es / pt / ar / fr / de / it) |
| (tự động) nhãn AIGC | Voice-over đã được clone | "AI-generated voice" / "Voz generada con IA"... ở trên cùng, kèm `action_required: aigc_label` |
| `unfixable` | Sản phẩm thuộc loại bị cấm (spyware, cờ bạc, app trả tiền khi xem/chơi, dịch vụ tình dục...), hoặc nội dung vi phạm chiếm gần hết video | Không sửa, item ra `no_changes` kèm lý do |

### Policy

- **Catalog** `app/policies/catalog.yaml` có 20 nhóm policy chính thức + Pangle. Mỗi nhóm khai báo:
  - `kind`: `creative` (sửa trong video), `conditional` (ngành được phép có điều kiện về tuổi, khu vực, giấy phép), `prohibited`, `out_of_scope`.
  - Từ khoá để nhận diện, luật (playbook), sản phẩm khiến ad không sửa được, disclaimer bắt buộc, tuổi tối thiểu, có hạn chế theo khu vực hay không.
  - Có thêm disclaimer theo ngôn ngữ và luật chung áp cho mọi ad.
- **Luật chung** (bất kể nhóm nào):
  - Không né review bằng nói giảm, viết sai chính tả, che một phần.
  - **Không blur / pixel hoá để che**: policy Ad format cấm "columns or pixels to partially cover images".
  - Không xoá watermark của footage bên thứ ba: cần có quyền sử dụng (`action_required: rights`).
  - Bỏ nút giả, cảnh báo hệ thống giả, QR code dẫn sang trang ngoài (EU / UK / Trung Đông).
  - Video có giọng clone phải công bố AIGC.
- **Ngành bị hạn chế theo khu vực / tuổi** (dating, rượu, tài chính, y tế, giảm cân...): làm sạch creative, gợi ý
  `targeting_fix` = loại các `affected_regions` + tuổi tối thiểu. Không sửa video để nó lọt qua ở khu vực mà ngành bị cấm.
- **Sản phẩm hợp lệ bị ad mô tả sai** (vd app chuyển dữ liệu nhưng ad nói "đọc trộm chat"):
  - Sửa được: viết lại voice-over theo chức năng thật, chỉ cắt cảnh minh hoạ chức năng giả.
  - `product` trong request (nếu có) được coi là sự thật về sản phẩm.

Service xoá hẳn claim vi phạm, không diễn đạt lại bằng từ né, không che một nửa, không dời sang chỗ khác để lọt review.

Trong cùng 1 job, video trùng audio hoặc trùng lời thoại chỉ được xử lý 1 lần (cache), các video sau dùng lại kết quả.

---

## 2. Chạy service

### Docker (Linux, NVIDIA / AMD / CPU)
```bash
cp .env.example .env        # điền OPENAI_API_KEY, API_KEYS, CALLBACK_SECRET, FILE_SIGNING_KEY, PUBLIC_BASE_URL
docker compose --profile nvidia up -d --build   # cần nvidia-container-toolkit
docker compose --profile amd up -d --build      # cần driver ROCm trên host
docker compose --profile cpu up -d --build
```
- Model (~2GB) được tải lần đầu vào volume `models`. Muốn đóng gói sẵn vào image: `--build-arg PRELOAD_MODELS=1`.
- AMD consumer GPU chưa có trong danh sách ROCm hỗ trợ có thể cần `HSA_OVERRIDE_GFX_VERSION` (xem `docker-compose.yml`).

### Mac Apple Silicon
Docker trên Mac không dùng được GPU, chạy native:
```bash
./scripts/run_mac.sh
```

### Windows / chạy tay
```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Tuỳ chọn engine XTTS: tạo env riêng (torch 2.5.1 cu121 + requirements-xtts.txt), đặt XTTS_PYTHON trỏ vào python của env đó
uvicorn app.main:app --host 127.0.0.1 --port 8000 --proxy-headers
```
Cần `ffmpeg` và `ffprobe` trong PATH. Worker TTS khởi động lúc start (OmniVoice ~5–10s, XTTS ~30s mỗi worker); trong lúc đó job vẫn nhận bình thường.
Chạy thường trực (tự restart, tự chạy khi đăng nhập): `scripts\service.ps1 install`, xem [Vận hành](#9-vận-hành).

---

## 3. Xác thực

Mỗi client có `client_id` và `secret`, khai báo trong `.env`:
```
API_KEYS=dashboard:<secret>,partner2:<secret2>
```
Tạo secret: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

### Ký request (bắt buộc khi `REQUIRE_SIGNED_REQUESTS=true`, mặc định)

Gửi 4 header với **mọi** request tới `/jobs*` và `/status`:

| Header | Giá trị |
|---|---|
| `X-Client-Id` | `client_id` |
| `X-Timestamp` | Unix time (giây). Lệch quá 300s so với server bị từ chối |
| `X-Nonce` | Chuỗi ngẫu nhiên 8–128 ký tự, không dùng lại |
| `X-Signature` | hex `HMAC_SHA256(secret, string_to_sign)` |

```
string_to_sign = X-Timestamp + "\n" + X-Nonce + "\n" + METHOD + "\n" + PATH_AND_QUERY + "\n" + hex(SHA256(body))
```
- `METHOD` viết hoa (`POST`, `GET`).
- `PATH_AND_QUERY` đúng như gửi lên, gồm cả query đã encode, ví dụ `/jobs?callback_url=https%3A%2F%2Fexample.com%2Fhook`.
- `body` là đúng các byte gửi đi; GET không có body thì hash chuỗi rỗng
  (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

Client mẫu có sẵn, không cần tự cài đặt phần ký:
- Python: [`clients/magician_client.py`](clients/magician_client.py)
- Node ≥ 18: [`clients/magician-client.mjs`](clients/magician-client.mjs)

```python
from magician_client import MagicianClient
c = MagicianClient("https://magician.qhuy204.id.vn", "dashboard", SECRET)
job = c.submit(payload, callback_url="https://your-host/magician-webhook")
print(c.status(job["job_id"]))
```
```js
import { MagicianClient } from "./magician-client.mjs";
const c = new MagicianClient("https://magician.qhuy204.id.vn", "dashboard", SECRET);
const job = await c.submit(payload, "https://your-host/magician-webhook");
```

Bash (curl + openssl):
```bash
BODY=$(cat payload.json); TS=$(date +%s); NONCE=$(openssl rand -hex 16); PATHQ="/jobs"
HASH=$(printf '%s' "$BODY" | openssl dgst -sha256 -hex | awk '{print $NF}')
SIG=$(printf '%s\n%s\n%s\n%s\n%s' "$TS" "$NONCE" POST "$PATHQ" "$HASH" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print $NF}')
curl -X POST "https://magician.qhuy204.id.vn$PATHQ" -H "Content-Type: application/json" \
  -H "X-Client-Id: dashboard" -H "X-Timestamp: $TS" -H "X-Nonce: $NONCE" -H "X-Signature: $SIG" --data-binary "$BODY"
```

### API key thuần (chỉ khi `REQUIRE_SIGNED_REQUESTS=false`)
Header `X-API-Key: <secret>`. Đơn giản hơn nhưng không chống được replay hay sửa body; chỉ nên dùng trong mạng nội bộ.

---

## 4. API

| Method | Path | Auth | Mô tả |
|---|---|---|---|
| POST | `/jobs` | ✓ | Tạo job, trả về ngay `202` (gửi trùng → trả job cũ) |
| GET | `/jobs` | ✓ | Danh sách job của client (`?state=&limit=&before=`) |
| GET | `/jobs/{job_id}` | ✓ | Trạng thái, tiến độ, kết quả từng item |
| GET | `/jobs/{job_id}/events` | ✓ | Timeline: từng lần chạy, retry, lỗi, kill (`?after=<id>` để lấy tiếp) |
| POST | `/jobs/{job_id}/retry` | ✓ | Chạy lại item lỗi / bị huỷ, có thể kèm link video mới |
| POST | `/jobs/{job_id}/cancel` | ✓ | Huỷ / kill cả job hoặc một số item |
| GET | `/jobs/{job_id}/webhooks` | ✓ | Lịch sử gửi webhook của job |
| POST | `/jobs/{job_id}/webhooks/resend` | ✓ | Gửi lại webhook `job.done` |
| GET | `/files/{job_id}/{smart_ad_id}.mp4?exp=&sig=` | link ký | Tải video kết quả |
| GET | `/status` | ✓ | Readiness chi tiết, hàng đợi, lỗi 1h gần nhất, webhook, phần cứng, worker TTS |
| GET | `/metrics` | ✓ hoặc localhost | Prometheus text format |
| GET | `/ready` | – | `200 {"ready":true}` / `503 {"ready":false}` (DB, ffmpeg, scheduler, worker TTS, đĩa) |
| GET | `/health` | – | `{"ok": true}` (process còn sống) |

`/docs` và `/openapi.json` tắt mặc định (`ENABLE_DOCS=true` để bật).

### POST /jobs

Query: `callback_url` (tùy chọn, ghi đè field trong body).
Header tùy chọn: `Idempotency-Key: <chuỗi ≤128 ký tự>`. Gửi lại cùng key (vd retry khi mạng lỗi) thì nhận lại đúng job cũ, không tạo job mới.

Body: JSON review y như dashboard xuất ra. Nhận cả 2 format:
- cũ: `{"advertiser_id", "campaign_name", "items": [{"smart_ad_id", "smart_ad_name", "status", "video_id", "video_url", "reject_reasons": [str]}]}`
- mới: `{"advertiser_id", "campaign": {"name", "campaign_id"}, "ads": [{"ad_name", "smart_ad_id", "review_status", "rejected_video_id", "video_url", "reasons": [...]}]}`

Field thừa được giữ nguyên, không gây lỗi. Ví dụ format mới:
```json
{
  "advertiser_id": "7531961948355788808",
  "campaign": {"name": "zubuu_3.11_drama_win_ar_1", "campaign_id": "1881737025976386"},
  "ads": [{
    "ad_name": "zubuu_3.11_drama_win_ar_1_1",
    "smart_ad_id": "1881737028989330",
    "review_status": "UNAVAILABLE",
    "rejected_video_id": "v1c033g50000dbre1d7og65ioocblltg",
    "video_url": "https://de-s3.storage.bunnycdn.com/...mp4?X-Amz-...",
    "reasons": [{
      "category": "Money-making opportunities",
      "detail": "A reward pop-up shows 11.35 SAR and Arabic text promoting downloading TopDrama to earn money...",
      "timestamps": ["0:22–0:24", "0:31–0:33"],
      "affected_regions": ["BO", "BR"]
    }],
    "violation_frames": ["https://...jpg"]
  }]
}
```

| Field | Bắt buộc | Mô tả |
|---|---|---|
| `advertiser_id` | ✓ | Dùng khi gọi TikTok API |
| `campaign_name` / `campaign.name` | | Gợi ý sản phẩm cho model (vd `shieldy_17.11_hook_ar`) |
| `items[]` / `ads[]` | ✓ | Tối đa 50 |
| `smart_ad_id` | ✓ | Dùng làm tên file kết quả |
| `status` / `review_status` | | Chỉ xử lý nếu thuộc `process_statuses` |
| `video_url` | | Bắt buộc với item được xử lý. Chỉ https, domain trong `VIDEO_URL_ALLOWLIST` |
| `video_id` / `rejected_video_id`, `preview_video_id` | | Khớp reject reason của TikTok với đúng video; cũng dùng để nhận ra ad gửi trùng |
| `reasons` / `reject_reasons` | | Mỗi phần tử là chuỗi hoặc object `{category, detail, timestamps, affected_regions, affected_placements, violation_frames}`. `timestamps` dạng `"0:22–0:24"` giúp planner tìm đúng chỗ. Có thì dùng luôn, không gọi TikTok API |
| `violation_frames` | | URL ảnh frame vi phạm (nếu có); planner nhìn ảnh để xác định phần bị flag |
| `product` | | Sản phẩm thật đang quảng cáo, vd `"Shieldy - call app: caller ID, chặn cuộc gọi rác"`. Đặt ở cấp job hoặc từng ad (ad ghi đè job). Planner coi đây là sự thật: creative lấy từ nơi khác (footage app hẹn hò, theo dõi người yêu...) thì các cảnh đó bị blur / cắt, voice-over viết lại theo sản phẩm này. Vẫn `unfixable` nếu footage sai chiếm phần lớn video (cắt hoặc xoá vùng > 30% khung quá `MAX_CUT_RATIO` thời lượng) hoặc chính sản phẩm bị cấm |
| `ad_text` (alias `ad_texts[0]`, `caption`, `ad_caption`) | | Caption của ad: được viết lại theo policy, trả về `ad_text_fix` |
| `landing_page_url` (alias `landing_url`, `landing_page`, `click_url`, `store_url`, `app_url`, `deeplink`) | | Trang đích: được kiểm tra, trả về `landing_page` |
| `call_to_action` | | CTA, dùng khi viết lại caption |
| `visual_method` | | `auto` (mặc định `VISUAL_METHOD`): inpaint / thay chữ / blur tuỳ chỗ. `blur`: mọi chỗ sửa hình đều làm mờ, kể cả text (không vẽ chữ mới). **Cảnh báo:** policy Ad format cấm che hình bằng blur/pixel, nên item có blur sẽ kèm `action_required: review` và có thể bị reject lại. Cấp job hoặc từng ad |
| `callback_url` | | Webhook nhận kết quả, chỉ https |
| `process_statuses` | | Mặc định `["UNAVAILABLE","PART_AVAILABLE"]` |

Không có `reasons` và không cấu hình `TIKTOK_ACCESS_TOKEN` thì item ra `no_changes`.

**Chống gửi trùng** (`DEDUP_ACTIVE=true`): một ad được coi là trùng nếu cùng client, cùng `smart_ad_id`, cùng video
(`video_id`, hoặc path của `video_url` khi không có `video_id`) và cùng reasons với một item **đang chờ / đang chạy**.
- Mọi ad trong request đều đang chạy ở 1 job khác → không tạo job mới, trả job đó kèm `"deduplicated": true`.
- Chỉ một phần trùng → vẫn tạo job mới; ad trùng có `state: "skipped"` và `duplicate_of: "<job_id đang chạy>"`.
- Ad đã xong (done / failed...) gửi lại thì chạy lại bình thường.

Response `202`:
```json
{"job_id": "dc8ef7f0ae22499692718be2adca6c0e", "status_url": "https://magician.qhuy204.id.vn/jobs/dc8e...",
 "items": 14, "deduplicated": false, "state": "queued"}
```

### GET /jobs/{job_id}
```json
{
  "job_id": "02bb2eb6d90741f7ae4ac0bbb0821a51",
  "client_id": "dashboard",
  "state": "running",
  "advertiser_id": "7687959493501976597",
  "campaign_name": "shieldy_17.11_hook_ar R1.2",
  "callback_url": "https://your-host/magician-webhook",
  "callback_status": null,
  "created_at": "2026-11-25T03:02:44.900Z", "started_at": "2026-11-25T03:02:45.010Z", "finished_at": null,
  "progress": {"total": 3, "finished": 1, "done": 1, "running": 1, "retrying": 1},
  "queue_position": 0,
  "cache_hits": {},
  "items": [
    {
      "smart_ad_id": "1882107813581906",
      "smart_ad_name": "Ad group 2_10",
      "state": "done",
      "stage": null, "attempt": 1, "error_code": null, "retryable": null, "next_retry_at": null,
      "started_at": "2026-11-25T03:02:45.010Z", "finished_at": "2026-11-25T03:03:11.420Z",
      "language": "ar",
      "reject_reasons": ["Exaggerated or misleading claims about product functionality"],
      "edits": [
        {
          "segment_ids": [0, 1, 2, 3], "start": 0.0, "end": 12.22,
          "original_text": "كيف تراقب واتساب غيرك على هاتفك؟ ...",
          "new_text": "كيف تتعرف على المكالمات المجهولة؟ ...",
          "reason": "...", "tempo": 1.0
        }
      ],
      "output_video_url": "https://magician.qhuy204.id.vn/files/02bb.../1882107813581906.mp4?exp=1795600000&sig=...",
      "error": null,
      "usage": {"transcribe_calls": 5, "transcribe_seconds": 15, "chat_calls": 3, "chat_prompt_tokens": 7350, "chat_completion_tokens": 924},
      "timings": {"download": 0.3, "separate": 1.2, "transcribe": 1.1, "triage": 3.3, "ocr": 7.2, "plan": 7.4, "rewrite": 2.6, "tts": 6.0, "render": 2.4},
      "plan": {
        "decision": "fix", "reason": "...", "fix_audio": true, "sync_captions": true,
        "visual_actions": [{"type": "remove_region", "start": 3.0, "end": 5.0, "bbox": [0.17, 0.44, 0.83, 0.53], "what": "Red 500 SAR prize pop-up"}],
        "out_of_scope": [], "notes": "..."
      },
      "visual_edits": [
        {"type": "remove_region", "start": 3.0, "end": 5.0, "bbox": [0.17, 0.44, 0.83, 0.53], "what": "Red 500 SAR prize pop-up"},
        {"type": "replace_text", "track_ids": [27, 28], "new_text": "يساعدك التطبيق في حظر المكالمات المزعجة.", "texts": ["تطبيقنا يحظر 100% من المكالمات المزعجة", "مضمون!"]}
      ]
    },
    {
      "smart_ad_id": "1882107813581907", "state": "running", "stage": "ocr", "attempt": 1
    },
    {
      "smart_ad_id": "1882107813581908", "state": "retrying", "attempt": 1,
      "error_code": "network", "retryable": true, "next_retry_at": "2026-11-25T03:03:20.000Z",
      "error": "không kết nối được de-s3.storage.bunnycdn.com: ConnectTimeout: [WinError 10060] ..."
    }
  ]
}
```

Trạng thái job: `queued` → `running` → `done` | `cancelled` (`failed` chỉ còn cho job cũ / không resume được).
`done` nghĩa là mọi item đã kết thúc; item có thể `failed`, xem từng item. Gọi `/retry` thì job quay lại `running`.

- `progress`: số item theo trạng thái; `finished` = số item đã kết thúc.
- `queue_position`: số item của các job khác đang chờ trước job này (`null` nếu job không còn item chờ).

Trạng thái item:

| `state` | Ý nghĩa |
|---|---|
| `pending` | Đang chờ trong hàng đợi |
| `running` | Đang xử lý; `stage` là bước hiện tại (`download`, `separate`, `transcribe`, `triage`, `ocr`, `plan`, `rewrite`, `tts`, `render`, `mux`) |
| `retrying` | Lỗi tạm thời, tự chạy lại lúc `next_retry_at` |
| `done` | Có video mới ở `output_video_url` |
| `no_changes` | Không sửa; lý do nằm trong `error` và `plan`. `error_code = "unfixable"` khi vi phạm ở bản chất sản phẩm, `"no_changes"` khi lý do khác (thiếu reason, cần cắt quá nhiều...) |
| `skipped` | `status` không thuộc `process_statuses`, hoặc trùng ad đang chạy (`duplicate_of`) |
| `failed` | Lỗi và đã hết lượt tự retry, xem `error_code` / `error`. `retryable: true` = gọi `/retry` có thể thành công |
| `cancelled` | Bị huỷ qua `/cancel` |

`attempt` = số lần item đã được chạy (tính cả lần tự retry).

### Kết quả theo policy (trong mỗi item)

```json
{
  "state": "partial",
  "policy_findings": [
    {"reason_index": 0, "group": "deceptive", "verdict": "fixed_in_video",
     "fix": "Rewrite the voice-over to MobileTrans's transfer function, cut the phone-number entry screen"}
  ],
  "action_required": [
    {"type": "aigc_label", "detail": "Voice-over được clone bằng AI: bật nhãn AIGC khi đăng ad"},
    {"type": "ad_text", "detail": "Thay caption bằng: Transfiere datos entre teléfonos con Wondershare MobileTrans."}
  ],
  "targeting_fix": {"exclude_regions": ["SA", "QA"], "exclude_placements": [], "min_age": 18, "why": ["Adult content: SA, QA"]},
  "compliance_check": {"passed": true, "issues": [], "passes": 1},
  "ad_text_fix": {"original": "Mira los chats de CUALQUIER persona!!!", "new": "Transfiere datos entre teléfonos con MobileTrans.",
                  "changed": true, "compliant": true, "issues": ["..."]},
  "landing_page": {"url": "...", "final_url": "...", "status": 200, "store_page": true, "title": "...",
                   "ok": true, "issues": [], "warnings": [], "consistent": true}
}
```

| Field | Ý nghĩa |
|---|---|
| `policy_findings[]` | Mỗi lý do reject: `group` (id nhóm trong catalog), `verdict` = `fixed_in_video` / `outside_video` / `unfixable` / `not_found`, `fix` mô tả cách xử lý |
| `action_required[]` | Việc phải làm ngoài video. `type`: `targeting`, `certificate` (giấy phép / chứng nhận ngành), `landing_page`, `ad_text`, `rights` (quyền dùng footage / nhạc / logo), `aigc_label`, `new_creative`, `fulfillment`, `account`, `review` |
| `targeting_fix` | Ngành hạn chế theo khu vực / tuổi: các khu vực nên loại khỏi targeting, placement nên bỏ (khi reject chỉ dính 1 placement, vd Pangle), tuổi tối thiểu |
| `compliance_check` | Kết quả recheck video đã sửa; `passes` = số lượt sửa (1 hoặc 2) |
| `ad_text_fix` | Caption đã viết lại (khi request có `ad_text`); `compliant: false` = caption vi phạm mà không có gì hợp lệ để giữ |
| `landing_page` | Kiểm tra trang đích (khi request có `landing_page_url`): HTTP lỗi, tự tải file, không thân thiện với mobile, không có privacy policy, trang trống / đang xây, nội dung không khớp sản phẩm trong ad. `ok: null` = không kết nối được nên chưa kết luận |

State liên quan:
- `done`: video đã sửa, recheck đạt, chỉ còn (nếu có) bật nhãn AIGC.
- `partial`: video đã sửa và recheck đạt, nhưng còn việc ngoài video trong `action_required` (targeting, caption, landing page, chứng nhận...).
- `needs_review`: đã sửa 2 lượt mà recheck vẫn thấy vấn đề. Vẫn có `output_video_url`, vấn đề nằm trong `compliance_check.issues` và `error`.
- `no_changes` + `error_code`:
  - `unfixable`: sản phẩm bị cấm.
  - `outside_video`: video không cần sửa, chỉ phải làm việc ngoài video.
  - `no_changes`: các trường hợp khác.

### `error_code` và cách retry

Item lỗi được tự chạy lại theo loại lỗi. Lần chạy lại **dùng lại kết quả các bước đã xong** (video đã tải, audio đã tách,
transcript, kết quả triage/plan/rewrite, câu TTS đã sinh), nên không tải lại video và không gọi lại OpenAI cho các bước đó.

| `error_code` | Ví dụ | Tự retry | Gọi `/retry` |
|---|---|---|---|
| `network` | Timeout kết nối (WinError 10060), reset, CDN trả 429/5xx | 4 lần, chờ ~5s → 15s → 45s | ✓ |
| `upstream_rate_limit` | OpenAI 429 | 6 lần, theo `Retry-After` (10s → 240s) | ✓ |
| `upstream_error` | OpenAI 5xx / mất kết nối | 4 lần (10s → 90s) | ✓ |
| `gpu_oom` | CUDA out of memory, paging file | 3 lần; concurrency tạm giảm một nửa trong `OOM_COOLDOWN_MINUTES` | ✓ |
| `worker_crash` | Worker TTS chết / treo (bị kill sau `TTS_CALL_TIMEOUT`) | 3 lần, worker được spawn lại | ✓ |
| `stage_timeout` | 1 bước chạy quá `STAGE_TIMEOUTS` | 2 lần | ✓ |
| `source_expired` | Link video trả 401/403/404/410 (link ký hết hạn / sai chữ ký) | ✗ | ✓ kèm `video_urls` mới |
| `upstream_auth` / `upstream_quota` | OpenAI key sai / hết quota | ✗ | sau khi sửa key |
| `invalid_input` | File không phải video, quá `MAX_VIDEO_MB`, URL redirect vào IP nội bộ | ✗ | ✗ |
| `internal` | Bug | ✗ | tuỳ |
| `cancelled` | Bị huỷ | ✗ | ✓ |
| `interrupted` | Job cũ bị gián đoạn trước khi có SQLite | ✗ | ✗ (gửi lại JSON) |

### POST /jobs/{job_id}/retry
Body (tùy chọn, bỏ trống = retry mọi item `failed` / `cancelled`):
```json
{
  "smart_ad_ids": ["1881737028989330"],
  "video_urls": {"1881737028989330": "https://de-s3.storage.bunnycdn.com/...mp4?X-Amz-...(link mới)"},
  "reasons": {"1881737028989330": [{"category": "...", "detail": "..."}]},
  "product": "Shieldy - call app: caller ID, spam call blocking",
  "visual_method": "auto",
  "ad_text": "caption mới cần kiểm tra",
  "landing_page_url": "https://play.google.com/store/apps/details?id=...",
  "force": false
}
```
- `video_urls`: link mới cho item `source_expired`. Đổi link thì toàn bộ file tạm của item bị xoá, chạy lại từ đầu.
- `reasons`: ghi đè reject reasons (kết quả triage/plan/rewrite cũ bị bỏ).
- `product` / `visual_method` / `ad_text` / `landing_page_url`: ghi đè cho các item được retry (đổi `product` thì bỏ kết quả triage/plan/rewrite cũ).
- `force: true` + `smart_ad_ids`: chạy lại cả item `done` / `no_changes` / `skipped` (bỏ kết quả triage/plan/rewrite, giữ video + transcript).
- `200 {"retried": [...], "state": "running"}`; `409` nếu không có item nào retry được.

### POST /jobs/{job_id}/cancel
Body tùy chọn `{"smart_ad_ids": [...]}`; bỏ trống = huỷ mọi item chưa xong và đánh dấu job `cancelled`.
- Item đang chờ / chờ retry → `cancelled` ngay.
- Item đang chạy → `state: "cancelled"`, `stage: "killing"` ngay lập tức; ffmpeg / tải video bị kill ngay, bước đang
  gọi OpenAI hoặc Demucs dừng khi bước đó trả về (thường < vài giây, test thật: ~1.6s từ lúc gọi tới lúc giải phóng slot).
- Trả `{"cancelled": ["smart_ad_id", ...], "state": "..."}`. Item bị huỷ có thể chạy lại bằng `/retry`.

### GET /jobs/{job_id}/events
```json
{"events": [
  {"id": 41, "ts": 1795575765.1, "smart_ad_id": null, "kind": "submitted", "message": "1 item cần xử lý / 1", "data": null},
  {"id": 42, "ts": 1795575765.2, "smart_ad_id": "1881737028989330", "kind": "attempt", "message": "lần chạy thứ 1"},
  {"id": 43, "ts": 1795575786.0, "smart_ad_id": "1881737028989330", "kind": "retry_scheduled", "message": "[network] ...", "data": {"delay": 5.4, "attempt": 1}},
  {"id": 44, "ts": 1795575810.3, "smart_ad_id": "1881737028989330", "kind": "done", "message": "", "data": {"seconds": 24.1}}
]}
```
`kind`: `submitted`, `attempt`, `retry_scheduled`, `resumed` (chạy tiếp sau restart), `kill`, `retry_requested`,
`done` / `no_changes` / `failed` / `cancelled`, `job_done` / `job_cancelled`, `tiktok_error`.

### GET /jobs
`?state=running&limit=50&before=<created_at>` → `{"jobs": [{job_id, state, campaign_name, created_at, finished_at, progress, status_url}], "next_before": ...}`.
Trang tiếp: truyền `next_before` vào `before`.

### GET /files/...
Link trong `output_video_url` có hạn `FILE_URL_TTL_HOURS` (mặc định 7 ngày). Thiếu hoặc sai `exp`/`sig` thì trả `404`.
Video bị xoá sau `OUTPUT_RETENTION_DAYS` kể từ lúc job xong.

### Mã lỗi HTTP

| HTTP | Khi nào |
|---|---|
| 400 | `video_url` / `callback_url` không hợp lệ: không phải https, ngoài allowlist, trỏ vào IP nội bộ |
| 401 | Thiếu chữ ký, sai chữ ký, `client_id` lạ, timestamp lệch quá 5 phút, nonce dùng lại |
| 404 | Job không tồn tại hoặc thuộc client khác; link file sai hoặc hết hạn |
| 409 | `/retry` không có item nào retry được; `/webhooks/resend` khi job chưa xong hoặc không có `callback_url` |
| 413 | Body > 2MB hoặc > 50 items |
| 422 | JSON sai schema |
| 429 | Vượt rate limit; IP sai xác thực ≥ 10 lần/phút (bị chặn 1 phút); client đang có ≥ 5 job chưa xong |
| 503 | Ổ đĩa còn dưới `MIN_FREE_GB`: tạm ngừng nhận job (`Retry-After: 300`) |

---

## 5. Webhook

Nếu job có `callback_url`, service gửi `POST` cho 2 loại event:

| Event | Khi nào | Body |
|---|---|---|
| `item.done` | Mỗi item kết thúc (`done` / `no_changes` / `failed` / `cancelled` / `skipped`). Tắt bằng `WEBHOOK_ITEM_EVENTS=false` | `{"event", "event_id", "job_id", "advertiser_id", "campaign_id", "campaign_name", "progress", "item": {...như 1 item của GET /jobs/{id}}}` |
| `job.done` | Mọi item đã kết thúc (cả khi bị huỷ) | Object Job như `GET /jobs/{id}` (bỏ `client_id`) + `"event"`, `"event_id"` |

| Header | Giá trị |
|---|---|
| `X-Magician-Event` | `item.done` / `job.done` |
| `X-Magician-Event-Id` | ID duy nhất của event, **giữ nguyên qua các lần gửi lại** → dùng để chống xử lý trùng |
| `X-Magician-Delivery-Attempt` | Lần gửi thứ mấy (1, 2, ...) |
| `X-Magician-Timestamp` | Unix time lúc gửi |
| `X-Magician-Signature` | `sha256=` + hex `HMAC_SHA256(CALLBACK_SECRET, timestamp + "." + raw_body)` |

- **Retry:** không nhận được 2xx thì gửi lại sau 10s, 30s, 2m, 5m, 15m, 30m (`WEBHOOK_RETRY_DELAYS`), tổng 7 lần trong ~53 phút.
  Hàng đợi webhook lưu trong SQLite nên server restart vẫn gửi tiếp.
- **Thứ tự:** event của cùng 1 job gửi tuần tự đúng thứ tự xảy ra; `job.done` luôn tới sau mọi `item.done`, kể cả khi
  `item.done` phải gửi lại nhiều lần. Event bị bỏ (hết lượt) không chặn event sau.
- **Theo dõi:** `GET /jobs/{id}/webhooks` (trạng thái từng event), `callback_status` của job phản ánh `job.done`.
  Gửi lại thủ công: `POST /jobs/{id}/webhooks/resend`.
- Không follow redirect. `callback_url` được kiểm tra SSRF lại lúc gửi.
- Gọi `/retry` làm job chạy lại → khi xong có `job.done` mới (event_id mới).

Phía nhận phải verify bằng **raw body** (không parse JSON rồi stringify lại), từ chối nếu timestamp lệch quá 5 phút,
trả 2xx nhanh (xử lý nặng thì đẩy vào queue riêng) và bỏ qua `event_id` đã xử lý:

```python
from magician_client import verify_webhook
ok = verify_webhook(CALLBACK_SECRET, raw_body, headers["X-Magician-Timestamp"], headers["X-Magician-Signature"])
```
```js
import { verifyWebhook } from "./magician-client.mjs";
// Express: app.post("/magician-webhook", express.raw({ type: "application/json" }), (req, res) => {...})
const ok = verifyWebhook(CALLBACK_SECRET, req.body, req.get("X-Magician-Timestamp"), req.get("X-Magician-Signature"));
```

---

## 6. Bảo mật & giới hạn

| Lớp | Cơ chế |
|---|---|
| Transport | Chỉ public qua Cloudflare Tunnel (TLS). Server bind `127.0.0.1`, không mở port ra ngoài |
| Xác thực | HMAC-SHA256 theo từng client, so sánh constant-time. Secret không bao giờ đi trên đường truyền |
| Chống replay | Timestamp ±300s + nonce chỉ dùng 1 lần |
| Toàn vẹn | Chữ ký bao gồm method, path, query, SHA256 của body |
| Phân quyền | Mỗi client chỉ xem được job của mình; job của client khác trả `404` (không lộ job có tồn tại hay không) |
| Brute force | Sai xác thực ≥ 10 lần/phút/IP → chặn IP 1 phút. IP lấy từ `CF-Connecting-IP` |
| Rate limit | `RATE_LIMIT_PER_MINUTE` request/phút/client (mặc định 60), gấp đôi số đó/IP; tối đa 5 job đang chạy/client |
| SSRF | `video_url` và `callback_url`: chỉ https, host phải resolve ra IP public (chặn loopback, LAN, link-local, metadata cloud), phải khớp allowlist; kiểm tra lại ở **mọi** hop redirect khi tải video |
| Giới hạn tài nguyên | Body ≤ 2MB, ≤ 50 items/job, video ≤ 300MB |
| Link tải file | Ký HMAC + hạn dùng; tên file và job_id được kiểm tra bằng regex (chặn path traversal) |
| Cache | Mọi response có `Cache-Control: private, no-store` để Cloudflare không cache video và dữ liệu API |
| Webhook | Ký HMAC có timestamp |
| Thông tin hệ thống | `/health` public chỉ trả `{"ok": true}`; `/status`, `/docs`, `/openapi.json` cần auth hoặc tắt |

Xoay vòng secret: thêm client mới vào `API_KEYS` (vd `dashboard-v2:<secret mới>`), restart, chuyển client sang secret mới,
rồi xóa entry cũ.

Khuyến nghị thêm ở phía Cloudflare: WAF rule chỉ cho IP của dashboard gọi `/jobs*`, hoặc bật Cloudflare Access (service token).

---

## 7. Cấu hình (`.env`)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `OPENAI_API_KEY` | | Bắt buộc |
| `REWRITE_MODEL` | `gpt-6-luna` | Model viết lại script |
| `TRANSCRIBE_MODEL` | `gpt-transcribe` | `whisper-1` thì dùng timestamp của model thay vì tự cắt câu |
| `TIKTOK_ACCESS_TOKEN` | | Để service tự lấy reject reason. Quyền: Ads Management > Ad > Read Ads |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | Dùng để tạo `status_url` và `output_video_url` |
| `API_KEYS` | | `client_id:secret,...` |
| `REQUIRE_SIGNED_REQUESTS` | `true` | `false` = cho phép thêm `X-API-Key` |
| `CALLBACK_SECRET` | | Ký webhook |
| `FILE_SIGNING_KEY` | ngẫu nhiên mỗi lần start | Ký link `/files`. Phải đặt cố định, nếu không link cũ chết sau restart |
| `FILE_URL_TTL_HOURS` | `168` | Hạn link tải |
| `RATE_LIMIT_PER_MINUTE` | `60` | Mỗi client |
| `MAX_BODY_BYTES` / `MAX_ITEMS_PER_JOB` / `MAX_ACTIVE_JOBS_PER_CLIENT` / `MAX_VIDEO_MB` | `2000000` / `50` / `5` / `300` | |
| `VIDEO_URL_ALLOWLIST` | Bunny + TikTok CDN | Suffix domain, phân tách dấu phẩy; trống = mọi host public |
| `CALLBACK_URL_ALLOWLIST` | trống | Như trên, cho webhook |
| `ALLOW_HTTP_URLS` | `false` | |
| `ALLOW_PRIVATE_URLS` | `false` | Chỉ bật khi test local |
| `ENABLE_DOCS` | `false` | Swagger `/docs` |
| `DEVICE` | `auto` | `auto` / `cuda` / `cuda:1` / `rocm` / `mps` / `cpu` |
| `TTS_ENGINE` | `omnivoice` | `omnivoice` hoặc `xtts` |
| `OMNIVOICE_LANG_MAP` | `ar:afb` | Mã ngôn ngữ OmniVoice cho tiếng Arab: `afb` Gulf/UAE, `ars` Saudi, `arz` Ai Cập, `arb` chuẩn |
| `OMNIVOICE_NUM_STEP` | `32` | Số bước diffusion; giảm thì nhanh hơn, chất lượng giảm |
| `OMNIVOICE_PYTHON` / `XTTS_PYTHON` | trống | Python chạy worker TTS; trống = cùng Python với server. XTTS bắt buộc env riêng |
| `TTS_WORKERS` | `0` (tự tính) | Số worker clone giọng |
| `MAX_TTS_PER_GPU` | `4` | |
| `ITEM_CONCURRENCY` | `0` (tự tính) | Số item xử lý song song trên toàn server |
| `MAX_TEMPO` | `1.3` | Tốc độ đọc tối đa để vừa khung thời gian cũ |
| `VISION_MODEL` | `gpt-6-luna` | Model đọc frame, triage + lập kế hoạch sửa (phải nhận ảnh) |
| `OCR_FPS` / `PLANNER_FPS` / `MAX_PLANNER_FRAMES` | `2` / `1` / `60` | Mật độ frame cho OCR và cho planner |
| `MAX_CUT_RATIO` | `0.35` | Thời lượng cắt + thời lượng xoá vùng lớn (> 30% khung) vượt tỉ lệ này thì coi là không sửa được |
| `VISUAL_METHOD` | `auto` | Mặc định cho `visual_method` của request: `auto` hoặc `blur` |
| `RECHECK` / `RECHECK_RETRY` | `true` / `true` | Kiểm tra lại video sau khi sửa / sửa thêm 1 lượt nếu còn vi phạm |
| `AIGC_OVERLAY` | `true` | Chèn nhãn "AI-generated voice" khi voice-over được clone |
| `VIDEO_CRF` | `18` | Chất lượng encode lại (thấp = đẹp hơn, file nặng hơn) |
| `MODELS_DIR` | `DATA_DIR/models` | Nơi lưu model EasyOCR + LaMa (tự tải lần đầu) |
| `COQUI_TOS_AGREED` | | Chỉ cho XTTS: đặt `1` để xác nhận license CPML |
| `DATA_DIR` | `./data` | `magician.db`, `jobs/` (file trung gian), `outputs/` (video kết quả), `logs/` |
| `DB_PATH` | `DATA_DIR/magician.db` | SQLite |
| `STAGE_TIMEOUTS` | `download:180,separate:240,transcribe:240,triage:180,ocr:300,plan:240,rewrite:180,tts:240,render:900,mux:180` | Giây tối đa mỗi bước |
| `TTS_CALL_TIMEOUT` | `120` | Giây tối đa 1 lần gọi worker TTS, quá thì kill worker |
| `DOWNLOAD_CONNECT_TIMEOUT` | `30` | Giây chờ kết nối khi tải video |
| `MIN_FREE_GB` | `5` | Dưới mức này thì ngừng nhận / chạy job |
| `WORK_RETENTION_HOURS` / `OUTPUT_RETENTION_DAYS` / `JOB_RETENTION_DAYS` | `24` / `7` / `30` | Dọn file trung gian / video kết quả / bản ghi |
| `WEBHOOK_ITEM_EVENTS` | `true` | Gửi `item.done` mỗi item |
| `WEBHOOK_RETRY_DELAYS` | `10,30,120,300,900,1800` | Giây chờ giữa các lần gửi lại webhook |
| `WEBHOOK_TIMEOUT` | `20` | Giây chờ phía nhận trả lời |
| `DEDUP_ACTIVE` | `true` | Gửi trùng ad đang chạy thì trả job cũ |
| `OOM_COOLDOWN_MINUTES` | `10` | Thời gian giảm concurrency sau CUDA OOM |
| `LOG_DIR` / `LOG_MAX_MB` / `LOG_BACKUPS` | `DATA_DIR/logs` / `50` / `10` | Log file xoay vòng |
| `TTS_WARMUP` | `true` | Khởi động worker TTS lúc start (tắt khi chạy test) |

---

## 8. Phần cứng & hiệu năng

Khi khởi động, service tự nhận GPU và tính số worker (xem `GET /status`):

| Máy | Cách tính |
|---|---|
| NVIDIA / AMD ROCm | Mỗi worker ~3GB VRAM (OmniVoice) / ~2.6GB (XTTS); Demucs ~1GB trên GPU đầu tiên; tối đa 4 worker/GPU; nhiều GPU thì chia đều |
| Apple Silicon (MPS) | GPU dùng chung RAM, tối đa 2 worker |
| CPU | 1–3 worker tùy số core và RAM |

- Mỗi worker TTS là 1 process riêng (RSS ~2.2GB OmniVoice / ~0.8GB XTTS). Lúc chạy chỉ bật thêm worker nếu VRAM/RAM còn đủ.
- **Windows:** mỗi worker OmniVoice chiếm ~9GB commit (RAM + pagefile) dù RSS chỉ ~2GB, nên service tính theo commit còn trống.
  Máy 32GB đang chạy app khác thường chỉ đủ 1 worker. Linux/Docker không bị giới hạn này.
- Số video song song = 3 × số worker + 2.

Đo trên RTX 3060 12GB (Windows), 10 video Arabic 15s chạy song song:

| Cấu hình | Thời gian | VRAM đỉnh | RAM đỉnh |
|---|---|---|---|
| XTTS, 1 worker | 64s | 3.5GB | 7.2GB |
| XTTS, 3 worker | 47s | 8.8GB | 8.2GB |
| XTTS, 3 worker + cache video trùng | 22s | 8.3GB | 8.5GB |
| **OmniVoice, 1 worker + cache** | **20s** | 7.5GB | 5.4GB |

Chất lượng (A/B 36 câu ar/en/hi/pt, cùng giọng mẫu; `data/abtest/eval.json`):

| | Giống giọng (ECAPA cosine ↑) | Đọc sai chữ (CER ↓) | Tự nhiên (UTMOS ↑) | Thời gian sinh / giây audio |
|---|---|---|---|---|
| XTTS-v2 | 0.55 | 6.4% | 3.20 | 0.36 |
| OmniVoice | **0.74** | 5.9% | **3.77** | 0.52 |

Chi phí OpenAI mỗi video 15s: `gpt-transcribe` ~5 lần gọi (15s audio), `gpt-6-luna` 1 lần (~735 token gửi vào, 200–1,200 token
nhận về). Số liệu thực tế của từng item nằm trong `usage`.

---

## 9. Vận hành

### Chạy thường trực trên Windows (Task Scheduler)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\service.ps1 install    # tạo + start 2 task
powershell -ExecutionPolicy Bypass -File scripts\service.ps1 status     # trạng thái task + /ready
powershell -ExecutionPolicy Bypass -File scripts\service.ps1 restart    # sau khi sửa code / .env
powershell -ExecutionPolicy Bypass -File scripts\service.ps1 stop|start|logs|uninstall [-Only api|tunnel]
```
- 2 task: `Magician-API` (uvicorn `127.0.0.1:8000`) và `Magician-Tunnel` (cloudflared, config `magician.yml`).
  Chạy khi user đăng nhập, không giới hạn thời gian.
- Mỗi task chạy `scripts\run_service.ps1` làm supervisor: process chết thì 5s sau tự bật lại (test: kill uvicorn → chạy lại,
  `/ready` 200 sau ~30s). Supervisor chết thì Task Scheduler chạy lại (tối đa 999 lần, cách 1 phút).
- Biến môi trường của supervisor (tùy chọn): `MAGICIAN_PYTHON` (mặc định `D:\envs\omnivoice\python.exe`),
  `MAGICIAN_CLOUDFLARED`, `MAGICIAN_TUNNEL_CONFIG`, `HF_HOME` (mặc định `D:\cache\hf`), `MAGICIAN_TEMP` (`D:\cache\tmp`).
- Task chạy theo phiên đăng nhập: máy reboot thì phải đăng nhập Windows thì service mới lên (muốn chạy không cần
  đăng nhập phải dùng tài khoản có mật khẩu + quyền admin, hoặc NSSM).

### Cloudflare Tunnel
- Tunnel `magician` → `http://localhost:8000`, config riêng `C:\Users\Qhuy\.cloudflared\magician.yml`.
- **Không** dùng `config.yml` mặc định (của tunnel `hoanganh`). Mọi lệnh `cloudflared tunnel ...` cho magician phải kèm
  `--config ...magician.yml`, nếu không DNS sẽ bị trỏ nhầm.

### Log
| File | Nội dung |
|---|---|
| `data/logs/magician.log` | Log app + access log, xoay vòng `LOG_MAX_MB` × `LOG_BACKUPS`. Mỗi dòng có `[job_id(8)/smart_ad_id]` |
| `data/logs/api.out.log`, `api.err.log` | stdout/stderr của uvicorn (thanh tiến trình load model, crash trước khi logging chạy) |
| `data/logs/tunnel.err.log` | Log cloudflared |
| `data/logs/supervisor.log` | Thời điểm start / exit / restart của từng process |

Timeline của 1 job (mỗi lần chạy, retry, lỗi): `GET /jobs/{id}/events`.

### Giám sát
- `GET /ready` (public): dùng cho uptime monitor. `503` khi thiếu DB, ffmpeg, scheduler chết, không có worker TTS
  (trong lúc khởi động ~30s cũng là 503) hoặc ổ đĩa dưới `MIN_FREE_GB`.
- `GET /status` (ký): `ready` từng check, `scheduler` (`concurrency`, `running`, `pending`, `retrying`, `active_jobs`,
  `errors_last_hour` theo `error_code`, `webhooks` theo trạng thái, `disk_free_gb`, `paused`), phần cứng, worker TTS.
- `GET /metrics` (Prometheus; không cần ký nếu gọi từ chính máy chủ, qua tunnel phải ký):
  `magician_items_running/pending/retrying`, `magician_items_finished_total{state}`, `magician_item_errors_total{code}`,
  `magician_stage_seconds_sum/count{stage}`, `magician_webhooks{state}`, `magician_tts_workers_alive`, `magician_disk_free_gb`...

### Watchdog & tự bảo vệ
- **Timeout từng bước** (`STAGE_TIMEOUTS`): quá giờ thì kill subprocess của bước đó, item `stage_timeout` → retry.
- **Worker TTS**: mỗi lần gọi có timeout `TTS_CALL_TIMEOUT`; worker treo bị kill. Watchdog 30s/lần bỏ worker chết và spawn lại.
- **CUDA OOM**: item retry, đồng thời số item chạy song song giảm một nửa trong `OOM_COOLDOWN_MINUTES`.
- **Hết đĩa**: còn dưới `MIN_FREE_GB` thì ngừng lấy item mới, `POST /jobs` trả `503`, `/ready` 503; đủ đĩa thì tự chạy lại.
- **Scheduler**: hàng đợi cấp item, tối đa `item_concurrency` item cùng lúc trên toàn server; xoay vòng giữa các client
  (client gửi 50 item không chặn client khác), trong 1 client thì FIFO theo job.

### Dữ liệu & dọn dẹp
- `data/magician.db` (SQLite, WAL): job + request gốc, event log, hàng đợi webhook. Backup: copy file khi service dừng,
  hoặc `sqlite3 data/magician.db ".backup backup.db"` khi đang chạy.
- `data/jobs/<job_id>/<smart_ad_id>/`: file trung gian (video gốc, stem, transcript, plan, TTS) để retry / resume.
  Tự xoá sau `WORK_RETENTION_HOURS` (24h) kể từ lúc job xong.
- `data/outputs/<job_id>/<smart_ad_id>.mp4`: video kết quả, tự xoá sau `OUTPUT_RETENTION_DAYS` (7 ngày, khớp hạn link).
- Bản ghi job / event / webhook xoá sau `JOB_RETENTION_DAYS` (30 ngày). Dọn dẹp chạy lúc start rồi mỗi giờ.
- `data/jobs/*/job.json` của bản cũ được import vào SQLite 1 lần lúc start.

### Restart / crash
- Job / item đang chạy khi server tắt hoặc crash được **chạy tiếp** khi start lại (event `resumed`), từ bước đã lưu
  (test thật: kill giữa bước OCR → sau restart download / separate / transcribe / triage được dùng lại, chỉ chạy tiếp từ plan).
- Webhook chưa gửi xong được gửi tiếp.
- Nonce chống replay và rate limit nằm trong RAM nên reset khi restart (chữ ký vẫn chỉ hợp lệ trong 5 phút).

---

## 10. Hạn chế

- **License:** cả 2 engine đều **chỉ cho phép dùng phi thương mại**: OmniVoice model là CC-BY-NC (code Apache 2.0),
  XTTS-v2 là CPML (Coqui đã đóng cửa, không mua được bản thương mại). Cần provider TTS có license thương mại trước khi
  dùng cho quảng cáo thật.
- **OmniVoice nói ngôn ngữ khác giọng gốc:** giọng Arab đọc tiếng Bồ Đào Nha bị ASR nhận sai ~20% ký tự (giọng nặng).
  Khi giọng mẫu và câu mới cùng ngôn ngữ (trường hợp bình thường của pipeline) thì không gặp.
- **Sửa hình:** inpaint đẹp nhất khi nền đứng yên (quay màn hình); nền chuyển động nhanh có thể thấy vệt mờ.
  Text mới dùng font Noto (Arab / Devanagari / Latin), màu chữ + viền đoán từ text cũ, chưa bắt chước được font đặc biệt
  hay hiệu ứng (gradient, bóng đổ, chữ chạy). OCR có thể sót text nhỏ hoặc text nghiêng.
- **Tốc độ OCR:** ~0.4s/frame trên RTX 3060 (video 45s đổi cảnh liên tục mất ~40s; màn hình tĩnh nhanh hơn nhiều).
- **Link video:** link Bunny/TikTok trong JSON hết hạn sau khoảng 1 giờ, phải gửi job ngay sau khi lấy JSON.
- **Kết quả rewrite:** không cố định giữa các lần chạy; cùng một video có lần model sửa, có lần từ chối.
- **Giọng clone:** chất lượng phụ thuộc giọng gốc sau khi tách nhạc nền; nên nghe lại trước khi dùng.
- **1 instance:** job / hàng đợi / webhook lưu SQLite (bền qua restart) nhưng chỉ 1 process được đọc ghi; rate limit và
  nonce nằm trong RAM. Chạy nhiều máy GPU dùng chung hàng đợi cần chuyển sang Postgres + Redis.
- **Kill item đang chạy:** ffmpeg và bước tải video dừng ngay; bước đang gọi OpenAI hoặc Demucs phải chờ bước đó trả về
  (OpenAI có timeout 180s) mới giải phóng slot, dù trạng thái item đã là `cancelled`.
