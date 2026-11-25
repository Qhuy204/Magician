# Magician

API service sửa voice-over của TikTok ad bị reject vì policy. Nhận JSON danh sách ad (dạng `historical_reviews`),
tải video, tìm câu thoại vi phạm, viết lại cho đúng policy, đọc lại bằng giọng clone của chính người nói trong video,
ghép vào video gốc và gửi kết quả về qua webhook.

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
dùng công cụ nào dựa trên lý do reject:

| # | Bước | Công cụ |
|---|---|---|
| 1 | Tải `video_url` (chỉ domain trong allowlist, tối đa 300MB) | httpx |
| 2 | Tách giọng nói khỏi nhạc nền, cắt câu theo khoảng lặng, transcribe từng câu | Demucs `htdemucs` + OpenAI `gpt-transcribe` |
| 3 | Lấy lý do reject (category, detail, timestamps, ảnh frame vi phạm) | `reasons` trong JSON hoặc TikTok `/smart_plus/ad/review_info/` |
| 4 | **Triage**: ad hỏng ở bản chất sản phẩm thì dừng ngay (chạy song song với OCR) | `VISION_MODEL` + frame 1 giây/lần |
| 5 | OCR text trên màn hình, gom thành "text track" (nội dung, vị trí, thời gian) | EasyOCR (GPU) |
| 6 | **Planner** chọn công cụ cho từng vi phạm (bảng dưới) | `VISION_MODEL` |
| 7 | Sửa voice-over nếu cần: viết lại câu vi phạm, clone giọng, ghép lại | `gpt-6-luna` + OmniVoice / XTTS |
| 8 | Sửa hình: xoá text/pop-up, vẽ text mới, cắt đoạn; phụ đề in sẵn tự đổi theo voice-over mới | LaMa inpaint, ffmpeg drawtext |
| 9 | Encode lại video (hoặc chỉ thay audio nếu hình không đổi) | ffmpeg / libx264 |

Công cụ planner có thể chọn:

| Công cụ | Khi nào | Cách làm |
|---|---|---|
| `fix_audio` | Lời thoại vi phạm | Viết lại câu vi phạm cùng ngôn ngữ, đọc bằng giọng clone, vừa khung thời gian cũ |
| `sync_captions` | Có phụ đề in sẵn lặp lại lời thoại | Phụ đề của câu bị sửa được thay bằng câu mới, cùng vị trí / màu / cỡ chữ |
| `replace_text` | Text vi phạm có cách nói hợp lệ | Xoá text cũ, vẽ text mới cùng vị trí, màu chữ + viền lấy từ text cũ |
| `remove_text` | Text vi phạm, không cần thay | Xoá text |
| `remove_region` | Pop-up, banner, số tiền, badge... | Xoá vùng; nền đứng yên thì chép lại nền từ frame sạch, nền chuyển động thì LaMa inpaint, vùng > 20% khung hình thì làm mờ |
| `cut` | Cả shot vi phạm (vd màn hình rút tiền) | Bỏ đoạn đó khỏi cả hình lẫn tiếng; tổng thời lượng cắt tối đa `MAX_CUT_RATIO` (35%) |
| `unfixable` | Vi phạm ở bản chất sản phẩm (cờ bạc, app trả tiền khi xem/chơi, sản phẩm cấm...) hoặc nội dung vi phạm chiếm gần hết video | Không sửa, item ra `no_changes` kèm lý do |

Service xoá hẳn claim vi phạm, không diễn đạt lại bằng từ né, không che một nửa, không dời sang chỗ khác để lọt review.
Vi phạm nằm ngoài video (landing page, targeting) được ghi vào `plan.out_of_scope`.

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
| POST | `/jobs` | ✓ | Tạo job, trả về ngay `202` |
| GET | `/jobs/{job_id}` | ✓ | Trạng thái và kết quả (chỉ xem được job của chính client đó) |
| GET | `/files/{job_id}/{smart_ad_id}.mp4?exp=&sig=` | link ký | Tải video kết quả |
| GET | `/status` | ✓ | Phần cứng, engine + số worker TTS, độ dài hàng đợi |
| GET | `/health` | – | `{"ok": true}` |

`/docs` và `/openapi.json` tắt mặc định (`ENABLE_DOCS=true` để bật).

### POST /jobs

Query: `callback_url` (tùy chọn, ghi đè field trong body).

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
| `video_id` / `rejected_video_id`, `preview_video_id` | | Khớp reject reason của TikTok với đúng video |
| `reasons` / `reject_reasons` | | Mỗi phần tử là chuỗi hoặc object `{category, detail, timestamps, affected_regions, affected_placements, violation_frames}`. `timestamps` dạng `"0:22–0:24"` giúp planner tìm đúng chỗ. Có thì dùng luôn, không gọi TikTok API |
| `violation_frames` | | URL ảnh frame vi phạm (nếu có); planner nhìn ảnh để xác định phần bị flag |
| `callback_url` | | Webhook nhận kết quả, chỉ https |
| `process_statuses` | | Mặc định `["UNAVAILABLE","PART_AVAILABLE"]` |

Không có `reasons` và không cấu hình `TIKTOK_ACCESS_TOKEN` thì item ra `no_changes`.

Response `202`:
```json
{"job_id": "dc8ef7f0ae22499692718be2adca6c0e", "status_url": "https://magician.qhuy204.id.vn/jobs/dc8e...", "items": 14}
```

### GET /jobs/{job_id}
```json
{
  "job_id": "02bb2eb6d90741f7ae4ac0bbb0821a51",
  "client_id": "dashboard",
  "state": "done",
  "advertiser_id": "7687959493501976597",
  "campaign_name": "shieldy_17.11_hook_ar R1.2",
  "callback_url": "https://your-host/magician-webhook",
  "callback_status": "HTTP 200",
  "cache_hits": {"rewrite": 8, "tts": 36},
  "items": [
    {
      "smart_ad_id": "1882107813581906",
      "smart_ad_name": "Ad group 2_10",
      "state": "done",
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
      "usage": {"transcribe_calls": 5, "transcribe_seconds": 15, "chat_calls": 1, "chat_prompt_tokens": 735, "chat_completion_tokens": 224},
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
    }
  ]
}
```

Trạng thái job: `queued` → `running` → `done` | `failed` (`failed` gồm cả job bị gián đoạn do server restart).

Trạng thái item:

| `state` | Ý nghĩa |
|---|---|
| `pending` / `running` | Đang chờ / đang xử lý |
| `done` | Có video mới ở `output_video_url` |
| `no_changes` | Không sửa; lý do nằm trong `error` và `plan` (không có reject reason, vi phạm ở bản chất sản phẩm, cần cắt quá nhiều...). `plan.stage = "triage"` nghĩa là bị dừng ở lượt lọc nhanh |
| `skipped` | `status` không thuộc `process_statuses` |
| `failed` | Lỗi, chi tiết trong `error` (vd link video hết hạn → 403) |

### GET /files/...
Link trong `output_video_url` có hạn `FILE_URL_TTL_HOURS` (mặc định 7 ngày). Thiếu hoặc sai `exp`/`sig` thì trả `404`.

### Mã lỗi

| HTTP | Khi nào |
|---|---|
| 400 | `video_url` / `callback_url` không hợp lệ: không phải https, ngoài allowlist, trỏ vào IP nội bộ |
| 401 | Thiếu chữ ký, sai chữ ký, `client_id` lạ, timestamp lệch quá 5 phút, nonce dùng lại |
| 404 | Job không tồn tại hoặc thuộc client khác; link file sai hoặc hết hạn |
| 413 | Body > 2MB hoặc > 50 items |
| 422 | JSON sai schema |
| 429 | Vượt rate limit; IP sai xác thực ≥ 10 lần/phút (bị chặn 1 phút); client đang có ≥ 5 job chưa xong |

---

## 5. Webhook

Job xong (`done` hoặc `failed`) thì service `POST` toàn bộ object Job (như `GET /jobs/{id}`, bỏ `client_id`) tới `callback_url`.

| Header | Giá trị |
|---|---|
| `X-Magician-Timestamp` | Unix time lúc gửi |
| `X-Magician-Signature` | `sha256=` + hex `HMAC_SHA256(CALLBACK_SECRET, timestamp + "." + raw_body)` |

- Retry 3 lần (chờ 5s, 10s) nếu không nhận được 2xx. Kết quả lần gửi cuối ghi vào `callback_status`.
- Không follow redirect. `callback_url` được kiểm tra SSRF lại lúc gửi.

Phía nhận phải verify bằng **raw body** (không parse JSON rồi stringify lại) và từ chối nếu timestamp lệch quá 5 phút:

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
| `ITEM_CONCURRENCY` | `0` (tự tính) | Số video xử lý song song trong 1 job |
| `MAX_TEMPO` | `1.3` | Tốc độ đọc tối đa để vừa khung thời gian cũ |
| `VISION_MODEL` | `gpt-6-luna` | Model đọc frame, triage + lập kế hoạch sửa (phải nhận ảnh) |
| `OCR_FPS` / `PLANNER_FPS` / `MAX_PLANNER_FRAMES` | `2` / `1` / `60` | Mật độ frame cho OCR và cho planner |
| `MAX_CUT_RATIO` | `0.35` | Cần cắt nhiều hơn tỉ lệ này thì coi là không sửa được |
| `VIDEO_CRF` | `18` | Chất lượng encode lại (thấp = đẹp hơn, file nặng hơn) |
| `MODELS_DIR` | `DATA_DIR/models` | Nơi lưu model EasyOCR + LaMa (tự tải lần đầu) |
| `COQUI_TOS_AGREED` | | Chỉ cho XTTS: đặt `1` để xác nhận license CPML |
| `DATA_DIR` | `./data` | `jobs/` (file tạm + `job.json`), `outputs/` (video kết quả) |

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

### Cloudflare Tunnel (Windows, máy hiện tại)
- Tunnel `magician` → `http://localhost:8000`, config riêng `C:\Users\Qhuy\.cloudflared\magician.yml`.
- Chạy: `"C:\Program Files (x86)\cloudflared\cloudflared.exe" --config C:\Users\Qhuy\.cloudflared\magician.yml tunnel run`
- **Không** dùng `config.yml` mặc định (của tunnel `hoanganh`). Mọi lệnh `cloudflared tunnel ...` cho magician phải kèm
  `--config ...magician.yml`, nếu không DNS sẽ bị trỏ nhầm.

### Dữ liệu
- `data/jobs/<job_id>/job.json` – trạng thái job, được nạp lại khi restart.
- `data/jobs/<job_id>/<smart_ad_id>/` – file tạm (audio tách, TTS). Có thể xóa sau khi job xong.
- `data/outputs/<job_id>/<smart_ad_id>.mp4` – video kết quả.
- Service chưa có cơ chế tự dọn; cần cron xóa thư mục cũ hơn `FILE_URL_TTL_HOURS`.

### Restart
Job đang chạy lúc restart bị đánh dấu `failed` (`callback_status: "interrupted by server restart"`); gửi lại JSON để chạy lại.

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
- **Trạng thái lưu trong 1 process:** rate limit, nonce và hàng đợi nằm trong RAM của 1 instance; chạy nhiều instance cần chuyển sang Redis.
