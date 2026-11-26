# Magician

FastAPI service sửa voice-over của TikTok ad bị reject vì policy.

Pipeline mỗi ad (`app/pipeline.py`): download video → ffmpeg tách audio → Demucs (`htdemucs`) tách vocal/BGM →
OpenAI transcribe (`gpt-transcribe` không trả timestamp nên cắt câu theo silencedetect trên vocal stem rồi transcribe từng câu; `whisper-1` thì dùng verbose_json) → reject reason (TikTok `/smart_plus/ad/review_info/`
hoặc `reject_reasons` trong JSON) → `REWRITE_MODEL` chọn segment vi phạm và viết lại → TTS clone giọng (mặc định OmniVoice,
tuỳ chọn XTTS-v2; mẫu từ chính vocal stem) → OmniVoice `duration` / atempo cho vừa khung → ghép vào vocal stem + BGM → mux lại video (copy video stream).

## Môi trường
- Env chính (server + Demucs + OmniVoice): `D:/envs/omnivoice/python.exe` (Python 3.11, torch 2.8.0 cu128, transformers 5.24).
  Cài: torch 2.8.0 từ index cu128 rồi `pip install -r requirements.txt`.
- Env XTTS (chỉ khi `TTS_ENGINE=xtts`): `C:/Users/Qhuy/Miniconda3/envs/magician/python.exe` (torch 2.5.1 cu121, transformers<5),
  đặt `XTTS_PYTHON` trỏ vào đây. Không gộp được 2 env vì xung đột transformers.
- Ổ C: gần đầy (từng còn 198MB → lỗi "paging file too small"). Env/cache mới để trên D: (`D:/envs`, `D:/cache`);
  chạy với `HF_HOME=D:/cache/hf TEMP=D:/cache/tmp` (model OmniVoice nằm ở đó).
- ffmpeg/ffprobe phải có trong PATH. Config qua `.env` (xem `.env.example`).

## Hiệu năng / phần cứng
- `app/hardware.py`: `DEVICE=auto` tự chọn cuda (NVIDIA) / rocm (AMD, torch ROCm vẫn dùng device `cuda:N`) / mps / cpu,
  đọc VRAM free + RAM để tính số worker TTS (`tts_slots`) và `item_concurrency`. Ép tay bằng `TTS_WORKERS`, `ITEM_CONCURRENCY`.
- `app/tts.py`: mỗi worker TTS là 1 **subprocess** (`app/tts_workers/{omnivoice,xtts}_worker.py`) nói JSON-lines qua stdin/stdout,
  nên chạy được bằng Python env khác server. Nhiều instance trong 1 process bị GIL nên không nhanh hơn. Warmup lúc startup.
- OmniVoice batch không nhanh hơn (~2.2s/câu cố định trên 3060), nên 1 request/worker. Có `duration` để sinh vừa khung.
- Windows: mỗi worker OmniVoice commit ~9GB (XTTS ~4GB) dù RSS chỉ ~2GB → `hardware.host_available_gb` dùng commit
  available (GlobalMemoryStatusEx), máy này chỉ đủ 1 worker OmniVoice. Linux không bị giới hạn này.
- A/B XTTS vs OmniVoice: `data/abtest/` (prep.py, gen_*.py, eval.py, eval.json, compare/).
- `JobCache` (`app/pipeline.py`): trong 1 job, video trùng audio/transcript dùng chung kết quả separate/transcribe/rewrite/TTS.
- Đo trên RTX 3060 12GB, 10 video Arabic 15s song song: 64s (1 instance) → 47s (3 worker) → 22s (3 worker + cache).
- Docker: `Dockerfile` (build arg `TORCH_VARIANT` = cu128 | rocm6.4 | cpu; `INSTALL_XTTS=1` thêm venv XTTS), `docker-compose.yml` profile nvidia/amd/cpu.
  Mac Apple Silicon chạy native `scripts/run_mac.sh` (Docker trên Mac không có GPU).

## Sửa hình (text onscreen / pop-up / cắt)
- `app/schemas.py`: nhận format review mới (`ads`, `campaign{}`, `review_status`, `rejected_video_id`, `reasons[{category,detail,timestamps}]`)
  và format cũ (`items`, `reject_reasons: [str]`).
- `app/planner.py`: `triage` (lọc nhanh, chạy song song OCR → unfixable thì dừng, TopDrama 52s → 10s) rồi `plan_fix`
  chọn tool: fix_audio / sync_captions / replace_text / remove_text / remove_region / cut / unfixable. Guardrail: vi phạm ở
  bản chất sản phẩm (cờ bạc, app trả tiền khi xem) → unfixable, không che/né.
- `app/vision.py`: EasyOCR 2fps → text track (gộp box cùng vị trí + nội dung; frame tĩnh dùng lại kết quả OCR).
- `app/video_edit.py`: không còn inpaint — mọi vùng xoá đều làm mờ (blur), replace_text vẽ text mới đè lên vùng đã mờ;
  text mới vẽ bằng ffmpeg drawtext (PIL trên Windows không có raqm nên không nối chữ Arab được);
  màu chữ đoán bằng diff với frame sạch + k-means. Video > `MAX_VIDEO_SECONDS` (180s) bị từ chối ngay sau download.
- `pip install easyocr` kéo torch PyPI ghi đè bản CUDA → luôn cài lại torch/torchvision từ index cu128 sau đó.
- Test: `data/test/visual/run_item.py <ad.json> <video.mp4> <tag>` (ad_synth*.json + synthetic*.mp4 tự tạo; ad_arabic3.json = TopDrama → unfixable).

## Backend: queue / retry / cancel (`app/jobs.py`, `app/store.py`, `app/errors.py`, `app/runctx.py`)
- State trong RAM, ghi xuống SQLite `data/magician.db` (jobs + request gốc, events, webhook outbox) mỗi lần đổi.
  Start lại: item `running` → `pending` (event `resumed`), webhook pending gửi tiếp. `data/jobs/*/job.json` cũ import 1 lần.
- Scheduler cấp item, toàn server tối đa `item_concurrency`, xoay vòng theo client. Item chạy trong thread (`asyncio.to_thread`)
  với `RunContext` (contextvar): stage hiện tại, timeout từng stage (`STAGE_TIMEOUTS`), kill. Subprocess phải chạy qua
  `runctx.run` / `runctx.tracked` để kill được; vòng lặp dài gọi `runctx.check()`. Thread con (ThreadPoolExecutor) KHÔNG
  thừa kế contextvar.
- Pipeline lưu kết quả từng bước vào work dir (`source.mp4`, `vocals.wav`/`bgm.wav`, `transcript.json`, `tracks.pkl`,
  `triage.json`, `plan_raw.json`, `rewrite.json`, `tts_*.wav`; ghi `.part` rồi rename) → retry/resume không gọi lại OpenAI.
  Pipeline chạy trên bản copy `ItemResult`, `jobs.py` copy `RESULT_FIELDS` về khi xong.
- `errors.classify` → `error_code` + `POLICIES` (số lần + delay). `source_expired` không tự retry, cần `/retry` + `video_urls`.
- Webhook outbox: `item.done` + `job.done`, cùng job gửi tuần tự theo rowid, retry `WEBHOOK_RETRY_DELAYS`.
- Dedup: cùng client + smart_ad_id + video + reasons với item chưa xong → trả job cũ; `Idempotency-Key` header.
- Cleanup mỗi giờ (`WORK_RETENTION_HOURS`, `OUTPUT_RETENTION_DAYS`, `JOB_RETENTION_DAYS`); ổ < `MIN_FREE_GB` → pause + 503.

## Lệnh
- Service (Task Scheduler, supervisor `scripts/run_service.ps1` tự restart sau 5s):
  `powershell -ExecutionPolicy Bypass -File scripts/service.ps1 install|status|restart|stop|start|logs|uninstall [-Only api|tunnel]`.
  Đã cài trên máy này (task `Magician-API`, `Magician-Tunnel`, chạy khi user đăng nhập). Sửa code xong: `service.ps1 restart -Only api`.
- Chạy tay: `HF_HOME=D:/cache/hf D:/envs/omnivoice/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --proxy-headers`
  (dừng task api trước, không thì trùng port).
- Kiểm tra compile: `D:/envs/omnivoice/python.exe -m compileall -q app`
- Test backend (pipeline giả, không GPU/OpenAI, ~20s): `D:/envs/omnivoice/python.exe -m pytest tests -q`
- E2E pipeline thật trên instance test port 8010: xem docstring `tests/e2e_real.py` (full / cancel / expired / resume).
  Máy này chỉ đủ commit RAM cho 1 worker OmniVoice → phải dừng service api trước khi chạy instance test.
- Log: `data/logs/magician.log` (có `[job/ad]`), `api.err.log`, `tunnel.err.log`, `supervisor.log`.
- Git: https://github.com/Qhuy204/Magician.git (`data/`, `.env` bị ignore).
- Shell heredoc dài (>~150 dòng) bị cắt → ghi script ra file bằng tool write rồi chạy.

## Lỗi đã gặp
- ffmpeg đọc lệnh từ stdin: khi stdin là pipe không có EOF (pytest, 1 số môi trường service) process ffmpeg TREO dù
  input là file. Mọi lệnh ffmpeg phải có `-nostdin`. Render qua pipe (video_edit.render) mà exception giữa chừng thì
  decoder vẫn ghi vào pipe không ai đọc → wait() treo vĩnh viễn: phải wait(timeout) rồi kill.

## API
- `POST /jobs?callback_url=...` (+ header `Idempotency-Key`) body = JSON review (format cũ `items` hoặc mới `ads`). 202 + `job_id`.
- `GET /jobs`, `GET /jobs/{id}` (progress, queue_position, item stage/attempt/error_code/retryable/next_retry_at),
  `GET /jobs/{id}/events`, `POST /jobs/{id}/retry`, `POST /jobs/{id}/cancel` (kill), `GET /jobs/{id}/webhooks`,
  `POST /jobs/{id}/webhooks/resend`, `GET /ready` (public), `GET /status`, `GET /metrics` (localhost không cần ký).
- `GET /files/{job_id}/{smart_ad_id}.mp4?exp=&sig=`: video kết quả.
- Webhook: `item.done` / `job.done`, headers `X-Magician-Event`, `X-Magician-Event-Id`, `X-Magician-Signature` =
  `sha256=HMAC(CALLBACK_SECRET, "{ts}.{body}")`. Spec đầy đủ: README mục 4–5.

## Public (Cloudflare Tunnel)
- Domain: https://magician.qhuy204.id.vn → tunnel `magician` (id 66ed9ef9-1d4e-420d-8f9d-054d79d43336) → http://localhost:8000
- Config riêng: `C:\Users\Qhuy\.cloudflared\magician.yml` (KHÔNG dùng `config.yml` mặc định, file đó của tunnel `hoanganh`;
  lệnh `tunnel route dns` không có `--config` sẽ trỏ nhầm DNS sang tunnel đó).
- Chạy tay: `"C:\Program Files (x86)\cloudflared\cloudflared.exe" --config C:\Users\Qhuy\.cloudflared\magician.yml tunnel run`
- Server bind 127.0.0.1. Tunnel + server chạy bằng Task Scheduler (xem Lệnh).

## Bảo mật (`app/security.py`, middleware trong `app/main.py`)
- Client: `API_KEYS=client_id:secret,...`. Mặc định bắt buộc ký HMAC: headers X-Client-Id/X-Timestamp/X-Nonce/X-Signature,
  string_to_sign = ts\nnonce\nMETHOD\npath?query\nsha256(body). Client mẫu: `clients/magician_client.py`, `clients/magician-client.mjs`.
- SSRF: `check_public_url` cho video_url (cả từng hop redirect) và callback_url; allowlist qua `VIDEO_URL_ALLOWLIST`.
  Test local với URL localhost phải đặt `ALLOW_PRIVATE_URLS=true` + `ALLOW_HTTP_URLS=true` + `VIDEO_URL_ALLOWLIST=`.
- `/files` cần `?exp=&sig=` (FILE_SIGNING_KEY). Mọi response `Cache-Control: private, no-store` (Cloudflare từng cache mp4 4h).
- Test bảo mật end-to-end: `python data/test/security_test.py` (30 case, chạy qua domain public).
- Docs đầy đủ cho người dùng API: `README.md`.
