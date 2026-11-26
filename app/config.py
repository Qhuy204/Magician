from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    rewrite_model: str = "gpt-6-luna"
    transcribe_model: str = "gpt-transcribe"

    tiktok_access_token: str = ""
    tiktok_api_base: str = "https://business-api.tiktok.com/open_api/v1.3"

    public_base_url: str = "http://localhost:8000"
    callback_secret: str = ""

    # --- Bảo mật ---
    # Danh sách client "client_id:secret,client2:secret2". API_KEY cũ = client "default".
    api_keys: str = ""
    api_key: str = ""
    # True: chỉ nhận request ký HMAC (X-Client-Id/X-Timestamp/X-Nonce/X-Signature). False: cho phép thêm X-API-Key thuần.
    require_signed_requests: bool = True
    rate_limit_per_minute: int = 60
    max_body_bytes: int = 2_000_000
    max_items_per_job: int = 50
    max_active_jobs_per_client: int = 5
    max_video_mb: int = 300
    # Allowlist domain (suffix, phân tách dấu phẩy). Trống = mọi host public.
    video_url_allowlist: str = "bunnycdn.com,b-cdn.net,tiktokcdn.com,tiktokcdn-us.com,byteoversea.com,ibyteimg.com,bytelemon.com"
    callback_url_allowlist: str = ""
    allow_http_urls: bool = False
    # Chỉ bật khi test local: cho phép tải/gọi URL trỏ vào IP nội bộ
    allow_private_urls: bool = False
    # Ký link /files; trống = sinh ngẫu nhiên mỗi lần start (link cũ mất hiệu lực sau restart)
    file_signing_key: str = ""
    file_url_ttl_hours: int = 168
    enable_docs: bool = False

    data_dir: Path = Path("./data")
    # Model tải về lúc chạy (EasyOCR, LaMa). Trống = DATA_DIR/models
    models_dir: Path | None = None

    # --- Sửa hình ---
    # Model đọc frame + lập kế hoạch sửa (cần hỗ trợ ảnh)
    vision_model: str = "gpt-6-luna"
    # Số frame/giây cho OCR và cho planner
    ocr_fps: float = 2.0
    planner_fps: float = 1.0
    max_planner_frames: int = 60
    # Tỉ lệ thời lượng tối đa được cắt bỏ; vượt quá thì coi là không sửa được
    max_cut_ratio: float = 0.35
    # Mặc định cho visual_method của request: auto | blur
    visual_method: str = "auto"
    lama_url: str = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    video_crf: int = 18
    # auto | cuda | rocm | mps | cpu (cuda/rocm có thể kèm index, vd cuda:1)
    device: str = "auto"
    # 0 = tự tính theo VRAM/RAM; >0 = ép số worker TTS
    tts_workers: int = 0
    max_tts_per_gpu: int = 4
    # omnivoice (mặc định, clone giọng giống hơn, có điều khiển độ dài) | xtts
    tts_engine: str = "omnivoice"
    # Python dùng chạy worker TTS; trống = cùng Python với server. XTTS cần env riêng (transformers<5)
    omnivoice_python: str = ""
    xtts_python: str = ""
    omnivoice_model: str = "k2-fsa/OmniVoice"
    omnivoice_num_step: int = 32
    # Map mã ngôn ngữ của transcribe -> mã OmniVoice. ar:afb = Gulf Arabic (UAE); Saudi: ar:ars, Ai Cập: ar:arz, chuẩn: ar:arb
    omnivoice_lang_map: str = "ar:afb"
    coqui_tos_agreed: bool = False
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    demucs_model: str = "htdemucs"
    max_tempo: float = 1.3
    # 0 = tự tính theo số XTTS worker
    item_concurrency: int = 0
    # Khởi động sẵn worker TTS lúc startup (tắt khi chạy test)
    tts_warmup: bool = True

    # --- Vận hành: queue / retry / timeout / dọn dẹp ---
    # SQLite lưu job + event + webhook outbox. Trống = DATA_DIR/magician.db
    db_path: Path | None = None
    # Timeout từng stage (giây), quá thì kill và retry 1 lần (error_code=stage_timeout)
    stage_timeouts: str = ("download:180,separate:240,transcribe:240,triage:180,ocr:300,plan:240,"
                           "rewrite:180,tts:240,render:900,mux:180")
    # Timeout 1 lần gọi worker TTS; quá thì kill worker, spawn lại
    tts_call_timeout: float = 120
    download_connect_timeout: float = 30
    # Ổ chứa DATA_DIR còn ít hơn => /ready=503, POST /jobs=503, ngừng lấy item mới
    min_free_gb: float = 5
    work_retention_hours: float = 24
    output_retention_days: float = 7
    job_retention_days: float = 30
    # Webhook: gửi thêm event item.done mỗi khi 1 item xong (ngoài job.done)
    webhook_item_events: bool = True
    # Giây chờ giữa các lần gửi lại webhook (số lần thử = len + 1)
    webhook_retry_delays: str = "10,30,120,300,900,1800"
    webhook_timeout: float = 20
    # Gửi lại cùng ad (cùng video + reasons) khi đang xử lý => trả job cũ
    dedup_active: bool = True
    # Sau khi gặp CUDA OOM: giảm item_concurrency trong N phút
    oom_cooldown_minutes: float = 10
    log_dir: Path | None = None
    log_max_mb: int = 50
    log_backups: int = 10

    def stage_timeout_map(self) -> dict[str, float]:
        return {k.strip(): float(v) for k, v in (x.split(":", 1) for x in self.stage_timeouts.split(",") if ":" in x)}

    def webhook_delays(self) -> list[float]:
        return [float(x) for x in self.webhook_retry_delays.split(",") if x.strip()]


settings = Settings()
if not settings.models_dir or str(settings.models_dir) in ("", "."):
    settings.models_dir = settings.data_dir / "models"
settings.db_path = settings.db_path or settings.data_dir / "magician.db"
settings.log_dir = settings.log_dir or settings.data_dir / "logs"
if not settings.file_signing_key:
    import secrets

    settings.file_signing_key = secrets.token_hex(32)
