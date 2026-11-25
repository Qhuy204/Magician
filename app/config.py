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
    video_url_allowlist: str = "bunnycdn.com,b-cdn.net,tiktokcdn.com,tiktokcdn-us.com,byteoversea.com,ibyteimg.com"
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


settings = Settings()
if not settings.models_dir or str(settings.models_dir) in ("", "."):
    settings.models_dir = settings.data_dir / "models"
if not settings.file_signing_key:
    import secrets

    settings.file_signing_key = secrets.token_hex(32)
