"""Log ra console + file xoay vòng (DATA_DIR/logs/magician.log), mỗi dòng kèm [job_id/smart_ad_id] nếu có."""
import logging
from logging.handlers import RotatingFileHandler

from . import runctx
from .config import settings

FORMAT = "%(asctime)s %(levelname)s %(name)s [%(ctx)s]: %(message)s"


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = runctx.current()
        record.ctx = f"{getattr(ctx, 'job_id', '')[:8]}/{getattr(ctx, 'smart_ad_id', '')}" if ctx else "-"
        return True


_file_handler: logging.Handler | None = None


def setup() -> logging.Handler:
    global _file_handler
    if _file_handler:
        return _file_handler
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(FORMAT)
    flt = ContextFilter()
    _file_handler = RotatingFileHandler(settings.log_dir / "magician.log", maxBytes=settings.log_max_mb * 1024 * 1024,
                                        backupCount=settings.log_backups, encoding="utf-8")
    console = logging.StreamHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (_file_handler, console):
        h.setFormatter(fmt)
        h.addFilter(flt)
        root.addHandler(h)
    return _file_handler


def attach_uvicorn() -> None:
    """uvicorn tự cấu hình logger riêng (không propagate) => gắn thêm file handler cho access/error log."""
    h = setup()
    for name in ("uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        if h not in lg.handlers:
            lg.addHandler(h)
