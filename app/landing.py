"""Kiểm tra landing page theo policy Ad format / After-conversion: chạy được trên mobile, không lỗi, không tự tải
file, có chính sách quyền riêng tư / liên hệ, nội dung khớp sản phẩm trong ad. Chỉ báo, không sửa."""
import html
import json
import logging
import re
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from .config import settings
from .security import check_public_url

log = logging.getLogger(__name__)

MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
             "Version/18.0 Mobile/15E148 Safari/604.1")
MAX_BYTES = 3 * 1024 * 1024
STORES = ("play.google.com", "apps.apple.com", "itunes.apple.com", "appgallery.huawei.com", "galaxystore.samsung.com")
PRIVACY = re.compile(r"privacy|privacidad|privacidade|confidentialit|datenschutz|riservatezza|الخصوصية", re.I)
CONTACT = re.compile(r"mailto:|tel:|contact|contacto|contato|kontakt|contatti|اتصل|تواصل", re.I)
PLACEHOLDER = re.compile(r"under construction|coming soon|en construcci[oó]n|em constru[cç][aã]o|page not found|404", re.I)
DOWNLOAD_TYPES = ("application/vnd.android.package-archive", "application/octet-stream", "application/x-msdownload",
                  "application/zip")

CONSISTENCY = """Compare a TikTok ad with its landing page. Is the landing page about the same product/brand as the ad
(ad content must be consistent with the promoted product on the landing page)? Respond with JSON only:
{"consistent": bool, "note": str}"""


def _text(page: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()


def check_consistency(client: OpenAI, page: dict, product: str | None, ad_text: str | None) -> dict:
    """So nội dung trang (đã tải bởi check_landing) với sản phẩm + caption SAU KHI sửa."""
    text = page.pop("_text", "")
    if not (product or ad_text) or not (page.get("title") or text):
        return page
    resp = client.chat.completions.create(
        model=settings.rewrite_model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": CONSISTENCY},
                  {"role": "user", "content": json.dumps(
                      {"ad_product": product, "ad_text": ad_text, "landing_url": page.get("final_url"),
                       "landing_title": page.get("title"), "landing_text": text[:3000]}, ensure_ascii=False)}])
    out = json.loads(resp.choices[0].message.content or "{}")
    page["consistent"], note = out.get("consistent"), str(out.get("note", ""))[:300]
    if page["consistent"] is False:
        page["issues"] = [*page.get("issues", []), f"nội dung landing page không khớp với sản phẩm trong ad: {note}"]
        page["ok"] = False
    return page


def check_landing(client: OpenAI | None, url: str, product: str | None, ad_text: str | None) -> dict:
    issues, warnings = [], []
    try:
        check_public_url(url, "")
    except ValueError as exc:
        return {"url": url, "ok": False, "issues": [f"URL không hợp lệ / trỏ vào mạng nội bộ: {exc}"]}
    hook = lambda req: check_public_url(str(req.url), "")
    try:
        with httpx.Client(headers={"User-Agent": MOBILE_UA, "Accept-Language": "en,es,pt,ar;q=0.8"},
                          follow_redirects=True, max_redirects=8, timeout=httpx.Timeout(20, connect=10),
                          event_hooks={"request": [hook]}) as c, c.stream("GET", url) as r:
            final, status = str(r.url), r.status_code
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            dispo = r.headers.get("content-disposition", "").lower()
            body = b""
            for chunk in r.iter_bytes(1 << 16):
                body += chunk
                if len(body) > MAX_BYTES:
                    break
            hops = len(r.history)
    except ValueError as exc:  # redirect vào mạng nội bộ
        return {"url": url, "ok": False, "issues": [f"landing page chuyển hướng tới địa chỉ không hợp lệ: {exc}"[:300]]}
    except httpx.TransportError as exc:
        # lỗi mạng từ phía Magician (timeout, DNS...) không chứng minh được trang hỏng => không kết luận
        return {"url": url, "ok": None, "issues": [],
                "warnings": [f"không kiểm tra được landing page: {exc.__class__.__name__}: {exc}"[:300]]}
    except httpx.HTTPError as exc:
        return {"url": url, "ok": False, "issues": [f"landing page lỗi: {exc.__class__.__name__}: {exc}"[:300]]}

    host = urlsplit(final).hostname or ""
    store = any(host == s or host.endswith("." + s) for s in STORES)
    if status >= 400:
        issues.append(f"landing page trả HTTP {status}")
    if "attachment" in dispo or ctype in DOWNLOAD_TYPES:
        issues.append("landing page tự tải file về máy (không được phép)")
    page = body.decode("utf-8", "ignore") if ctype.startswith("text/") or not ctype else ""
    title = html.unescape((re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I) or [None, ""])[1]).strip()[:200]
    text = _text(page)
    if page and not store:
        if not re.search(r'<meta[^>]+name=["\']viewport["\']', page, re.I):
            issues.append("không có meta viewport: trang có thể không thân thiện với điện thoại")
        if len(text) < 300:
            issues.append("trang gần như không có nội dung")
        if PLACEHOLDER.search(title + " " + text[:2000]):
            issues.append("trang có dấu hiệu đang xây dựng / lỗi (under construction, coming soon, 404)")
        if not PRIVACY.search(page):
            issues.append("không thấy link chính sách quyền riêng tư")
        if not CONTACT.search(page):
            warnings.append("không thấy thông tin liên hệ (bắt buộc với ad thương mại điện tử)")
    if hops > 3:
        warnings.append(f"chuyển hướng {hops} lần trước khi tới trang đích")

    page_info = {"url": url, "final_url": final, "status": status, "store_page": store, "title": title,
                 "ok": not issues, "issues": issues, "warnings": warnings, "consistent": None, "_text": text}
    if client is None:
        return page_info  # giữ _text để gọi check_consistency sau
    try:
        page_info = check_consistency(client, page_info, product, ad_text)
    except Exception as exc:
        log.warning("landing consistency check failed: %s", exc)
    page_info.pop("_text", None)
    return page_info
