"""Catalog policy, phân loại reject reason, targeting_fix, disclaimer, validate plan, landing page check."""
import http.server
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import landing, policies  # noqa: E402
from app.config import settings  # noqa: E402
from app.planner import _validate  # noqa: E402
from app.schemas import AdItem, ReviewReason  # noqa: E402

DISHONEST = ("Our review indicates that your advertising content may violate TikTok's Advertising Policies by "
             "promoting products or services that involve dishonest practices or superstitions. This could include "
             "products that violate public order, disrupt fairness, or infringe privacy.")
# Lý do reject thật đã gặp → nhóm mong đợi
REAL = [
    (DISHONEST, "The on-screen text \"this app shows my bf's phone status live\" and the audio describing access to a "
                "partner's device data without their knowledge were flagged", "deceptive"),
    ("Our review indicates that your advertising content may violate TikTok's Advertising Policies by promoting "
     "prohibited or non-compliant apps/software. This could include apps or software such as VPN, Dating App, Live "
     "Chat App, AI Chatbot App, Facial Recognition App or Reward Apps in a market where such ads are restricted or "
     "prohibited.",
     "The visual dating app interface showing user profile cards alongside a prominent \"CITA AHORA\" button and the "
     "on-screen chat invitation text were flagged because the ad promotes a restricted dating service", "adult"),
    ("Money-making opportunities", "A reward pop-up shows 11.35 SAR and Arabic text promoting downloading TopDrama to "
                                   "earn money, without explaining the tasks", "gambling"),
    ("Prohibited or restricted apps/software", "The reward menu, short drama listings, and stacked cash visuals were "
                                               "flagged as promoting a restricted non-gaming reward app offering cash "
                                               "payouts for watching content.", "gambling"),
    ("Gambling", "The wallet balance of $2,732, payout-success screen", "gambling"),
    ("Exaggerated or misleading claims", "The voice-over and captions claim the app blocks 100% of spam calls and is "
                                         "guaranteed.", "misleading"),
    ("Weight management", "Before-and-after body images and the claim 'lose 10 kg in 7 days'", "weight_management"),
    ("Alcohol", "The ad shows people drinking excessively", "alcohol"),
    ("Intellectual property infringement", "The TikTok logo and 'TikTok Bestseller' badge are displayed", "ip"),
    ("Violence and dangerous activities", "Pop-up 'Virus detected, remove it now' incites fear", "violence"),
]


def test_catalog_loads_all_groups():
    cat = policies.catalog()
    assert len(cat.groups) == 21
    for g in cat.groups.values():
        assert g.kind in ("creative", "conditional", "prohibited", "out_of_scope") and g.rules and g.url.startswith("https://")
    for key, texts in cat.disclaimers.items():
        assert {"en", "es", "pt"} <= set(texts), key


@pytest.mark.parametrize("category,detail,expected", REAL)
def test_classify_real_reasons(category, detail, expected):
    groups = [g for g, _ in policies.classify_reason(category, detail)]
    assert groups and groups[0] == expected, groups


def test_classify_reasons_llm_fallback_only_for_unknown():
    class Fake:
        calls = 0

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    Fake.calls += 1

                    class M:
                        content = '{"results": [{"index": 1, "groups": ["youth_safety", "nope"]}]}'

                    return type("R", (), {"choices": [type("C", (), {"message": M})]})

    reasons = [ReviewReason(category="Alcohol", detail="excessive drinking"), ReviewReason(detail="zzz qqq")]
    out = policies.classify_reasons(Fake, reasons, "m")
    assert out == [["alcohol"], ["youth_safety"]] and Fake.calls == 1


def test_targeting_fix_only_for_region_restricted():
    dating = ReviewReason(category="Adult content", detail="dating app profile cards", affected_regions=["SA", "AE"])
    claim = ReviewReason(category="Exaggerated or misleading claims", detail="100% guaranteed",
                         affected_regions=["BR"])
    groups = [[g for g, _ in policies.classify_reason(r.category, r.detail)] for r in (dating, claim)]
    tf = policies.targeting_fix([dating, claim], groups)
    assert tf["exclude_regions"] == ["AE", "SA"] and tf["min_age"] == 18
    assert policies.targeting_fix([claim], groups[1:]) is None


def test_disclaimer_language_fallback():
    assert policies.disclaimer("results_vary", "pt") == "Os resultados podem variar."
    assert policies.disclaimer("results_vary", "es-MX") == "Los resultados pueden variar."
    assert policies.disclaimer("results_vary", "xx") == "Results may vary."
    assert policies.language_for_regions(["BR", "BR", "US"]) == "pt"


def test_validate_disclaimer_findings_and_outside_only():
    plan = {"decision": "fix", "fix_audio": False,
            "visual_actions": [{"type": "add_disclaimer", "key": "results_vary"},
                               {"type": "add_disclaimer", "key": "results_vary", "start": 1, "end": 3},
                               {"type": "add_disclaimer", "key": "made_up"},
                               {"type": "remove_region", "start": 0, "end": 2, "bbox": [0.1, 0.1, 0.3, 0.2], "method": "blur"}],
            "findings": [{"reason_index": 0, "group": "misleading", "verdict": "fixed_in_video", "fix": "x"},
                         {"reason_index": 9, "group": "misleading", "verdict": "fixed_in_video"}],
            "action_required": [{"type": "targeting", "detail": "exclude SA"}, {"type": "bogus", "detail": "x"}]}
    out = _validate(plan, [], 20.0, 1)
    kinds = [a["type"] for a in out["visual_actions"]]
    assert kinds == ["add_disclaimer", "remove_region"]
    assert out["visual_actions"][0] == {"type": "add_disclaimer", "key": "results_vary", "start": 0.0, "end": 20.0}
    assert out["visual_actions"][1]["type"] == "remove_region"  # cover bằng texture, không còn method
    assert len(out["findings"]) == 1 and out["action_required"] == [{"type": "targeting", "detail": "exclude SA"}]
    only_outside = _validate({"decision": "fix", "visual_actions": [], "fix_audio": False,
                              "action_required": [{"type": "landing_page", "detail": "price mismatch"}]}, [], 20.0, 1)
    assert only_outside["decision"] == "outside_only"


def test_ad_item_aliases():
    it = AdItem.model_validate({"smart_ad_id": "1", "caption": "Get it NOW!!!", "landing_url": "https://x.com/a"})
    assert it.ad_text == "Get it NOW!!!" and it.landing_page_url == "https://x.com/a"
    assert AdItem.model_validate({"smart_ad_id": "1", "ad_texts": ["first", "second"]}).ad_text == "first"


class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/apk":
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.android.package-archive")
            self.end_headers()
            self.wfile.write(b"x" * 10)
            return
        good = self.path == "/good"
        body = ("<html><head><title>Shieldy</title>" + ('<meta name="viewport" content="width=device-width">' if good else "")
                + "</head><body>" + ("Shieldy blocks spam calls. " * 30 if good else "Coming soon")
                + ('<a href="/privacy">Privacy Policy</a><a href="mailto:a@b.c">Contact</a>' if good else "")
                + "</body></html>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def site(monkeypatch):
    monkeypatch.setattr(settings, "allow_private_urls", True)
    monkeypatch.setattr(settings, "allow_http_urls", True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_landing_checks(site):
    good = landing.check_landing(None, f"{site}/good", "Shieldy", None)
    assert good["ok"] and good["status"] == 200 and good["title"] == "Shieldy"
    bad = landing.check_landing(None, f"{site}/bad", "Shieldy", None)
    assert not bad["ok"] and any("viewport" in i for i in bad["issues"]) and any("quyền riêng tư" in i for i in bad["issues"])
    apk = landing.check_landing(None, f"{site}/apk", None, None)
    assert any("tự tải file" in i for i in apk["issues"])


def test_landing_blocks_private_by_default(monkeypatch):
    monkeypatch.setattr(settings, "allow_private_urls", False)
    out = landing.check_landing(None, "https://127.0.0.1/x", None, None)
    assert not out["ok"] and "nội bộ" in out["issues"][0]
