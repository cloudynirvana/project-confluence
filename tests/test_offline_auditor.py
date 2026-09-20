"""Contract tests for the same-origin Grok auditor without an xAI key."""

from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def test_grok_review_soft_fails_when_key_missing():
    src = (EVIDENCE / "api" / "grok-review.js").read_text(encoding="utf-8")
    assert "XAI_API_KEY" in src
    assert 'reason: "auditor_offline"' in src
    assert "ok: false" in src
    assert "Evidence auditor offline (no API key)" in src
    assert "citations still load from ledger" in src


def test_thesis_page_shows_offline_banner_not_a_dead_button():
    html = (EVIDENCE / "thesis.html").read_text(encoding="utf-8")
    js = (EVIDENCE / "thesis.js").read_text(encoding="utf-8")
    banner = "Evidence auditor offline (no API key) — citations still load from ledger"
    assert banner in html
    assert banner in js
    assert "auditor-offline" in html
    assert "setAuditorOffline" in js
    assert "probeAuditor" in js
    assert "[1,2]" in html
    assert "EVID-001" in html
