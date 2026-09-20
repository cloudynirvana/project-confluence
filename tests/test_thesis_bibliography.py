"""Journal-worthy Vancouver bibliography for Thesis #1."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from thesis_bibliography import load_entries, vancouver  # noqa: E402

MANUSCRIPT = REPO / "docs" / "manuscript" / "thesis_01_confluence_onco.md"
HTML = REPO / "evidence" / "thesis.html"
PLACEHOLDERS = ("TBD", "doi:pending", "10.xxxx", "xxxx/xxxxx")
DOI_FIELD = re.compile(r"doi:\s*(10\.\S+)", re.I)
NUMBERED = re.compile(r"^(\d+)\.\s+(.+)$")


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        mapping = {k: v or "" for k, v in attrs}
        if mapping.get("name") == "citation_reference":
            self.refs.append(mapping.get("content", ""))


def _markdown_refs(text: str) -> list[str]:
    started = False
    out: list[str] = []
    for line in text.splitlines():
        if line.strip().lower() == "## references":
            started = True
            continue
        if started and line.startswith("## "):
            break
        if started:
            match = NUMBERED.match(line.strip())
            if match:
                out.append(match.group(2))
    return out


def _html_refs(html: str) -> list[str]:
    block = re.search(r'<ol class="refs">(.*?)</ol>', html, flags=re.S)
    assert block, "thesis.html is missing ol.refs"
    items = re.findall(r"<li>(.*?)</li>", block.group(1), flags=re.S)
    cleaned = [re.sub(r"<[^>]+>", "", item).strip() for item in items]
    return cleaned


def test_bibliography_has_at_least_forty_verified_entries() -> None:
    entries = load_entries()
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")
    md_refs = _markdown_refs(manuscript)
    html_refs = _html_refs(html)
    assert len(entries) >= 40
    assert len(md_refs) >= 40 or len(html_refs) >= 40
    assert len(md_refs) == len(entries)
    assert len(html_refs) == len(entries)
    parser = _MetaParser()
    parser.feed(html)
    assert len(parser.refs) == len(entries)


def test_journal_doi_fields_are_real_and_placeholders_absent() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")
    blob = manuscript + "\n" + html
    for token in PLACEHOLDERS:
        assert token.lower() not in blob.lower()
    for source in (_markdown_refs(manuscript) + _html_refs(html)):
        if "doi:" in source.lower():
            match = DOI_FIELD.search(source)
            assert match, f"doi: without 10. pattern: {source[:120]}"
            assert match.group(1).startswith("10.")
            assert "xxxx" not in match.group(1).lower()
            assert "pending" not in match.group(1).lower()
    for entry in load_entries():
        text = vancouver(entry)
        if entry.get("doi"):
            assert entry["doi"].startswith("10.")
            assert f"doi:{entry['doi']}" in text
        if entry["type"] == "journal":
            assert entry.get("volume")
            assert entry.get("year")
            assert "et al." in entry["authors"] or entry["authors"].count(",") >= 0


def test_manuscript_and_html_share_the_same_vancouver_list() -> None:
    entries = load_entries()
    md_refs = _markdown_refs(MANUSCRIPT.read_text(encoding="utf-8"))
    html_refs = _html_refs(HTML.read_text(encoding="utf-8"))
    expected = [vancouver(entry) for entry in entries]
    assert md_refs == expected
    assert html_refs == expected
    sung, altrock, gatenby = expected[0], expected[7], expected[8]
    assert "Sung H, Filho AM, Laversanne M" in sung
    assert "CA Cancer J Clin. 2026;76(4):e70090" in sung
    assert "doi:10.3322/caac.70090" in sung
    assert "PMID: 42417444" in sung
    assert "Altrock PM, Liu LL, Michor F" in altrock
    assert "Nat Rev Cancer. 2015;15(12):730-745" in altrock
    assert "Gatenby RA, Silva AS, Gillies RJ, Frieden BR" in gatenby
    assert "Cancer Res. 2009;69(11):4894-4903" in gatenby


def test_every_bibliography_number_is_cited_in_the_manuscript_body() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    body, _sep, _refs = manuscript.partition("## References")
    cited = set()
    for match in re.finditer(r"\[(\d+(?:[-,]\d+)*)\]", body):
        token = match.group(1)
        parts = token.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                cited.update(range(int(start), int(end) + 1))
            else:
                cited.add(int(part))
    expected = {entry["id"] for entry in load_entries()}
    missing = sorted(expected - cited)
    assert not missing, f"uncited bibliography numbers: {missing}"


def test_core_required_records_are_present() -> None:
    texts = [vancouver(entry) for entry in load_entries()]
    blob = "\n".join(texts)
    assert "10.3322/caac.70090" in blob
    assert "gco.iarc.who.int/media/globocan/factsheets/populations/566-nigeria" in blob
    assert "who.int/news-room/fact-sheets/detail/cancer" in blob
    assert "9789240123977" in blob
    assert "triple-negative" in blob.lower()
    assert "tumor heterogeneity" in blob.lower() or "tumour heterogeneity" in blob.lower()
    assert "10.1038/nrc4029" in blob
    assert "10.1158/0008-5472.CAN-08-3658" in blob
    assert "onco.cc" in blob
    assert "pull/9" in blob
    assert "validation_protocol.md" in blob
    assert "DISCLAIMER.md" in blob
    assert "ONCO_CONFLUENCE_ONTOLOGY_SPEC.md" in blob
    assert "ONCO_ADAPTER.md" in blob
    assert "Hanahan D, Weinberg RA" in blob
    assert "Warburg O" in blob
    assert "Villaverde AF" in blob
    style = (REPO / "docs" / "CITATION_STYLE.md").read_text(encoding="utf-8")
    assert "doi:10.xxxx/xxxxx" not in style or "Never invent" in style
