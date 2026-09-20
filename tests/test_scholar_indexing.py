"""Google Scholar surface: Highwire tags, sitemap, citeable PDF."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "evidence"
ORIGIN = "https://confluence-research.vercel.app"
PDF_URL = ORIGIN + "/thesis.pdf"
HTML_URL = ORIGIN + "/thesis"
AUTHOR = "Kelechi Emeka Ogbonna"
TITLE = (
    "CONFLUENCE × OnCo: An Evidence-Gated Dynamical Framework for Integrating "
    "Oncology Knowledge Graphs with Adaptive Cancer-State Models"
)
SUNG_DOI = "10.3322/caac.70090"
FORBIDDEN_DOIS = (
    "10.1158/0008-5472.CAN-08-3658",
    "10.5281/zenodo",
    "10.2471/",
)


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        mapping = {k: v or "" for k, v in attrs}
        name = mapping.get("name", "")
        content = mapping.get("content", "")
        if name.startswith("citation_"):
            self.tags.append((name, content))


def _citation_map(html: str) -> dict[str, list[str]]:
    parser = _MetaParser()
    parser.feed(html)
    out: dict[str, list[str]] = {}
    for name, content in parser.tags:
        out.setdefault(name, []).append(content)
    return out


def test_thesis_html_has_scholar_citation_tags() -> None:
    html = (EVIDENCE / "thesis.html").read_text(encoding="utf-8")
    tags = _citation_map(html)
    assert "citation_title" in tags
    assert tags["citation_title"] == [TITLE]
    assert tags["citation_author"] == [AUTHOR]
    assert tags["citation_publication_date"] == ["2026/09/20"]
    assert tags["citation_pdf_url"] == [PDF_URL]
    assert tags["citation_fulltext_html_url"] == [HTML_URL]
    assert "/thinking" not in tags["citation_pdf_url"][0]
    assert tags["citation_technical_report_institution"] == [
        "Project Confluence / GitHub cloudynirvana"
    ]
    assert "citation_doi" not in tags
    assert "citation_arxiv_id" not in tags
    refs = tags.get("citation_reference", [])
    assert len(refs) == 12
    blob = "\n".join(refs)
    assert "Global cancer statistics 2024" in blob
    assert SUNG_DOI in blob
    assert blob.count("citation_doi=") == 1
    for banned in FORBIDDEN_DOIS:
        assert banned not in blob
    assert AUTHOR in html
    assert '<p class="byline citation_author">Kelechi Emeka Ogbonna</p>' in html
    lowered = html.lower()
    assert "not a medical device" in lowered
    assert "not a clinical decision-support system" in lowered
    assert "cc by-nc" in lowered
    assert "not a confluence parameter" in lowered
    listed = re.findall(r'<ol class="refs">(.*?)</ol>', html, flags=re.S)
    assert listed and listed[0].count("<li>") == 12


def test_sitemap_lists_thesis_html_and_pdf() -> None:
    sitemap = (EVIDENCE / "sitemap.xml").read_text(encoding="utf-8")
    assert HTML_URL in sitemap
    assert PDF_URL in sitemap
    assert "/thinking" not in sitemap
    robots = (EVIDENCE / "robots.txt").read_text(encoding="utf-8")
    assert "Allow: /thesis" in robots
    assert "Allow: /thesis.pdf" in robots
    assert "Googlebot" in robots
    assert "Googlebot-Scholar" in robots
    assert "Disallow: /" not in re.sub(r"#.*", "", robots)
    assert ORIGIN + "/sitemap.xml" in robots


def test_citeable_pdf_is_present_and_honest() -> None:
    pdf_path = EVIDENCE / "thesis.pdf"
    assert pdf_path.is_file()
    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF")
    assert 8_000 < len(data) < 5 * 1024 * 1024
    haystack = data.replace(b"\x00", b" ")
    assert AUTHOR.encode("utf-8") in haystack or AUTHOR.encode("latin-1") in haystack
    lowered = haystack.lower()
    assert b"not a medical device" in lowered
    manuscript = (REPO / "docs" / "manuscript" / "thesis_01_confluence_onco.md").read_text(
        encoding="utf-8"
    )
    assert TITLE in manuscript
    for heading in (
        "## Abstract",
        "## Keywords",
        "## Introduction",
        "## Specific aims",
        "## Background",
        "## Methods",
        "## Results / architectural findings",
        "## Discussion",
        "## Limitations",
        "## Future work",
        "## References",
        "## Disclaimer",
    ):
        assert heading in manuscript
    assert "refuse_knowledge_as_parameter" in manuscript
    assert "not a medical device" in manuscript.lower()
    vercel = (EVIDENCE / "vercel.json").read_text(encoding="utf-8")
    assert "/thesis.pdf" in vercel
    assert '"/thesis"' in vercel
    assert (REPO / "docs" / "SCHOLAR_INDEXING.md").is_file()
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert "SCHOLAR_INDEXING.md" in readme
    gsc = (REPO / "docs" / "SCHOLAR_INDEXING.md").read_text(encoding="utf-8")
    assert "Search Console" in gsc
    assert "Zenodo" in gsc
