"""Vancouver rendering for Thesis #1. Data: docs/manuscript/thesis_01_bibliography.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIB_PATH = REPO / "docs" / "manuscript" / "thesis_01_bibliography.json"
CITED = "2026 Sep 20"
PLACEHOLDERS = ("TBD", "doi:pending", "10.xxxx", "xxxx/xxxxx")


def load_entries() -> list[dict]:
    data = json.loads(BIB_PATH.read_text(encoding="utf-8"))
    entries = data["entries"]
    if len(entries) < 40:
        raise ValueError("bibliography must contain at least 40 entries")
    for i, entry in enumerate(entries, start=1):
        if int(entry["id"]) != i:
            raise ValueError(f"entry id {entry['id']} is not sequential at {i}")
        text = vancouver(entry)
        for token in PLACEHOLDERS:
            if token.lower() in text.lower():
                raise ValueError(f"placeholder {token!r} in ref {i}")
        if "doi:" in text.lower():
            if not re.search(r"doi:10\.\S+", text):
                raise ValueError(f"ref {i} has doi: but no 10. pattern")
    return entries


def vancouver(entry: dict) -> str:
    kind = entry["type"]
    if kind == "journal":
        loc = _journal_location(entry)
        authors = entry["authors"].rstrip(".")
        title = entry["title"].rstrip()
        title_sep = "" if title.endswith(("?", "!")) else "."
        bits = [
            f"{authors}. {title}{title_sep} {entry['journal']}. {loc}.",
        ]
        if entry.get("doi"):
            bits.append(f"doi:{entry['doi']}.")
        if entry.get("pmid"):
            bits.append(f"PMID: {entry['pmid']}.")
        return " ".join(bits)
    if kind in {"internet", "software"}:
        place = entry.get("place")
        publisher = entry.get("publisher")
        imprint = ""
        if place and publisher:
            imprint = f"{place}: {publisher}; "
        elif publisher:
            imprint = f"{publisher}; "
        date = entry.get("date") or str(entry.get("year") or "")
        version = entry.get("version")
        authors = entry["authors"].rstrip(".")
        head = f"{authors}. {entry['title']} [Internet]."
        if version:
            head = f"{head} {version}."
        return (
            f"{head} {imprint}{date} [cited {entry.get('cited', CITED)}]. "
            f"Available from: {entry['url']}"
        )
    raise ValueError(f"unknown bibliography type {kind}")


def _journal_location(entry: dict) -> str:
    year = entry["year"]
    volume = entry.get("volume")
    issue = entry.get("issue")
    pages = entry.get("pages")
    if volume and issue and pages:
        return f"{year};{volume}({issue}):{pages}"
    if volume and pages:
        return f"{year};{volume}:{pages}"
    if volume and issue:
        return f"{year};{volume}({issue})"
    if volume:
        return f"{year};{volume}"
    return str(year)


def highwire(entry: dict) -> str:
    parts = [f"citation_title={entry['title']}"]
    for author in entry.get("hw_authors") or [entry["authors"]]:
        parts.append(f"citation_author={author}")
    date = entry.get("hw_date") or str(entry.get("year") or "")
    if date:
        parts.append(f"citation_publication_date={date}")
    if entry["type"] == "journal":
        parts.append(f"citation_journal_title={entry['journal']}")
        if entry.get("volume"):
            parts.append(f"citation_volume={entry['volume']}")
        if entry.get("issue"):
            parts.append(f"citation_issue={entry['issue']}")
        if entry.get("pages"):
            first = str(entry["pages"]).split("-")[0]
            parts.append(f"citation_firstpage={first}")
        if entry.get("doi"):
            parts.append(f"citation_doi={entry['doi']}")
    else:
        inst = entry.get("publisher") or entry.get("authors")
        parts.append(f"citation_technical_report_institution={inst}")
    return "; ".join(parts)


def html_item(entry: dict) -> str:
    text = vancouver(entry)
    if entry.get("doi"):
        doi = entry["doi"]
        href = f"https://doi.org/{doi}"
        text = text.replace(f"doi:{doi}.", f'<a href="{href}">doi:{doi}</a>.')
    elif entry.get("url"):
        url = entry["url"]
        text = text.replace(url, f'<a href="{url}">{url}</a>')
    if entry["type"] == "journal":
        journal = entry["journal"]
        text = text.replace(f". {journal}.", f". <em>{journal}</em>.", 1)
    return f"<li>{text}</li>"


def numbered_markdown() -> str:
    lines = []
    for entry in load_entries():
        lines.append(f"{entry['id']}. {vancouver(entry)}")
    return "\n".join(lines)
