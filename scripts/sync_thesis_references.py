#!/usr/bin/env python3
"""Sync Thesis #1 Vancouver list into the manuscript and thesis.html."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from thesis_bibliography import html_item, highwire, load_entries, numbered_markdown

MANUSCRIPT = REPO / "docs" / "manuscript" / "thesis_01_confluence_onco.md"
HTML = REPO / "evidence" / "thesis.html"
BIB_START = "<!-- BIBSTART -->"
BIB_END = "<!-- BIBEND -->"


def sync_manuscript() -> None:
    raw = MANUSCRIPT.read_text(encoding="utf-8")
    block = f"{BIB_START}\n{numbered_markdown()}\n{BIB_END}"
    pattern = re.compile(re.escape(BIB_START) + r".*?" + re.escape(BIB_END), re.S)
    if not pattern.search(raw):
        raise SystemExit("manuscript is missing BIBSTART/BIBEND markers")
    MANUSCRIPT.write_text(pattern.sub(block, raw), encoding="utf-8")


def sync_html() -> None:
    raw = HTML.read_text(encoding="utf-8")
    entries = load_entries()
    metas = "\n".join(
        f'  <meta name="citation_reference" content="{highwire(entry)}" />'
        for entry in entries
    )
    raw = re.sub(
        r"(?:  <meta name=\"citation_reference\" content=\".*?\" />\n)+",
        metas + "\n",
        raw,
        count=1,
        flags=re.S,
    )
    items = "\n".join(f"      {html_item(entry)}" for entry in entries)
    raw = re.sub(
        r'(<ol class="refs">).*?(</ol>)',
        r"\1\n" + items + r"\n    \2",
        raw,
        count=1,
        flags=re.S,
    )
    HTML.write_text(raw, encoding="utf-8")


def main() -> None:
    sync_manuscript()
    sync_html()
    n = len(load_entries())
    print(f"synced {n} Vancouver references into manuscript and thesis.html")


if __name__ == "__main__":
    main()
