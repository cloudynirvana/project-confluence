#!/usr/bin/env python3
"""Render Thesis #3 markdown to a print PDF (research / in-silico only)."""

from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "manuscript" / "thesis_03_disease_profile_method.md"
PDF = REPO / "docs" / "manuscript" / "thesis_03_disease_profile_method.pdf"
MIRROR = REPO / "evidence" / "papers" / "thesis_03_disease_profile_method.pdf"

CSS = """
@page { size: A4; margin: 22mm 20mm 24mm 20mm; }
html, body { background: #fff; color: #1b1a17; }
body {
  font: 11pt/1.45 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
  max-width: 170mm; margin: 0 auto;
}
h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 10pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 8pt; border-bottom: 0.6pt solid #c9c2b4; padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 6pt; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p { margin: 0 0 8pt; orphans: 3; widows: 3; }
strong { font-weight: 700; }
em { font-style: italic; }
a { color: #1b3a4b; text-decoration: none; }
ul, ol { margin: 0 0 8pt 18pt; }
li { margin: 0 0 3pt; }
blockquote {
  margin: 8pt 0 10pt; padding: 6pt 10pt;
  border-left: 2.5pt solid #8a6a2f; background: #f7f3ea;
}
pre, code { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; font-size: 8.6pt; }
pre {
  background: #f4f1ea; border: 0.4pt solid #d9d1c3; padding: 8pt 9pt;
  white-space: pre-wrap; overflow-wrap: anywhere;
}
table { border-collapse: collapse; width: 100%; margin: 0 0 10pt; font-size: 9.4pt; }
th, td { border: 0.4pt solid #c9c2b4; padding: 4pt 5pt; vertical-align: top; }
th { background: #f3eee4; text-align: left; }
hr { border: 0; border-top: 0.6pt solid #c9c2b4; margin: 14pt 0; }
.stamp {
  font: 700 8.5pt/1.3 ui-sans-serif, system-ui, sans-serif;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: #5c3a10; background: #ead7a8; padding: 6pt 8pt; margin: 0 0 14pt;
}
.front p { margin-bottom: 4pt; }
"""


def inline(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        chunk = html.escape(part)
        chunk = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", chunk)
        chunk = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", chunk)
        chunk = re.sub(r"\[([^\]]+)\]\((https?://[^)]+|[^)]+)\)", r'<a href="\2">\1</a>', chunk)
        out.append(chunk)
    return "".join(out)


def render_markdown(md: str) -> str:
    lines = md.splitlines()
    body: list[str] = []
    i = 0
    in_code = False
    code: list[str] = []
    para: list[str] = []
    list_kind = None

    def flush_para() -> None:
        nonlocal para
        if para:
            body.append(f"<p>{inline(' '.join(para))}</p>")
            para = []

    def flush_list() -> None:
        nonlocal list_kind
        if list_kind:
            body.append(f"</{list_kind}>")
            list_kind = None

    while i < len(lines):
        raw = lines[i]
        if raw.startswith("```"):
            flush_para()
            flush_list()
            if in_code:
                body.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(raw)
            i += 1
            continue
        if raw.strip() == "":
            flush_para()
            flush_list()
            i += 1
            continue
        if raw.strip() == "---":
            flush_para()
            flush_list()
            body.append("<hr />")
            i += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", raw)
        if heading:
            flush_para()
            flush_list()
            level = len(heading.group(1))
            body.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if raw.startswith("> "):
            flush_para()
            flush_list()
            quote = [raw[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith("> "):
                quote.append(lines[i][2:])
                i += 1
            body.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue
        if raw.startswith("|"):
            flush_para()
            flush_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            data_rows = [r for r in rows if not re.match(r"^\|\s*[-:| ]+\|$", r.replace("||", "|"))]
            html_rows = []
            for idx, row in enumerate(data_rows):
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                tag = "th" if idx == 0 else "td"
                html_rows.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            body.append("<table>" + "".join(html_rows) + "</table>")
            continue
        ul = re.match(r"^[-*]\s+(.*)$", raw)
        ol = re.match(r"^\d+\.\s+(.*)$", raw)
        if ul or ol:
            flush_para()
            kind = "ul" if ul else "ol"
            if list_kind != kind:
                flush_list()
                list_kind = kind
                body.append(f"<{kind}>")
            body.append(f"<li>{inline((ul or ol).group(1))}</li>")
            i += 1
            continue
        flush_list()
        para.append(raw.strip())
        i += 1
    flush_para()
    flush_list()
    if in_code:
        body.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
    return "\n".join(body)


def build_html(md: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Thesis #3 — Disease Profiles for Complex Pathologies</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="stamp">Research / in-silico only — not a medical device — not personalized medicine as a clinical product</div>
  {render_markdown(md)}
</body>
</html>
"""


def chrome_pdf(html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    user_dir = Path(tempfile.mkdtemp(prefix="thesis03-chrome-"))
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-background-networking",
        f"--user-data-dir={user_dir}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO), timeout=120)


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    html_doc = build_html(md)
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "thesis_03.html"
        html_path.write_text(html_doc, encoding="utf-8")
        chrome_pdf(html_path, PDF)
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    MIRROR.write_bytes(PDF.read_bytes())
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes)")
    print(f"mirrored {MIRROR}")


if __name__ == "__main__":
    main()
