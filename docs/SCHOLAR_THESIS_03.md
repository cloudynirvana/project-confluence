# Scholar checklist — Thesis #3

**Title:** Disease Profiles for Complex Pathologies: A Gated Method for Systemic Personalized-Medicine Research Objects  
**Author:** Kelechi Emeka Ogbonna  
**Date:** 2026-09-20  
**Status:** working method manuscript. Research / in-silico only.

**This thesis is not personalized medicine as a clinical product.**

## Files

| Role | Path |
|---|---|
| Manuscript (source) | `docs/manuscript/thesis_03_disease_profile_method.md` |
| Canonical PDF | `docs/manuscript/thesis_03_disease_profile_method.pdf` |
| Evidence-site PDF copy | `evidence/papers/thesis_03_disease_profile_method.pdf` |
| Highwire landing page | `evidence/thesis-03.html` |
| Sitemap fragment | `evidence/sitemap-thesis03.xml` |
| This checklist | `docs/SCHOLAR_THESIS_03.md` |
| PDF renderer | `scripts/render_thesis_03_pdf.py` |

## Highwire / Google Scholar tags (`evidence/thesis-03.html`)

- [x] `citation_title`
- [x] `citation_author` — Ogbonna, Kelechi Emeka
- [x] `citation_publication_date` — 2026/09/20
- [x] `citation_pdf_url` — raw GitHub URL of the canonical PDF (stable until an evidence-site origin exists)
- [x] `citation_fulltext_html_url`
- [x] `citation_abstract`
- [x] `citation_keywords`
- [x] `citation_technical_report_institution` — Project Confluence
- [x] human-visible PDF link (`evidence/papers/…`)

## Sitemap note

`evidence/sitemap.xml` did **not** exist on `main` when Thesis #3 was packaged.
Created `evidence/sitemap-thesis03.xml` instead of inventing a full-site sitemap.
If a general evidence sitemap is added later, merge these four `<url>` entries into it
and point `loc` at the deployed Vercel origin (Root Directory `evidence`).

## Independence

- [x] No `CancerODE` / `ode_system.py` edits
- [x] Manuscript is self-contained if pull request #11 (exporter) stays dirty
- [x] Schema described as proposed contract `1.0.0`
- [x] CaseCards and P0 cited as repository documents / pull requests — no fabricated outcomes
- [x] Worked examples are qualitative boards, not simulated patient benefit
- [x] Disclaimer states: not personalized medicine as a clinical product

## Citation hygiene

- [x] Numbered Vancouver list in the manuscript
- [x] DOIs only when verified (publisher / PubMed / PMC / Crossref)
- [x] No placeholder DOIs
- [x] OnCo attribution line present
- [x] GLOBOCAN / Nigeria figures keep their estimate year when cited

## Rebuild PDF

```bash
python3 scripts/render_thesis_03_pdf.py
```

Requires Google Chrome (`google-chrome --headless=new`). The script overwrites both PDF copies.

## Word count

**9,554 words** in `docs/manuscript/thesis_03_disease_profile_method.md` (20 September 2026 render). PDF is 27 pages.

Count the markdown source (abstract through disclaimer), not this checklist.

```bash
python3 - <<'PY'
from pathlib import Path
text = Path("docs/manuscript/thesis_03_disease_profile_method.md").read_text(encoding="utf-8")
print(len(text.split()), "words")
PY
```

## Non-claims (do not drop)

Not a medical device. Not CDS. Not a patient chart. Not a dose. Not a cure.
OnCo knowledge is not Θ. Thinking-lab boards are not patient outcomes.
