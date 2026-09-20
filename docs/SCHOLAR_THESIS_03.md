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
- [x] `citation_reference` — 77 Highwire metas, one per bibliography entry (authors / journal / year / volume / issue / pages / DOI / PMID when they exist)
- [x] human-visible numbered Vancouver list on the landing page (`#ref-1` … `#ref-77`)
- [x] human-visible PDF link (`evidence/papers/…`)

## Sitemap note

`evidence/sitemap.xml` did **not** exist on `main` when Thesis #3 was packaged.
Created `evidence/sitemap-thesis03.xml` instead of inventing a full-site sitemap.
If a general evidence sitemap is added later, merge these four `<url>` entries into it
and point `loc` at the deployed Vercel origin (Root Directory `evidence`).

## Independence / sit on latest main

- [x] Branched / rebased onto `main` at `64ba76b` (PR #11 merged)
- [x] No `CancerODE` / `ode_system.py` edits
- [x] Cites shipped paths: `confluence/profiles/`, `data/profiles/cases/`, `schemas/disease_profile.schema.json`
- [x] Does **not** propose a parallel schema contract
- [x] Distinguishes `confluence.profiles.HypothesisObject` from `confluence.onco.schemas.HypothesisObject`
- [x] CaseCards and P0 cited as repository documents / pull requests — no fabricated outcomes
- [x] Worked examples are qualitative boards that point at pack JSON, not simulated patient benefit
- [x] Disclaimer states: not personalized medicine as a clinical product

## Citation hygiene

- [x] Numbered Vancouver list in the manuscript — **77** references
- [x] In-text `[n]` numbers and bibliography entries are bijective (1–77; no gaps, no orphans)
- [x] Journal form: authors (first six + et al.); year;volume(issue):pages; DOI; PMID when PubMed indexes the work
- [x] **40** journal DOIs verified on Crossref (2026-09-20). None invented.
- [x] Ivy GAP is Puchalski et al., Science 2018;360(6389):660-663, doi:10.1126/science.aaf2666, PMID 29748285 — not the unrelated `aan6814` DOI
- [x] GLOBOCAN 2024 is Sung et al., CA Cancer J Clin. 2026;76(4):e70090, doi:10.3322/caac.70090, PMID 42417444
- [x] Bechhofer 2013 and Bellman 1970 have verified DOIs and no PMID (not PubMed-indexed)
- [x] Internet / repository items use Vancouver electronic form with `[cited 2026 Sep 20]`
- [x] No placeholder DOIs
- [x] OnCo attribution line present
- [x] GLOBOCAN / Nigeria figures keep their estimate year when cited
- [x] Not a clinical CDS / personalized-medicine product claim

## Rebuild PDF

```bash
python3 scripts/render_thesis_03_pdf.py
```

Requires Google Chrome (`google-chrome --headless=new`). The script overwrites both PDF copies.

## Word count

**10,363 words** in `docs/manuscript/thesis_03_disease_profile_method.md` (20 September 2026, Vancouver-complete bibliography, 77 references). Regenerated PDF is 24 pages.

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
