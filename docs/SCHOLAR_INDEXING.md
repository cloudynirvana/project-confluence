# Google Scholar indexing — Project Confluence thesis

Owner checklist for scholarly discoverability of
[https://confluence-research.vercel.app/thesis](https://confluence-research.vercel.app/thesis).
This is not generic SEO.

**Honesty.** Abstracts, Highwire tags, and the PDF are computational research.
Not a medical device, not clinical decision support, not a cure, not a dose,
not OnCo-as-parameter.

## What the repo ships

- Highwire `citation_*` on `evidence/thesis.html`: title, author **Kelechi Emeka Ogbonna**,
  date `2026/09/20`, technical-report institution, absolute
  `citation_fulltext_html_url` and `citation_pdf_url`.
- One `citation_reference` tag per numbered Vancouver item on the page
  (same list as `docs/manuscript/thesis_01_confluence_onco.md`). Journal DOIs
  are Crossref/PubMed-verified only. Agency and GitHub items are URL citations.
  No invented DOIs. No ORCID (none is recorded in this repo). Style:
  [CITATION_STYLE.md](CITATION_STYLE.md).
- Citeable PDF: `evidence/thesis.pdf` → **https://confluence-research.vercel.app/thesis.pdf**
  (same site-root directory as `/thesis`, as Scholar requires). Built from
  `docs/manuscript/thesis_01_confluence_onco.md`.
- `evidence/robots.txt` allows `/thesis` and `/thesis.pdf` for Googlebot and
  Googlebot-Scholar.
- `evidence/sitemap.xml` lists the thesis HTML and the PDF only. `/thinking`
  is not a Scholar article URL.

No `citation_doi` / `citation_arxiv_id` for this chapter.
`CITATION.cff` Zenodo `10.5281/zenodo.21446803` is the **software** record.
Do not copy it onto the thesis.

## Google Search Console (owner)

1. Open [Google Search Console](https://search.google.com/search-console).
2. Add the URL-prefix property `https://confluence-research.vercel.app/`
   (or the domain property if you already own the DNS).
3. Verify with the Vercel / DNS method you already use.
4. **Sitemaps → Add** `https://confluence-research.vercel.app/sitemap.xml`.
5. **URL inspection** → inspect
   `https://confluence-research.vercel.app/thesis` and
   `https://confluence-research.vercel.app/thesis.pdf` → Request indexing.
6. Confirm `robots.txt` is fetched (GSC robots tester). Preview deployments
   often send `x-robots-tag: noindex`; production must not.

GSC helps Google Search. Google Scholar has its own crawler and delay.

## Google Scholar author profile

1. [scholar.google.com](https://scholar.google.com/) → My profile.
2. Name **exactly**: Kelechi Emeka Ogbonna.
3. Homepage: `https://confluence-research.vercel.app/thesis` and
   `https://github.com/cloudynirvana`.
4. After the paper appears, confirm it. Do not write clinical, dosing, or CDS
   language in the profile or abstract fields.

First coverage is often **days to several weeks**. Updates to an already-indexed
paper commonly take **6–9 months**.

## Later: Zenodo / preprint DOI (optional)

No DOI is minted for this chapter now.

1. Deposit `evidence/thesis.pdf` on [Zenodo](https://zenodo.org/) or a preprint server.
2. Keep the research-only abstract. No cure / CDS / dosing claims.
3. Only then add `citation_doi` (and `citation_arxiv_id` only if an arXiv id exists).
4. Do not invent identifiers. Do not reuse the software Zenodo DOI unless that
   record is explicitly this PDF.

## Regeneration

```bash
python scripts/sync_thesis_references.py
python scripts/build_thesis_pdf.py
```

Bibliography source: `docs/manuscript/thesis_01_bibliography.json`. Style:
[CITATION_STYLE.md](CITATION_STYLE.md). Requires `reportlab` for the PDF. CI
asserts the committed PDF, Highwire tags, and a ≥40-item Vancouver list.
