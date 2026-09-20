# Vancouver citation style — Thesis #1

This is the bibliographic style for Thesis #1
(`docs/manuscript/thesis_01_confluence_onco.md`, `evidence/thesis.html`,
`evidence/thesis.pdf`). Public-claim honesty rules remain in
[CITATION_POLICY.md](CITATION_POLICY.md).

Numbered Vancouver citations `[n]` match the References list in order. Do not
reuse a number for a different work. Do not leave uncited bibliography entries:
if a source is added, the manuscript body must cite it.

## Journal article

```
Author AA, Author BB, Author CC, et al. Title. Journal Abbreviation. Year;Volume(Issue):pages. doi:10.xxxx/xxxxx. PMID: ####.
```

- List up to six authors, then `et al.`
- Initials without periods (NLM / Vancouver)
- NLM journal abbreviation (`CA Cancer J Clin`, not the full title)
- Include volume, issue, and pages (or the publisher article number, e.g. `e70090`) when Crossref or PubMed supplies them
- `PMID` only when PubMed returns that id for the DOI or title
- Omit any field that is not verified. Never invent volume, issue, pages, DOI, or PMID

Example (Crossref + PubMed, 20 September 2026):

Sung H, Filho AM, Laversanne M, Ferlay J, Siegel RL, Soerjomataram I, et al. Global cancer statistics 2024: GLOBOCAN estimates of incidence and mortality worldwide for 34 cancers in 186 countries. CA Cancer J Clin. 2026;76(4):e70090. doi:10.3322/caac.70090. PMID: 42417444.

## Web / government / technical report

```
Author/Org. Title [Internet]. Place: Publisher; Year Month Day [cited YYYY Mon DD]. Available from: URL
```

Use this form for WHO, IARC, NCI, and other agency pages. If the exact day is
not on the page or in a publisher record, give year (and month if known) and
keep `[cited …]`. Do not invent a report DOI.

## GitHub / software

```
Author. Title [Internet]. Version/commit; Year [cited YYYY Mon DD]. Available from: URL
```

Cite a pull request, protocol, disclaimer, or ontology spec as software /
technical documentation. Do not attach the repository Zenodo DOI
(`10.5281/zenodo.21446803`) to Thesis #1 itself; that record identifies the
software, not this manuscript.

## DOI verification policy

1. A `doi:` field is allowed only after a live lookup on
   [Crossref](https://api.crossref.org/) (`/works/{doi}`) or
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/) (`{doi}[doi]`).
2. **Never invent** a DOI, PMID, volume, issue, or page range. Guessing
   `10.xxxx/xxxxx`, `doi:pending`, or `TBD` is forbidden.
3. If Crossref returns 404, or the record is a different paper than the one
   being cited, drop the DOI and use a complete URL citation instead.
4. PubMed IDs are optional completeness. Absence of a PMID is not a reason to
   invent one.
5. URL-only is required (not a failure) for agency pages, GitHub objects, and
   any journal article whose DOI cannot be verified on the day of writing.

## When URL-only is allowed

- WHO / IARC / NCI / GCO fact sheets and news releases
- GitHub repositories, pull requests, and file blobs
- Software versions and working specifications
- Journal articles with a verified bibliographic skeleton but no Crossref DOI

## Highwire `citation_reference` tags

`evidence/thesis.html` carries one `citation_reference` meta per bibliography
entry (same numbering as the visible list and the manuscript). Include
`citation_doi=` only for Crossref-verified DOIs. Do not add `citation_doi` or
`citation_arxiv_id` for Thesis #1 itself until a document DOI exists.

Author byline and `citation_author` are **Kelechi Emeka Ogbonna**. Do not invent
an ORCID.

## Regeneration

Bibliography source of truth: `docs/manuscript/thesis_01_bibliography.json`.

```bash
python scripts/sync_thesis_references.py
python scripts/build_thesis_pdf.py
```
