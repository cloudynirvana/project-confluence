# Citation policy (CONFLUENCE public surfaces)

Research / in-silico only. This is not a medical device, not clinical
decision support, not personalized medicine as clinical CDS, and not a
claim of cure, diagnosis, or dosing. See [DISCLAIMER.md](../DISCLAIMER.md).

## Rules

1. **Every public claim needs a ledger id or a numbered reference.**
   Thesis HTML uses Vancouver-style [n] matching the reference list, plus
   `EVID-xxx` keys in `evidence/claims.json`. The findings chapter uses
   numbered Vancouver citations [1] matching its reference list.
2. **Do not fabricate DOIs.** Include a DOI only when it is verified
   (publisher, Crossref, or the issuing agency). Prefer a stable URL
   (PMC, WHO, IARC, NCI, GitHub) when no DOI is on hand. Thesis #1
   Vancouver form, DOI verification, and software-citation pattern:
   [CITATION_STYLE.md](CITATION_STYLE.md).
3. **OnCo attribution.** OnCo data is CC BY-NC 4.0. Any OnCo-derived
   sentence keeps the line: *Data from OnCo (onco.cc), CC BY-NC 4.0;
   commercial use needs a licence.* Adapter code in this repository is MIT;
   cached OnCo payloads are not.
4. **No clinical outcome claims.** Simulated trajectories, auditor
   classifications, thinking-lab boards, and disease profiles are
   computational artefacts. They are not patient outcomes, not a protocol,
   and not a dose.
5. **Knowledge is not a parameter.** OnCo pages, `confidence.probability`,
   Idea maturity, and legacy `gene_to_parameter_map.json` values must not
   be cited as identified ODE Θ.

## Where this applies

- `docs/manuscript/ONCO_CONFLUENCE_THESIS_FINDINGS.md`
- `docs/manuscript/thesis_01_confluence_onco.md`
- `evidence/thesis.html`, `evidence/thesis.pdf`, and `evidence/claims.json`
- Disease Profile JSON from the thinking lab
- `data/profiles/cases/` research case pack
- README / HOSTING copy that restates burden statistics
