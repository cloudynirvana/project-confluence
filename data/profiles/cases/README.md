# Disease Profile case pack

In-silico research artefacts only. Not a medical device, not clinical CDS,
not personalized medicine as CDS, not a dose, and not a cure. Public
summaries stay Scholar-safe: cited, research-only, no clinical-outcome
claims. See `SUMMARY.md` and DISCLAIMER.md. A separate change on `main`
owns citation_* meta tags and a citeable PDF.

Rebuild:

```bash
python3 scripts/build_disease_profile_pack.py
python3 scripts/build_disease_profile_pack.py --dry-run
```

Source table: `cases.yaml`. JSON files are gated exports. OnCo LDHA /
confidence.probability / Idea maturity never become admitted Θ.
