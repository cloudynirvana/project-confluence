# Awaiting external clinical validation

**Gate:** the computational / in-silico work on PR #2 is **feature-complete for the current research scope**. The next step is **outside this repository**.

This file is the stop line. The simulator does **not** validate itself as a cure, a Phase II result, or FDA/EMA readiness.

## What is complete here (research)

- Mechanistic closed loop: observations → controller A/B/E/F → unitless `U` → PK/PD → `CancerODE` (LSODA)
- Disease taxonomy + immune/antibody readiness (state signatures, not TNM)
- In-silico endpoint mapping (RECIST-like, H-band CTCAE-like, KM / log-rank / Cox) with computed statistics
- Scientific visualization: flybody `fruitfly.xml` + Blender sidecar / HUD path
- Honesty package: [DISCLAIMER.md](../DISCLAIMER.md), [MERGE_READINESS.md](MERGE_READINESS.md)

## What happens next (external)

Partners and reviewers — not this git repo — own:

1. **IRB / ethics** review for any human-data protocol
2. **Wet-lab / preclinical** models (in vitro, organoid, in vivo) that can falsify the ODE
3. **Clinical trials** (Phase I → III) if, and only if, independent evidence supports that path
4. **Regulatory** submissions (FDA, EMA, or equivalent) if a product is ever proposed

None of those artifacts are produced by `pytest` or by a virtual cohort.

## What reviewers should read

| Artifact | Why |
|----------|-----|
| [DISCLAIMER.md](../DISCLAIMER.md) | Non-claims |
| [MERGE_READINESS.md](MERGE_READINESS.md) | Scope of the research package |
| `python3 -m confluence.benchmarks.validation_suite` | Taxonomy / readiness falsifiable scores |
| `python3 -m confluence.benchmarks.closed_loop_translation` | Translation layer + 4-panel (in silico) |
| `docs/demo/blender/` | Data-driven viz, not generative biology |
| `results/validation_translation/IN_SILICO_ENDPOINT_MAPPING.txt` | Plain-text honesty + computed stats |

## What this repo will not claim while awaiting that gate

- Cure, disease eradication, or a treatment path
- FDA / EMA readiness or a cleared medical device
- A real Phase I / II / III result
- That a log-rank *p* or Cox HR on a virtual cohort is a clinical effect
- That F-256 (or 166k) “thinks,” synthesizes biologics, or doses patients
- That RECIST/CTCAE-like labels are adjudicated imaging or toxicity

Until an external lab, IRB, and trial apparatus exist, Confluence remains a **research codebase**.
