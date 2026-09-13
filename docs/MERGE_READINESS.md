# Merge readiness — PR #2 (research codebase)

Merging this branch lands a **computational research package**. It does **not** land a medical product, a treatment planner, or an FDA/EMA-ready system. See [DISCLAIMER.md](../DISCLAIMER.md) and [AWAITING_CLINICAL_VALIDATION.md](AWAITING_CLINICAL_VALIDATION.md).

The human owner merges when checks are green. Agents should not merge to `main` unless the owner explicitly asks and CI is green.

## What’s in PR #2

- Installable Confluence v2 package: 15-D cancer ODE (`H` at index 10, `T_f` at 11, readiness / dormancy extras), observation layer, PK/PD catalog
- Controllers **A** (MTD), **B** (Gatenby), **E** (plastic mushroom body), **F** (sparse net; figures are **F-256 proxy** unless 166700 is requested)
- Disease taxonomy state signatures: benign / malignant / occult / dormant / terminal
- Immune / antibody readiness prior and catalog biologics (research PK slots)
- Real closed loop: `CancerODE.rhs` / `step` / `ClosedLoopSimulator`, default **LSODA**
- In-silico endpoint mapping (RECIST-like, H-band CTCAE-like, KM / log-rank / Cox) — not a trial
- TuragaLab/flybody `fruitfly.xml` hero + cinematic / immune demos (fail closed if mesh missing)
- Blender scientific export: MuJoCo PNG sequence + JSON/CSV HUD sidecar + Blender 4.x HUD stills (`docs/demo/blender/renders/`) stamped `SIMULATION / RESEARCH`
- DISCLAIMER and no-cure framing throughout

v1 `models/` Φ / BAC is **kept**; v2 does not replace it.

## Install and test

```bash
pip install -e ".[dev]"
# optional extras
pip install -e ".[stats]"      # lifelines Cox / KM cross-check
pip install -e ".[flybody]"    # then scripts/install_flybody.sh for fruitfly.xml

pytest -q -m "not slow"
# skip 166k jobs unless CONFLUENCE_FULL_BRAIN=1

python3 -m confluence.benchmarks.closed_loop_translation --quick
export MUJOCO_GL=osmesa
python3 -m confluence.demo_blender --out docs/demo/blender --frames 8
```

Interactive session: `python -m confluence` → http://127.0.0.1:8765

## Known limitations / honesty

- Research scores on an ODE. Not a clinical outcome. Not a cure.
- Controller **F** in default figures is **F-256**, not 166,700 neurons
- Infusion `U` is **unitless** `[0, 1]`, not mg/kg
- CTCAE labels are an **H-band surrogate**, not organ-system CTCAE v5.0
- RECIST true CR only if burden ≈ 0; most virtual GBM runs are PD / unconfirmed-PR
- FlyWire graph is a structured stub unless CAVE credentials + real FAFB ingest are added
- Placeholder knobs are tagged `PLACEHOLDER` in `PARAM_PROVENANCE` / drug catalog
- See [DISCLAIMER.md](../DISCLAIMER.md)

## Pre-merge checklist (human owner)

- [ ] `pytest -q -m "not slow"` is green on this branch (or GitHub Actions `pytest` is green). CI runs the v2 research-package files and skips `slow` / `flybody`.
- [ ] [DISCLAIMER.md](../DISCLAIMER.md) is present and linked from the README
- [ ] No cure / FDA-ready / Phase II claims in UI copy or README
- [ ] Blender export dry-run documented: `python3 -m confluence.demo_blender` (sidecar always; PNG dump needs `.[flybody]`); HUD still: `blender --background --python docs/demo/blender/confluence_blender_hud.py -- --root docs/demo/blender`
- [ ] Translation layer titled **in silico endpoint mapping**, not a trial
- [ ] Merge lands a **research codebase**, not a medical product

## How to merge

1. Review PR #2 on GitHub (`cursor/confluence-v2-connectome-ab9f` → `main`).
2. Confirm the Actions `pytest` workflow is green.
3. Merge via the GitHub UI (squash or merge commit — owner’s choice).
4. Do **not** treat merge as clinical validation. Next gate: [AWAITING_CLINICAL_VALIDATION.md](AWAITING_CLINICAL_VALIDATION.md).
