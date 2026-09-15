# Project Confluence — handoff questionnaire

Filled 2026-09-15 from `main` @ `7732f0628746a98d9d63d5a29f1bae0be98b4acc`.  
Author of this fill: incoming agent, from the repository as it stands plus an independent 2-clone reconstruction that is **not** in this repo.  
Unknowns are labeled **Unknown**. Gaps are left visible on purpose.

This file is a handoff, not a theory change and not a merge of [PR #7](https://github.com/cloudynirvana/project-confluence/pull/7). Research / in-silico only. See [DISCLAIMER.md](../DISCLAIMER.md).

---

## 1. Scope and authority

**What is in scope this cycle?**

- Open, unmerged: [PR #7](https://github.com/cloudynirvana/project-confluence/pull/7) (`theory/agent-complexity-temporal-control`) — therapeutic-module note: complexity belongs in the Gatenby adaptive loop (Controller B), not inside a smarter ligand. Already cites the bake-off as **1/200 (0.5%)** adaptive vs **178/200 (89.0%)** MTD. Do not merge unless the owner asks.
- Related but **out of this repository**: an independent 2-clone Lotka–Volterra + Gatenby reconstruction (Grok Build solver). It is a reconstruction, not the 15-D closed loop and not the bake-off runner.
- Incoming science that *is* in-repo work: reproduce **178/200 vs 1/200** with a pinned runner, seed, and CSV. Keep v1 (`models/`) and v2 (`confluence/`) ODEs distinct — do not collapse them.

**Who may merge?**

Single maintainer: Kelechi Ogbonna ([@cloudynirvana](https://github.com/cloudynirvana)). [docs/MERGE_READINESS.md](MERGE_READINESS.md) is explicit: *“The human owner merges when checks are green. Agents should not merge to `main` unless the owner explicitly asks and CI is green.”*

**What is already on `main`?**

Merged: PR #2 (v2 research package), #5 (deployables), #6 (Vercel clinical briefing).  
Open besides #7: draft [PR #4](https://github.com/cloudynirvana/project-confluence/pull/4) (Neural Therapy Brain, targets `cursor/setup-cloud-env-007a`, not `main`) and draft [PR #3](https://github.com/cloudynirvana/project-confluence/pull/3) (cloud-agent env).  
Open issues: **0**.

**Deadline / reviewer of record for this cycle?**

**Unknown.**

---

## 2. Theory

**Is the core hypothesis settled law, or a working model to stress-test?**

Working model to stress-test. Theory docs (`theory/bounded_adaptive_coherence.md`, `theory/complexity_sustainment.md`, `theory/universal_sustainment_theorem.md`, and companions) state that formal claims require experimental validation. [NOTICE](../NOTICE) calls the repo Phase 1 computational validation.

**Bounded Adaptive Coherence (BAC)**

Viability inequality, as written in `theory/complexity_sustainment.md`:

\[
V(t) = \sigma_{\min}(C(t)) - \max_k[\dot{s}_k(t)] > 0
\]

`models/coupling_tensor.py` implements a coupling tensor on **synthetic ODE trajectories**. It has not been checked on real longitudinal patient data. [CALL_FOR_DATA.md](../CALL_FOR_DATA.md) is still open and still asks for **≥3 timepoints per patient**.

**4–6 dimensional control floor**

PR #7 / `theory/agent_complexity_and_temporal_control.md` (unmerged) maps the floor onto metabolic phenotype, redox/TME, immune competence, clonal composition, observable burden, and fusion-AF. A point ligand cannot cover that space; Controller B is the tractable closed loop. That mapping is a working therapeutic-module claim, not a fitted clinical result.

**MAP / FIM numerical outputs committed?**

Code exists (`models/geometric_pathways.py`, `models/fisher_geometry.py`, `models/network_curvature.py`). [progress_report.md](../progress_report.md) (13 May 2026) lists “run `test_pathways.py` to obtain numerical validation output” as a **next step**. Committed MAP/FIM numbers: **Unknown** (not found as a pinned CSV/report in this pass).

---

## 3. Architecture

**Do not collapse the two ODE stacks.** They are different state spaces with different scientific jobs.

| Stack | File | DIM | State | Job |
|-------|------|-----|-------|-----|
| v2 cancer TME | [`confluence/cancer_env/ode_system.py`](../confluence/cancer_env/ode_system.py) | **15** (`CORE_DIM = 12`) | `T_s, T_r, I_act, I_exh, S_fib, L, O, G, C_tgfb, C_ifng, H, T_f, I_surv, A_ready, awake` + PK | Closed-loop control (A/B/E/F), interactive session |
| v1 Φ / BAC | [`models/ode_system.py`](../models/ode_system.py) | **16** | Metabolic 10 (Glucose…ROS) + immune 3 + microenv 2 + `psi_coherent` | Complexity profiler, BAC tensor, geometric calibration |

README honesty section still describes a **12-D ODE** plus extras in places, and older copy still talks **10+3+2**. `progress_report.md` still diagrams “15D SAEM System” for `ComplexAttractorODE`. The code on `main` is **v2 = 15, v1 = 16**. Treat README/progress_report as stale where they disagree with `DIM`.

**Controllers (v2)**

- **A** — continuous MTD (`confluence/controllers/mtd.py`)
- **B** — Gatenby: treat until burden ≤ **50% of reference**, holiday, resume at **100%** (`confluence/controllers/gatenby.py`; Gatenby et al., *Cancer Research* 2009)
- **E** — plastic mushroom body (interactive default ~256 KC)
- **F** — sparse net; figures are **F-256 proxy** unless 166,700 is requested

**Controllers (v1, different thresholds)**

[`models/adaptive_controller.py`](../models/adaptive_controller.py) `PolicyParams`: `dose_on_threshold = 0.50`, `dose_off_threshold = 0.30` of carrying capacity, plus robust / epigenetic / Landauer constraints. **Not the same loop as v2 Controller B.**

**2-clone ecology (v1)**

[`models/clonal_dynamics.py`](../models/clonal_dynamics.py): Lotka–Volterra S/R. Default `r_S=0.10`, `r_R=0.07`, `α_RS=0.9`, `α_SR=0.6`. Melanoma override: `r_S=0.11`. No phenotypic-switch `eps` in this file.

**Melanoma persister (v2)**

[`confluence/cancer_env/archetypes/melanoma_persister.py`](../confluence/cancer_env/archetypes/melanoma_persister.py): `r_s=0.20`, `r_r=0.13`, `eps0=0.028`, `alpha_rs=0.75`, `alpha_sr=0.45`. Fast HDAC-sensitive persister switch. Fitness-cost gap is narrower than a high-cost clone — the MTD–adaptive gap is expected to shrink here.

**Interactive session vs public face vs reconstruction**

| Surface | What it is | What it is not |
|---------|------------|----------------|
| `python -m confluence` → FastAPI/WebSocket **:8765** | Live research session | Not on Vercel |
| [`clinical/`](../clinical/) Vercel briefing | 60-second clinician read | Not the simulator |
| [`evidence/`](../evidence/) lab reel | Cinematic flybody HUD | Not an interactive session |
| Grok Build 2-clone solver | Independent reconstruction | Not this repo; not the 15-D bake-off |

`mirofish_engine` is a **git submodule** pointer (`160000` @ `985f89f4…`). Whether a clean checkout populates it is **Unknown** without `git submodule update`.

---

## 4. Validation

**Canonical bake-off (README, claimed)**

| Metric | MTD (standard care) | Confluence adaptive |
|--------|---------------------|---------------------|
| Resistant takeover | **178/200 (89.0%)** | **1/200 (0.5%)** |
| Tumor controlled at day 180 | 200/200 (100%) | 36/200 (18.0%) |
| Mean final burden | 0.271 | 0.952 |
| Mean final resistant fraction | 91.5% | **11.9%** |

Near-zero is not exactly zero. **Do not rewrite 0.5% as 0.0%.**

**What is actually in `results/monte_carlo/`**

Only [`monte_carlo_results.png`](../results/monte_carlo/monte_carlo_results.png). **No CSV, no seed, no runner script** in that directory. Which file produced 178/200: **Unknown**. Confidence in 1/200 is **low until a re-run with a pinned seed**.

**Independent reconstruction (not this repo)**

A 2-clone LV + Gatenby + phenotypic switch + hidden-dimension observation noise did **not** recover 0.5% adaptive takeover on GBM at a 5-extra-dim floor (adaptive ~19% vs MTD 100%, n=200, seed 20260913). Melanoma-persister (clonal melanoma rates + `eps0=0.028`) narrowed the gap (~75.5% adaptive at extra-5; ~2.5% at burden-only). Dimension count moved adaptive rates. That is a reconstruction result, not a replacement of the README table.

**Clinical / wet-lab**

[docs/AWAITING_CLINICAL_VALIDATION.md](AWAITING_CLINICAL_VALIDATION.md) is the stop line. No IRB, no wet-lab falsification of the ODE, no trial. [CALL_FOR_DATA.md](../CALL_FOR_DATA.md) timeline (first public dataset April 2026, separate-arm June 2026) is **not evidenced as met** in this pass.

**External clean-checkout reproduction?**

**Unknown.**

---

## 5. Discrepancies (do not paper over)

1. **15-D vs 16-D vs 12-D vs 10+3+2** — three (or four) stories in docs; two real systems in code. Leave both ODEs. Fix copy, do not “unify” the state vectors.
2. **v1 Gatenby 0.50/0.30 of K vs v2 Gatenby 50%/100% of reference burden** — same name, different policy.
3. **Bake-off 0.5%** is a README figure plus a PNG. No seed/CSV. A 2-clone reconstruction did not recover 0.5% on GBM. Until the original runner is pinned, treat 1/200 as **claimed, unreproduced here**.
4. **`progress_report.md` (May 2026)** still says `ComplexAttractorODE` is 15-D SAEM; `models/ode_system.py` is 16-D with `psi_coherent`.
5. **Vercel `confluence-research` / `evidence/`** is a cinematic reel. The interactive session is Python FastAPI. Do not describe the reel as the session.
6. **Infusion `U` is unitless `[0, 1]`**, not mg/kg. CTCAE labels are an H-band surrogate. RECIST-like labels are in-silico.
7. **Controller F figures are F-256** unless 166,700 was actually run.
8. **Two install paths** (`requirements.txt` vs `pip install -e ".[dev]"`) disagree on what “the” environment is. `docs/setup.md` still documents v1 modules (`ComplexityProfiler`, `PatientFitter`, `RADOEngine`) and an i5-3380M / 8 GB laptop. `pyproject.toml` is the v2 package (`confluence==2.0.0`).

---

## 6. Data rights

- License: **MIT** ([LICENSE](../LICENSE)). Copyright 2026 Kelechi Ogbonna.
- Attribution request in [NOTICE](../NOTICE) and [CITATION.cff](../CITATION.cff) (Zenodo DOI `10.5281/zenodo.21446803`, date-released 2026-06-06).
- Optional flybody embodiment: TuragaLab / DeepMind / HHMI Janelia, **Apache 2.0**; not vendored. Cite Vaxenburg et al., *Nature* 643:1312–1320 (2025).
- Nigeria NSTG 2022 guardrails: CC-BY-4.0, Federal Ministry of Health; HuggingFace curation attributed to **Chisom Rutherford** (`chisomrutherford/nigeria-clinical-guidelines-dataset`). NSTG curator beyond that HF dataset: **Unknown**.
- [CALL_FOR_DATA.md](../CALL_FOR_DATA.md) lists contact `kelechi@projectconfluence.org`. Whether that inbox is live: **Unknown**.
- METABRIC / MIMIC (or any committed longitudinal patient cohort meeting ≥3 timepoints): **Unknown** — not found as a rights-cleared, in-repo analysis-ready dataset in this pass. README’s CCLE metabolomics claim is a separate, static cell-line matrix, not a longitudinal patient trajectory.

Do not ingest identifiable patient data into this public MIT repo.

---

## 7. Safety

Not a medical device, not a dosing protocol, not FDA/EMA-ready. Human oversight path is IRB → preclinical → trials → regulator ([DISCLAIMER.md](../DISCLAIMER.md)). Whether DISCLAIMER has had a **clinical/legal review**: **Unknown**.

Safety is **not a single point**. Layered, all research-grade:

| Layer | Where | What |
|-------|-------|------|
| Host death | v2 ODE | `H ≤ 0.2` is terminal (`HOST_DEATH_H`) |
| CTCAE-like JSON | [`validation/clinical_guardrails.json`](../validation/clinical_guardrails.json) | max grade 2; ROS/ATP/lactate/glucose; CYP3A4; organ labs |
| NSTG 2022 JSON | [`validation/nigeria_clinical_guardrails.json`](../validation/nigeria_clinical_guardrails.json) | HIV / malaria / SCD / anaemia; tighter ANC/platelet/Hb; resource-aware drug lists |
| v1 controller | `PolicyParams` + `_apply_constraints` | `robust_max_dose=0.70`, toxicity budget, min holiday 3 d, max continuous 14 d, Landauer thermal override on OSKM |
| Honesty package | DISCLAIMER, AWAITING_CLINICAL_VALIDATION, MERGE_READINESS | no cure / no Phase II / no device claims |

These JSON bounds are **not** a cleared safety case. Do not emit mg/kg regimens from `U`.

---

## 8. Reproducibility

- **No lockfile.** `requirements.txt` is lower-bounded ranges. `pyproject.toml` extras: `dev`, `stats`, `torch`, `flybody`.
- Two documented installs: `pip install -r requirements.txt` ([docs/setup.md](setup.md)) vs `pip install -e ".[dev]"` ([docs/MERGE_READINESS.md](MERGE_READINESS.md), README). They pull different stacks.
- Tests: `pytest -q -m "not slow"` is the advertised gate. `slow` / `flybody` / `CONFLUENCE_FULL_BRAIN=1` are opt-in. Whether CI is green on current `main` at fill time: check Actions; not re-run in this fill.
- Bake-off: **not reproducible from the committed artifact**. Pin runner + seed + CSV before quoting 1/200 as a result.
- Interactive default KC count is 256 for FPS. Do not treat UI numbers as 166,700-neuron runs.
- Hardware note in `docs/setup.md` (i5-3380M, 8 GB, Windows) is a historical laptop profile, not a container spec.

---

## 9. Priorities

Highest-value next science, in order:

1. **Pin the bake-off.** Find or rewrite the Monte Carlo runner. Commit seed, trial count (200), day-180 horizon, takeover definition, CSV, and the PNG. Re-run. If 1/200 does not come back, **document the negative result** — do not silently keep 0.5% as the bake-off.
2. **External clean-checkout.** Fresh clone, `pip install -e ".[dev]"`, `pytest -q -m "not slow"`, short `python -m confluence --benchmark --trials 2 --horizon 40`. Record what fails.
3. **One real longitudinal dataset through Φ / BAC** (≥3 timepoints). CALL_FOR_DATA is still asking; synthetic ODE trajectories are not that gate.

Do **not**, this cycle:

- Merge v1 16-D and v2 15-D into one state.
- Merge PR #7 (or #3, #4) unless the owner asks.
- Claim the Grok 2-clone solver, the Vercel reel, or the FastAPI session as each other.
- Quote 0.0% adaptive takeover as the bake-off.

---

## 10. Communication

- Land work as a **PR against `main` + a short summary**. Do not merge unless asked.
- Escalate **theory changes** (BAC inequality, 4–6-D floor, bake-off numbers) to the owner. Do not “fix” 0.5% to 0.0% or to the reconstruction’s 19%.
- **Document negative results** in the PR body (failed re-runs, missing seed, reconstruction mismatch).
- Public clinician-facing copy stays in `clinical/` and must keep the research disclaimer.
- Contact in CALL_FOR_DATA: `kelechi@projectconfluence.org` — liveness **Unknown**. GitHub identity: [@cloudynirvana](https://github.com/cloudynirvana).

---

## Open unknowns (checklist)

- [ ] Deadline for this cycle
- [ ] External reproduction of `pytest` / bake-off
- [ ] Committed MAP / FIM numerical report
- [ ] Live status of `kelechi@projectconfluence.org`
- [ ] METABRIC / MIMIC (or any rights-cleared ≥3-timepoint cohort) in use
- [ ] Which script wrote 178/200 vs 1/200, and which seed
- [ ] Clinical or legal review of DISCLAIMER.md
- [ ] NSTG curator beyond the HuggingFace attribution
- [ ] Whether `mirofish_engine` submodule is populated on a default clone

---

*Handoff fill only. Not a medical device. Not a merge.*
