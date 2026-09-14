# Agent Complexity vs Temporal Control
## Project Confluence — Therapeutic Module Note

**Module:** confluence-therapeutic  
**Status:** Research / in-silico — not a medical device, not a dosing protocol  
**Author:** Kelechi Emeka Ogbonna · cloudynirvana  
**Companions:** [`biologics_integration.md`](biologics_integration.md), [`confluence/controllers/gatenby.py`](../confluence/controllers/gatenby.py), [`confluence/controllers/mtd.py`](../confluence/controllers/mtd.py), [`confluence/pharmacology/drug_catalog.json`](../confluence/pharmacology/drug_catalog.json)

---

Neither one alone — and that mismatch is worth sitting with rather than resolving prematurely.

## Biochemical (small molecule) agents

Pharmacokinetically simple, single or dual-node action (e.g., a glycolysis inhibitor hitting HK2 or PFK). They're static — they don't sense cell state, don't distinguish a glycolytic cell from an OXPHOS-shifted one, and can't apply conditional logic. Given the 4–6 dimensional floor from before, a single small molecule is structurally incapable of covering that space; it's a point intervention in a system that needs a controller.

Confluence mapping: the five small-molecule slots in `U` (`anti_pd1`, `tgfb_inhibitor`, `mct1`, `hdac`, `targeted_kinase`) and the fusion TKIs (`tki_imatinib_like`, `tki_alk`) are this class. HK2, PKM2, LDHA, IDH1/2, PDK1, G6PD remain grounded ODE channels, not smart sensors inside the ligand.

## Biologics

Antibodies, bispecifics, engineered proteins — capable of more selective targeting (e.g., binding a surface marker correlated with metabolic phenotype) but still largely open-loop. A bispecific can hit two nodes simultaneously, which gets you closer, but it still can't read the redox/microenvironmental state and adjust its action in real time.

Confluence mapping: Controllers E/F emit protein channels (`protein_anti_pd1`, `protein_tgfb_trap`, `protein_ifng`, `protein_il2`, `protein_chimeric_engager`, `protein_surveillance_igg`, `protein_fusion_mab`). These are simulated infusion / expression *rates*, not ribosomal synthesis and not closed-loop molecules. See [`biologics_integration.md`](biologics_integration.md) for the operator formalism `B_k(Φ, t) = PK_k(t) · A_k · σ_k(Φ)`.

## What actually gets closer

What actually gets closer to the needed complexity is neither category cleanly — it's a synthetic/engineered system with conditional logic:

- Engineered cell therapies (CAR-T or similar) with AND/OR/NOT sensing gates — these can integrate multiple signals (e.g., act only in low-oxygen + high-lactate + specific antigen context) before triggering a response. That's a biologic platform, but the complexity lives in the circuit logic, not the molecule itself.
- Synthetic biology constructs (engineered bacteria, gene circuits) that sense metabolic microenvironment and respond dynamically.

Confluence already approximates a *controller circuit* — not inside a cell product, but as a mushroom-body-style associative net (Controllers E/F, sparse Kenyon-cell expansion, DA-modulated plasticity) that maps noisy `Y` onto `U(t)`. That is research simulation of conditional logic, not a CAR-T manufacturing plan.

## The more important reframe

Given the Monte Carlo result (resistant takeover **1/200 (0.5%)** under adaptive therapy vs **178/200 (89.0%)** under MTD; day-180 horizon, Confluence bake-off), the complexity may not need to live in the *agent* at all — it can live in the **dosing schedule / temporal control loop**, with a comparatively simple agent. Adaptive therapy achieves multi-dimensional responsiveness not by making the drug smarter, but by making the *administration protocol* a closed-loop controller that reads tumor burden and modulates accordingly. That's cheaper to engineer and more clinically tractable than trying to build a single agent that internally encodes 5–6 dimensions of biological logic.

This is Controller B — Gatenby adaptive (Gatenby et al., *Cancer Research* 2009): treat at the current mix until burden drops to 50% of the reference, then holiday; resume when burden recovers to the reference. Controller A is continuous MTD. Same agent mix. Different temporal loop. Opposite resistance geometry.

```
Y.tumor_burden → reference lock → treat / holiday gate → U(t) → PK → X
```

Sensitive clones (`T_s`) are retained as competitors. Resistant clones (`T_r`) are suppressed by niche occupancy rather than by a smarter ligand. Competitive release is a *scheduling* failure, not a molecular one.

## The 4–6 dimensional floor

A point intervention cannot cover this space. The floor, as used by this module:

| Dim | Biology | Confluence channel |
|-----|---------|--------------------|
| 1 | Metabolic phenotype (glycolysis vs OXPHOS) | HK2 / PKM2 / LDHA / PDK1 / G6PD; `G`, `L` |
| 2 | Redox / microenvironmental state | `O`, `L`, ROS-adjacent |
| 3 | Immune competence | `I_act`, `I_exh`, `I_surv`, `A_ready` |
| 4 | Clonal composition | `T_s`, `T_r`, `T_f` |
| 5 | Tumor burden (observable) | `Y.tumor_burden` — the Gatenby reference |
| 6 | Fusion AF / junction neoantigen | noisy `Y` fusion pair |

Coverage of that floor:

| Architecture | Sensing | Actuation | Loop |
|--------------|---------|-----------|------|
| Biochemical (small molecule) | none | 1–2 nodes | open |
| Biologic (mAb / bispecific) | ligand-gated, static | 1–2 nodes | open |
| Gated cell / synbio circuit | AND/OR/NOT on local cues | local payload | closed, expensive |
| Simple agent + adaptive protocol | burden (and optionally clonal proxies) | existing catalog mix | closed, tractable |

## Honest answer

A biochemical agent, wrapped in a biologically-informed adaptive protocol, is more achievable than either a "smart" biologic or a "smart" small molecule alone.

**Yes — this is in service of a Confluence therapeutic module.** Complexity is placed in Controller B (and, later, in the connectome controllers E/F as research-grade circuit logic), not inside a 5–6 dimensional molecule. The catalog stays class-referenced. The loop is the product.

Research simulation only. In-silico resistance / burden / fusion-AF scores are research numbers, not a clinical outcome and not a treatment recommendation. See [DISCLAIMER.md](../DISCLAIMER.md) and [docs/AWAITING_CLINICAL_VALIDATION.md](../docs/AWAITING_CLINICAL_VALIDATION.md).
