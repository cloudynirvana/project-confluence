# SAEM TNBC Research Protocol

## Objective
Validate coherence engineering approach to TNBC therapy using real metabolomics data.

---

## Phase 1: Synthetic Validation (COMPLETE)

- [x] Build generator extraction pipeline
- [x] Implement coherence metrics
- [x] Create intervention mapping for TNBC
- [x] Demonstrate on synthetic TNBC model
- [x] Identify gaps: ROS, NAD+, mitochondrial axes

**Result:** 6.6% restoration with 2-drug protocol. Need expansion.

---

## Phase 2: Intervention Library Expansion

### Additional Interventions to Implement

1. **High-dose Vitamin C**
   - Mechanism: Pro-oxidant in cancer cells
   - Target: ROS-ROS coupling
   - δA effect: Increase ROS generation, deplete glutathione

2. **NAD+ Precursors (NMN/NR)**
   - Mechanism: Restore mitochondrial NAD+ pool
   - Target: NADH-ATP coupling
   - δA effect: Improve electron transport, sirtuin activation

3. **CoQ10**
   - Mechanism: Electron carrier, antioxidant
   - Target: Mitochondrial complex I-III
   - δA effect: Improve OXPHOS efficiency

4. **Metformin** (already partially mapped)
   - Mechanism: Complex I inhibition
   - Target: OXPHOS suppression, AMPK activation
   - δA effect: Metabolic stress in glycolysis-dependent cells

---

## Phase 3: Real Data Application

### Recommended Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| MTBLS791 | MetaboLights | TNBC cell line metabolomics |
| GSE81002 | GEO | Breast cancer metabolic profiling |
| CCLE | Broad | Cancer Cell Line Encyclopedia |

### Workflow

1. Download time-series or multi-condition metabolomics
2. Normalize and preprocess
3. Extract A using generator.py
4. Compare to healthy controls
5. Compute δA_deficit
6. Map to expanded intervention library
7. Validate predictions against known therapy responses

---

## Phase 4: Multi-Drug Optimization

### Objective
Find optimal 4-5 drug combination that maximizes coherence restoration.

### Method
1. For each intervention i, compute δA_i
2. Solve: min ||A_healthy - (A_cancer + Σ w_i × δA_i)||
3. Subject to: Σ w_i ≤ budget, w_i ≥ 0
4. Output: Optimal weights (dosing ratios)

### Expected Outcome
- 4-drug protocol achieving >50% coherence restoration
- Specific dosing recommendations
- Synergy/antagonism identification

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Coherence restoration | >50% toward healthy |
| Stability margin | Lyapunov < -0.15 |
| Intervention count | ≤5 drugs |
| Evidence level | ≥2 established, ≤3 emerging |

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Synthetic validation | 1 week | COMPLETE |
| Intervention expansion | 1 week | NEXT |
| Real data application | 2 weeks | PENDING |
| Multi-drug optimization | 1 week | PENDING |
