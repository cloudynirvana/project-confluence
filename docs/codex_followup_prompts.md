# Codex Follow-Up Prompts

After each of the 4 Codex prompts completes, use these follow-up questions to extract maximum insight and drive the next iteration.

---

## After Prompt 1: Exhaustive Combination Sweep

### Diagnostic Questions

```
Read results/combination_sweep.json and answer:

1. What are the top 3 drug combinations by robustness × coherence?
   For each, explain WHY this combination works geometrically.

2. Are there any "surprise" drugs that appear in top combos but
   seem counterintuitive? Explain their geometric contribution.

3. Compute the OVERLAP between the top 10 protocols — how many
   share 3+ drugs in common? Is there a "universal core" protocol?

4. For the top protocol, what is the MINIMUM number of drugs
   that achieves >80% of the full protocol's escape rate?
   (Identify the "marginal value" of each additional drug.)

5. Did any drug ALWAYS make combinations worse when added?
   These are potential ANTAGONISTS in the geometric framework.

6. Create a matrix: for every pair of drugs, compute how often
   they co-occur in the top 20 vs bottom 20 combinations.
   Identify the strongest SYNERGY pairs and ANTAGONISM pairs.

7. Save a file results/universal_core_protocol.json with the
   minimum effective combination that works across all analyses.
```

---

## After Prompt 2: Statistical Robustness Deep-Dive

### Diagnostic Questions

```
Read results/robustness_analysis.json and answer:

1. Which drug has the STEEPEST sensitivity curve?
   (i.e., small dose changes cause large robustness changes)
   This is the "precision drug" that needs careful dosing.

2. Which drug has the FLATTEST sensitivity curve?
   (i.e., robust across wide dose ranges)
   This is the "forgiving drug" that's easiest to prescribe.

3. Plot the MINIMUM EFFECTIVE DOSE for each drug. If we had to
   cut costs by 40%, which drugs should we dose-reduce FIRST?

4. At what dose does DCA's effectiveness plateau? Is there a
   dose beyond which adding more DCA gives diminishing returns?

5. Are there any dose combinations where two drugs COMPENSATE
   for each other? (i.e., reducing Drug A can be rescued by
   increasing Drug B)

6. What is the protocol's FAILURE MODE? When it fails in Monte
   Carlo trials, what goes wrong? Is it:
   a) Curvature not sufficiently reduced in Phase 1?
   b) Insufficient noise in Phase 2?
   c) Immune exhaustion in Phase 3?
   Categorize all failures and report the distribution.

7. Generate an "insurance protocol" — the minimum-cost version
   that maintains >70% robustness. Save to results/minimum_viable_protocol.json.
```

---

## After Prompt 3: Comprehensive Test Suite

### Diagnostic Questions

```
Read results/bugs_found.md and the test output. Answer:

1. How many bugs did you find in the source code? For each:
   - Root cause
   - Severity (would it affect simulation results?)
   - Which module was affected?

2. What is the current test coverage by module? List:
   - generator.py: X tests, Y% coverage
   - coherence.py: X tests, Y% coverage
   - (etc.)

3. Are there any NUMERICAL STABILITY issues? Specifically:
   - Does compute_basin_curvature ever return NaN/Inf?
   - Does compute_kramers_escape_rate overflow for extreme inputs?
   - Are there division-by-zero edge cases in coherence.py?

4. Run a FUZZ TEST: generate 100 random 10x10 matrices and run
   every public function. Report any crashes. Save crash cases
   to results/fuzz_failures.json.

5. Write a single "golden test" that freezes the EXACT numerical
   output of the full pipeline (TNBC → optimize → Monte Carlo)
   with a fixed seed. Save the expected values to tests/golden_values.json.
   This prevents future regressions.

6. Is the GeometricOptimizer deterministic given the same input?
   Run it 10 times with identical inputs and verify outputs match.
```

---

## After Prompt 4: Pan-Cancer Generator Templates

### Diagnostic Questions

```
Read results/pan_cancer_analysis.md and answer:

1. RANKING: Order all 6 cancer types by attractor depth
   (curvature). Which has the deepest well? Shallowest?
   Does this match clinical prognosis data?

2. PROTOCOL PORTABILITY: Does the TNBC-optimized Geometric
   Achievement Protocol work on ALL cancer types?
   For each type, report:
   - Success rate (% of Monte Carlo trials achieving cure)
   - If <50% success, what CHANGE is needed?

3. UNIVERSAL vs SPECIFIC: Is there a SINGLE protocol that
   achieves >50% success rate on ALL 6 cancer types?
   If not, what's the minimum number of distinct protocols
   needed to cover all 6?

4. VULNERABILITY MAPPING: For each cancer type, what is its
   PRIMARY geometric vulnerability? (curvature? anisotropy?
   ROS tolerance? immune evasion?)

5. Create a "Pan-Cancer Decision Tree":
   Given a cancer type → recommended Phase 1 drugs →
   recommended Phase 2 timing → recommended Phase 3 agents.
   Save to results/decision_tree.json.

6. RESISTANCE PREDICTION: For each cancer type, simulate what
   happens if resistance_rate doubles (from 0.05 to 0.10).
   Which cancers escape the protocol? Which remain cured?

7. Which cancer type is MOST SIMILAR to TNBC geometrically?
   Which is MOST DIFFERENT? Use Frobenius norm of the
   generator matrix difference.
```

---

## Meta-Analysis Prompt (After All 4 Complete)

```
## Task: Cross-Result Synthesis

Read ALL files in results/:
- combination_sweep.json
- robustness_analysis.json
- bugs_found.md
- pan_cancer_analysis.md
- Any other generated files

Synthesize a comprehensive report:

1. UNIVERSAL CORE: What is the absolute minimum drug combination
   that works acceptably across all cancer types?

2. CLINICAL FEASIBILITY: Given the sensitivity analysis, which
   drugs are most robust to dosing errors? Rank by "clinical
   forgiveness" (flattest sensitivity × widest therapeutic window).

3. PRIORITY RESEARCH: Based on all analyses, what is the single
   most impactful improvement to the SAEM framework?
   Options: better PK modeling, more cancer types, resistance
   dynamics, spatial modeling, clinical data integration.

4. Write a 1-page "executive summary" aimed at a non-technical
   reader explaining what the SAEM framework achieves and what
   the evidence shows. Save to results/executive_summary.md.

5. Create a "gaps and limitations" document listing every
   assumption in the model and rating how likely each is to
   break in clinical reality. Save to results/limitations.md.
```
