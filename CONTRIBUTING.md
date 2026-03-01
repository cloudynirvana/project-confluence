# Contributing to Project Confluence

Thank you for your interest in this project! We especially welcome **expert scientific review** from researchers in computational biology, cancer biology, pharmacology, and mathematical modeling.

## 🔬 How to Provide Expert Review

The most valuable contributions are critiques and feedback. Here's what we're looking for:

### High-Priority Review Areas

| Area | What to Critique | Key Files |
|------|-----------------|-----------|
| **Biological plausibility** | Are the 10×10 generator matrices reasonable representations of cancer metabolic states? | `src/tnbc_ode.py` |
| **Drug modeling** | Do the `expected_effect` matrices reasonably approximate drug mechanisms? | `src/intervention.py` |
| **Immune dynamics** | Is the lymphocyte force field model biologically defensible? | `src/immune_dynamics.py` |
| **Resistance model** | Is the exponential resistance decay (τ=18d) realistic? | `resistance_model.py` |
| **Clinical translatability** | Could the generated protocols be meaningfully tested in vitro? | `results/*_lab_protocol.md` |
| **Mathematical framework** | Is the Kramers escape theory application valid for this domain? | `src/geometric_optimization.py` |

### What Good Feedback Looks Like

- "This assumption in `immune_dynamics.py` line 45 contradicts [published_paper]"
- "The DCA mechanism matrix overestimates pyruvate shunting — real EC50 is X, not Y"
- "The resistance model should include [mechanism] — see [citation]"
- "This protocol wouldn't be feasible in vitro because [reason]"

### What We're NOT Looking For (Yet)

- UI/UX improvements
- Performance optimization
- Refactoring for code style

## 🧪 How to Reproduce Results

```bash
# 1. Install dependencies
pip install numpy scipy scikit-learn matplotlib

# 2. Run the full pan-cancer pipeline
python confluence_runner.py --all --lab-protocols

# 3. Run tests (including adversarial stress tests)
python -m pytest tests/ -v

# 4. Results will be in results/
```

## 📋 Submitting Feedback

### Option 1: GitHub Issues (Preferred)

Open an issue with one of these tags:
- `[biology]` — Biological plausibility concerns
- `[math]` — Mathematical model critique
- `[pharmacology]` — Drug library / PK model feedback
- `[clinical]` — Clinical translatability comments
- `[bug]` — Model producing incorrect behavior

### Option 2: Pull Requests

If you want to contribute code:
1. Fork the repository
2. Create a feature branch (`git checkout -b fix/resistance-model-tau`)
3. Make your changes with clear comments explaining the biological rationale
4. Add or update tests
5. Submit a PR with references to supporting literature

## 📚 Key References

Before reviewing, you may want to familiarize yourself with:
- Kramers escape theory (Kramers 1940, Physica)
- Adaptive therapy (Gatenby et al. 2009, Cancer Research)
- Cancer metabolomics (Vander Heiden et al. 2009, Science)
- The Flatten→Heat→Push geometric protocol (see `README.md`)

## Code of Conduct

Be constructive. The goal is to improve the science. All feedback should be evidence-based and referenced where possible.
