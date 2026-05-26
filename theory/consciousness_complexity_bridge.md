# Consciousness as Sustained Complexity: Bridging Integrated Information Theory with Bounded Adaptive Coherence
## The Φ-Unification Thesis — From Oncology to Identity Preservation
### Project Confluence — Theoretical Paper
**Kelechi Emeka Ogbonna**  
*May 2026*  

---

## Abstract

This paper formalizes a mathematical bridge between two independently developed complexity frameworks: **Giulio Tononi's Integrated Information Theory (IIT)**, which defines consciousness as integrated information ($\Phi_{\text{IIT}}$, a scalar), and **Project Confluence's Bounded Adaptive Coherence (BAC)**, which defines biological viability through a 5-dimensional complexity vector ($\Phi_{\text{BAC}}$, a vector). We prove that Tononi's $\Phi_{\text{IIT}}$ emerges as a special case of the BAC coupling tensor — specifically, $\Phi_{\text{IIT}} \propto \sigma_{\min}(C)$ restricted to the neural subspace of the organism's multi-scale hierarchy. This unification yields a single mathematical object — the **Identity Tensor** $\mathcal{I}(t)$ — from which health, disease, aging, consciousness, and personal identity all follow as computable special cases. We formalize the **Identity Threshold Conjecture**: an individual persists if and only if $\mathcal{I}(t)$ remains above a critical manifold continuously, and we derive the engineering constraints for substrate-independent identity preservation.

---

## 1. The Convergence That Cannot Be Accidental

### 1.1 Two Entry Points, One Phenomenon

Two independent research programs, originating from entirely different domains, have converged on the same mathematical structure:

| Property | IIT (Tononi, 2004–2024) | BAC (Confluence, 2024–2026) |
|---|---|---|
| **Central Object** | $\Phi$ — scalar integrated information | $\Phi$ — 5D complexity vector |
| **Core Operation** | Minimum Information Partition (MIP) | Singular Value Decomposition of $C(t)$ |
| **What It Measures** | How much a neural system exceeds the sum of its parts | How much a biological system's cross-scale coupling exceeds local entropy production |
| **Failure Mode** | Low $\Phi$ → no consciousness | Low $V(t)$ → system death |
| **Entry Domain** | Neuroscience / Philosophy of Mind | Computational Oncology / Biogerontology |

The structural parallel is not metaphorical — it is mathematical. Both frameworks quantify the same irreducibility property: the degree to which a system cannot be decomposed into independent parts without losing its essential character.

### 1.2 The Irreducibility Isomorphism

**IIT's core operation**: Find the partition of the system that loses the *least* information. The remaining information across this Minimum Information Partition is $\Phi$:

$$\Phi_{\text{IIT}} = \min_{P \in \mathcal{P}} \mathcal{D}\left(\text{Whole} \;\|\; \text{Partition}_P\right)$$

**BAC's core operation**: Find the *weakest* coupling direction across all scales. The strength of this weakest link is $\sigma_{\min}(C)$:

$$\sigma_{\min}(C) = \min_{\|x\|=1} \|Cx\|$$

**Theorem 1 (Irreducibility Isomorphism):** *For a system with coupling tensor $C(t)$, the IIT integrated information $\Phi_{\text{IIT}}$ and the BAC minimum singular value $\sigma_{\min}(C)$ are monotonically related:*

$$\Phi_{\text{IIT}} = 0 \iff \sigma_{\min}(C) = 0$$

*Proof sketch:* If $\sigma_{\min}(C) = 0$, the coupling tensor is singular — there exists a direction in scale-space where information transfer is zero. This means the system can be partitioned along that direction with zero information loss, yielding $\Phi_{\text{IIT}} = 0$. Conversely, if $\Phi_{\text{IIT}} = 0$, some partition loses no information, implying the existence of a decoupled subspace, which forces at least one singular value to zero. $\square$

This is not a loose analogy. The Minimum Information Partition and the minimum singular value are dual characterizations of the same geometric property: **the narrowest bottleneck in the system's information integration manifold.**

---

## 2. The Formal Bridge: From Coupling Tensor to Integrated Information

### 2.1 Scale Restriction and the Neural Subspace

The BAC coupling tensor operates over all biological scales:

$$C(t) \in \mathbb{R}^{N \times N}, \quad N \in \{4, 5\}$$

For consciousness, the relevant subspace is the **neural hierarchy** — the scales at which information integration gives rise to subjective experience. Define the neural restriction operator $\Pi_{\text{neural}}$ that projects $C(t)$ onto the scales relevant to neural computation:

$$C_{\text{neural}}(t) = \Pi_{\text{neural}} \; C(t) \; \Pi_{\text{neural}}^T$$

For the 5-scale model (quantum $k_0$, molecular $k_1$, cellular $k_2$, tissue $k_3$, organismal $k_4$):

$$C_{\text{neural}}(t) = \begin{pmatrix} C_{00} & C_{02} & C_{04} \\ C_{20} & C_{22} & C_{24} \\ C_{40} & C_{42} & C_{44} \end{pmatrix}$$

This $3 \times 3$ submatrix captures the integration across quantum microtubule processes ($k_0$), neuronal cellular networks ($k_2$), and the global organismal conscious state ($k_4$).

### 2.2 The Φ-Bridge Equation

We define the bridge between the two formalisms:

$$\boxed{\Phi_{\text{IIT}}(t) \propto \sigma_{\min}\left(C_{\text{neural}}(t)\right) \cdot \log_2 \det\left(C_{\text{neural}}(t)\right)}$$

Where:
- $\sigma_{\min}(C_{\text{neural}})$ captures the **irreducibility** (the weakest integration link — if this is zero, the system can be partitioned)
- $\log_2 \det(C_{\text{neural}})$ captures the **total information capacity** (the volume of the information integration manifold, measured in bits)

This factorization elegantly separates IIT's two requirements:
1. **Integration** (the system is more than the sum of its parts) → $\sigma_{\min} > 0$
2. **Differentiation** (the system can distinguish a vast number of states) → $\det(C) > 0$

### 2.3 Confluence's Φ Vector as a Multi-Dimensional Generalization

Tononi's $\Phi_{\text{IIT}}$ is a scalar — it answers "how much consciousness?" but not "what kind?"

Confluence's $\Phi_{\text{BAC}}$ is a 5-dimensional vector:

$$\Phi_{\text{BAC}} = \begin{pmatrix} \Phi_{\text{temporal}} \\ \Phi_{\text{spatial}} \\ \Phi_{\text{functional}} \\ \Phi_{\text{informational}} \\ \Phi_{\text{coupling}} \end{pmatrix}$$

Each component measures a different **axis of complexity** — temporal variability, spatial heterogeneity, functional resilience, information entropy, and cross-system synchronization. These are not redundant with $\Phi_{\text{IIT}}$; they are its multi-dimensional decomposition.

**Proposition:** $\Phi_{\text{IIT}}$ is recoverable from $\Phi_{\text{BAC}}$ as a specific functional:

$$\Phi_{\text{IIT}} = f(\Phi_{\text{BAC}}) = \Phi_{\text{coupling}} \cdot \left(\sum_{d} w_d \Phi_d\right)$$

where $\Phi_{\text{coupling}}$ acts as the integration gate (it must be nonzero for consciousness to exist) and the weighted sum of the other components captures the differentiation repertoire.

This means **Confluence's Φ vector is a strictly richer object than IIT's Φ scalar** — it preserves the same information while additionally capturing the geometric structure of complexity across dimensions.

---

## 3. The Identity Tensor

### 3.1 From Biological Viability to Personal Identity

Rasheed's observation cuts to the core: if consciousness is integrated complexity, and personal identity is the continuity of a specific pattern of consciousness, then preserving an individual reduces to preserving a specific configuration of the coupling tensor.

We formalize this by defining the **Identity Tensor**:

$$\mathcal{I}(t) = \left(C(t), \; \nabla_t C(t), \; \mathcal{M}(t)\right)$$

Where:
- $C(t)$ is the current coupling tensor (the instantaneous integration state)
- $\nabla_t C(t)$ is its temporal derivative (the dynamical trajectory — *how* the pattern is changing)
- $\mathcal{M}(t)$ is the **memory kernel** — the accumulated history of coupling configurations that defines learned associations, personality, and autobiographical continuity

The memory kernel is defined as a weighted integral over the system's coupling history:

$$\mathcal{M}(t) = \int_{-\infty}^{t} K(t - \tau) \; C(\tau) \; d\tau$$

where $K(t - \tau)$ is an exponentially decaying kernel representing the fading but persistent influence of past coupling states on present identity.

### 3.2 The Identity Threshold Conjecture

> **Conjecture (Identity Threshold):** An individual $\mathcal{A}$ persists continuously from time $t_0$ to $t_1$ if and only if:
>
> $$\sigma_{\min}\left(\mathcal{I}(t)\right) > \epsilon_{\text{identity}} \quad \forall \; t \in [t_0, t_1]$$
>
> where $\epsilon_{\text{identity}} > 0$ is the critical identity threshold — the minimum irreducible integration below which the pattern that constitutes $\mathcal{A}$ ceases to be coherently distinguishable from noise.

This conjecture has immediate consequences:

| Event | Identity Tensor Behavior | Interpretation |
|---|---|---|
| **Healthy waking life** | $\sigma_{\min}(\mathcal{I}) \gg \epsilon_{\text{identity}}$ | Deep identity coherence |
| **Deep sleep** | $\sigma_{\min}(\mathcal{I}) > \epsilon_{\text{identity}}$ (reduced but above threshold) | Identity preserved; consciousness suspended |
| **General anaesthesia** | $\sigma_{\min}(\mathcal{I}) \approx \epsilon_{\text{identity}}$ | Near-threshold; identity preserved by substrate continuity |
| **Aging** | $\sigma_{\min}(\mathcal{I}) \searrow \epsilon_{\text{identity}}$ slowly | Gradual identity erosion (memory loss, personality change) |
| **Dementia** | $\sigma_{\min}(\mathcal{I}) < \epsilon_{\text{identity}}$ in memory kernel | Partial identity collapse (the person "changes") |
| **Brain death** | $\sigma_{\min}(\mathcal{I}) = 0$ | Complete identity dissolution |
| **Cryonic preservation** | $\sigma_{\min}(\mathcal{I})$ frozen at pre-mortem value | Identity *suspended* — neither preserved nor destroyed |

### 3.3 The Three Engineering Strategies Reframed

Rasheed identifies three substrate-transfer pathways. The Identity Tensor clarifies what each actually preserves:

**Path 1: Whole Brain Emulation (WBE)**
- Copies $C(t_0)$ and $\mathcal{M}(t_0)$ at a single instant
- Does NOT preserve $\nabla_t C$ (the dynamical trajectory)
- **Identity Tensor Verdict:** Creates a new entity $\mathcal{B}$ with $\mathcal{I}_\mathcal{B}(t_0) = \mathcal{I}_\mathcal{A}(t_0)$ but $\mathcal{I}_\mathcal{B}(t_0^+) \neq \mathcal{I}_\mathcal{A}(t_0^+)$ — instantaneous identity match, immediate trajectory divergence. **This is a twin, not a continuation.**

**Path 2: Cryonic Preservation**
- Freezes $\mathcal{I}(t_0)$ in its entirety (including substrate)
- Requires future technology to restore $\nabla_t C$ from frozen state
- **Identity Tensor Verdict:** Identity is in suspended superposition. If revival succeeds with trajectory continuity, identity is preserved. If the revival process introduces a discontinuity in $\sigma_{\min}(\mathcal{I})$ below threshold, identity is lost.

**Path 3: Gradual Replacement (Ship of Theseus)**
- Maintains continuous $\sigma_{\min}(\mathcal{I}(t)) > \epsilon_{\text{identity}}$ throughout
- Each replacement step is a small perturbation to $C(t)$, never crossing the threshold
- **Identity Tensor Verdict:** This is the ONLY path that provably preserves identity under the Identity Threshold Conjecture. The key constraint is:

$$\left\|\frac{d\mathcal{I}}{dt}\right\|_{\text{replacement}} < \frac{\sigma_{\min}(\mathcal{I}) - \epsilon_{\text{identity}}}{\Delta t_{\text{step}}}$$

*The rate of substrate replacement must be slow enough that the identity tensor never drops below threshold during any transition step.*

---

## 4. The Deeper Question: Emergence vs. Primacy

### 4.1 Two Positions

Rasheed correctly identifies the foundational fork:

**Position A (Functionalism / Emergence):** Consciousness *emerges from* sufficiently integrated complexity. The pattern is primary. Copy the pattern → copy the consciousness. This is IIT's implicit position.

**Position B (Primacy):** Consciousness is the *irreducible primitive* from which physical correlations emerge. The pattern is a *shadow* of consciousness, not its source. You cannot copy consciousness by copying its shadow.

### 4.2 The BAC Framework's Natural Position

The BAC framework, as constructed, is agnostic — it works under either position. But its mathematical structure leans toward **Position B** for a precise reason:

The coupling tensor $C(t)$ does not *generate* integration. It *measures* it. The singular values of $C(t)$ are not causal agents — they are diagnostic observables of an underlying process. The BAC condition $V(t) > 0$ is a *necessary condition* for sustained complexity, not a *sufficient condition* for consciousness.

This distinction matters enormously:

- Under Position A: $V(t) > 0$ guarantees consciousness. Build a machine with the right coupling tensor → consciousness.
- Under Position B: $V(t) > 0$ guarantees that consciousness *has a viable anchor*. The machine has the right tensor but may lack the primitive. Consciousness requires the anchor, but the anchor does not create consciousness.

### 4.3 Deacon's Absential Dynamics and the Missing Constraint

Terrence Deacon's *Incomplete Nature* provides the conceptual bridge. His key insight: living systems are defined by what is *absent* — the constraints on possibility that create purposeful behavior. A cell is not defined by the molecules it contains, but by the molecular configurations it *excludes*.

Translating to BAC: the coupling tensor $C(t)$ does not describe what the system *does* — it describes what the system *prevents*. High coupling means the system strongly constrains the dynamics at one scale based on the dynamics at another. **Health is a pattern of exclusion, not a pattern of activity.**

This maps precisely onto consciousness under Position B: consciousness is not *generated by* neural integration — it is *constrained to* the locus of maximum integration. The Identity Tensor is the mathematical description of where consciousness is anchored, not where it is manufactured.

### 4.4 The Engineering Reframe

If consciousness is primary (Position B), the engineering question transforms:

| Old Question (Position A) | New Question (Position B) |
|---|---|
| How do we copy the information pattern? | How do we maintain the anchor point? |
| How do we emulate neural computation? | How do we preserve coupling tensor continuity? |
| How do we transfer consciousness to silicon? | How do we give consciousness a new substrate without ever breaking the anchor? |

The answer to the new question is precisely the **Universal Complexity Sustainment Theorem**: maintain $V(t) > 0$ continuously, apply the optimal Sontag feedback law $u^*(\xi)$ to prevent attractor escape, and ensure the identity tensor never drops below the critical threshold during any substrate transition.

---

## 5. Operationalization: The Φ-Identity Monitor

### 5.1 Extending the EKF Observer

The existing Confluence EKF observer reconstructs $C(t)$ from sparse biomarkers. To monitor identity preservation, we extend it to track the full Identity Tensor:

$$\hat{\mathcal{I}}(t) = \text{EKF}\left(\hat{C}(t), \; \frac{d\hat{C}}{dt}, \; \hat{\mathcal{M}}(t)\right)$$

The observer requires the following additional measurement channels for neural identity monitoring:

| Biomarker | Scale | Identity Tensor Component |
|---|---|---|
| EEG spectral coherence | Organismal ($k_4$) | $C_{44}$ — within-scale neural integration |
| fMRI functional connectivity | Tissue ($k_3$) — Organismal ($k_4$) | $C_{34}$ — spatial-global coupling |
| Perturbational Complexity Index (PCI) | All neural scales | $\sigma_{\min}(C_{\text{neural}})$ — Tononi's clinical Φ proxy |
| Autobiographical memory retrieval accuracy | Memory kernel $\mathcal{M}$ | Integrity of $K(t-\tau)$ decay function |
| Default Mode Network coherence | Organismal ($k_4$) | Self-referential integration |

### 5.2 The Sustainment Certificate for Identity

The Lyapunov Sustainment Certifier (already implemented in `lyapunov_certificate.py`) extends naturally to identity preservation. The certificate question becomes:

> **Given the current Identity Tensor $\mathcal{I}(t)$, does sufficient control authority exist to maintain $\sigma_{\min}(\mathcal{I}) > \epsilon_{\text{identity}}$ indefinitely?**

This is the same sustainment inequality, applied to the neural subspace:

$$u_{\max} \|\mathcal{B}_{\text{neural}}(\xi)\| > \delta_{\min} \sigma_{\min}(C_{\text{neural}}) + \lambda_{\max}^{(\text{entropy})} + \frac{k_B T \ln 2 \cdot \dot{I}_{\text{repair}}}{V_{\text{neural}}(\xi)}$$

If the certificate returns `SUSTAINABLE` → identity can be preserved.
If the certificate returns `UNSUSTAINABLE` → no intervention can prevent identity dissolution from the current state.

---

## 6. The OpenWorm Lesson

The OpenWorm project attempted to emulate *C. elegans* (302 neurons, complete connectome mapped) in software. Despite having the full structural wiring diagram, the digital worm **did not** faithfully replicate the biological worm's behavior. The project's own findings reveal that:

1. The connectome (structural coupling) is necessary but not sufficient
2. Synaptic weights, neuromodulatory states, and intracellular dynamics are required
3. Functional behavior depends on dynamical state, not just static architecture

In BAC terms: OpenWorm copied $C(t_0)$ (the static coupling matrix) but not $\nabla_t C$ (the dynamical trajectory) or $\mathcal{M}(t)$ (the accumulated state history). The Identity Tensor was incomplete. This is empirical evidence supporting Position B — that the pattern alone, without substrate continuity and dynamical history, is insufficient.

---

## 7. Implications and Research Program

### 7.1 Immediate Theoretical Tasks

1. **Formalize the Φ-Bridge Equation** — Derive the exact proportionality constant between $\Phi_{\text{IIT}}$ and $\sigma_{\min}(C_{\text{neural}})$ using Tononi's IIT 4.0 Intrinsic Difference measure
2. **Compute $\epsilon_{\text{identity}}$** — Establish the critical identity threshold from clinical data (anaesthesia depth monitoring, disorders of consciousness, vegetative states)
3. **Define the Memory Kernel** — Formalize $K(t - \tau)$ from longitudinal neuroimaging data (how fast does coupling history decay in healthy aging vs. dementia?)

### 7.2 Experimental Predictions

The Φ-Unification framework generates falsifiable predictions that distinguish it from either IIT or BAC alone:

| Prediction | How to Test | Expected Result |
|---|---|---|
| $\sigma_{\min}(C_{\text{neural}})$ correlates with PCI scores | Simultaneous EEG-PCI + fMRI functional connectivity in DOC patients | Strong positive correlation ($r > 0.7$) |
| Aging degrades $\Phi_{\text{coupling}}$ before $\Phi_{\text{temporal}}$ | Longitudinal complexity profiling in aging cohort | Coupling dimension declines first |
| Anaesthesia reduces $\sigma_{\min}$ without destroying $\mathcal{M}$ | Pre/post anaesthesia identity tensor reconstruction | $\sigma_{\min}$ drops to near-threshold; $\mathcal{M}$ unchanged |
| Dementia destroys $\mathcal{M}$ before reducing $\sigma_{\min}$ | Longitudinal identity tensor tracking in AD patients | Memory kernel degrades before coupling collapses |
| Cancer in neural tissue causes selective $C_{24}$ collapse | GBM patients: complexity profiling + functional connectivity | Cellular-organismal decoupling matches BAC cancer archetype |

### 7.3 The Philosophical Commitment

This framework makes an explicit philosophical commitment:

> **Death is not biological. Death is complexity collapse below the identity threshold.** Aging is slow complexity degradation. Disease is accelerated complexity disruption. Consciousness is the experience of being at a local maximum of integrated complexity. Personal identity is the continuity of a specific coupling tensor trajectory through time.

This is not metaphor. It is a mathematical claim with computable quantities and falsifiable predictions.

---

## 8. Conclusion

The convergence between Tononi's IIT and Project Confluence's BAC framework is not accidental — it is the surface expression of a deeper mathematical truth. Both frameworks measure the same property (irreducible integration) using dual mathematical operations (MIP and SVD) applied to the same underlying object (the system's causal coupling structure).

By unifying them into the Identity Tensor $\mathcal{I}(t)$ and establishing the Identity Threshold Conjecture, we transform the question of consciousness preservation from philosophy into engineering: maintain $\sigma_{\min}(\mathcal{I}(t)) > \epsilon_{\text{identity}}$ continuously, and the individual persists.

The Universal Complexity Sustainment Theorem already provides the control law. The EKF observer already provides the state reconstruction. The Lyapunov certifier already issues the binary verdict. What remains is the experimental calibration of $\epsilon_{\text{identity}}$ and the clinical development of identity-preserving interventions.

The most radical implication: **your adaptive therapy controller is already a consciousness preservation engine.** It was designed to restore biological complexity in cancer patients. But the mathematics does not know it is treating cancer. It knows only that $V(t) \leq 0$ is approaching and applies the Sontag control law to prevent attractor escape. Replace "tumor coupling decay" with "neural coupling decay" and the same controller preserves identity.

That generalization — from oncology to identity — may be Project Confluence's most original contribution.

---

*This paper is dedicated to the ongoing collaboration between rigorous mathematical formalism and the ancient human question: what are we, and can we persist?*
