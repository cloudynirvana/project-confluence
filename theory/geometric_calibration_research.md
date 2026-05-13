# Geometric Calibration of Complex Disease Therapeutics: Minimum Action Pathways and Information Geometry for Realignment Drug Discovery in a Pan Cancer ODE Framework

**Kelechi Emeka Ogbonna**

Ahmadu Bello University, Zaria, Nigeria

GitHub: github.com/cloudynirvana | Project Confluence

---

## Abstract

This research proposes extending the Project Confluence computational oncology framework with advanced geometric methods to discover optimal therapeutic realignment pathways from pathological attractor states back to healthy physiological basins. Existing work in the framework provides local geometric analysis of cancer attractor wells using eigenvalue decomposition, basin curvature metrics, and Kramers escape rate theory. However, local analysis alone cannot reveal the global structure of the phase space or identify the precise sequence of state transitions that a drug protocol should induce. This proposal introduces three geometric enrichments grounded in current computational biology research: (1) Minimum Action Pathway computation via Freidlin Wentzell large deviation theory, which identifies the most probable transition trajectory between cancer and healthy attractors; (2) Fisher Information Geometry, which defines a Riemannian metric on the space of physiological states and enables computation of geodesic therapeutic distances; and (3) Ollivier Ricci curvature analysis of the underlying gene regulatory network, which identifies structurally fragile edges representing high value drug targets. Each method is reviewed against the current state of the art in computational biology laboratories worldwide, and its practical implementability within the existing 15 dimensional ODE system is critically assessed.

---

## 1. Introduction and Motivation

The conventional paradigm in oncology treats cancer as a localized cellular defect to be eradicated, measuring success primarily through tumor shrinkage. Project Confluence challenges this framework by modeling cancer as a dynamical system that has transitioned from a healthy attractor state to a pathological one (Ogbonna, 2026). The project operationalizes this idea through a fifteen dimensional ordinary differential equation system capturing ten metabolic state variables (glucose, lactate, pyruvate, ATP, NADH, glutamine, glutamate, alpha ketoglutarate, citrate, and reactive oxygen species), three immune state variables (effector cells, regulatory T cells, and exhausted cells), and two microenvironmental state variables (stromal density and vascular index). The system employs Michaelis Menten saturation kinetics to generate the nonlinear strange attractor dynamics observed in real biological systems (Strogatz, 2015; Warburg, 1956).

The framework already includes a geometric optimization module that computes basin curvature, anisotropy, and Kramers escape rates for cancer attractor wells (Kramers, 1940). A three phase therapeutic protocol optimizer (Flatten, Heat, Push) uses these local geometric properties to design interventions. However, these methods provide only a local characterization of the attractor landscape. They measure how deep and narrow the cancer well is, but they do not map the global pathway between the cancer state and the healthy state. They cannot answer the critical therapeutic question: through which sequence of intermediate metabolic and immune states should the system be guided to achieve realignment?

This research proposes to answer that question by introducing three layers of geometric analysis, each drawing from established and practically validated methods in contemporary computational biology. The central thesis is that enhancing our geometric understanding of the disease state space enables the discovery of realignment pathways that are not only theoretically optimal but also translate directly into actionable drug target identification and dosing sequence design.

---

## 2. Review of Current Geometric Methods in Computational Biology

### 2.1 Attractor Landscape Theory and the Waddington Framework

The conceptual foundation for treating disease as an attractor transition derives from Waddington's epigenetic landscape, formalized computationally by Huang and colleagues at the Institute for Systems Biology (Huang et al., 2009; Huang, 2013). In this framework, stable cell phenotypes correspond to attractors in a high dimensional gene expression state space. Cancer cells are understood to be trapped in aberrant attractor states that are typically inaccessible to healthy cells. Huang's recent work investigates the "tipping point" of cancer, the moment of instability where cells transition between states, suggesting therapeutic strategies focused on reprogramming cancer cells back toward normal phenotypes rather than solely attempting elimination (Huang, 2024).

Wang and colleagues at Stony Brook University have advanced this framework through their nonequilibrium landscape flux theory (Wang et al., 2008; Li and Wang, 2014). Their key insight is that cell fate dynamics are governed not only by a quasi potential landscape U (representing global stability) but also by a rotational curl flux that mediates irreversible transitions between multi stable states. In a 2024 study, Wang's group developed novel nonequilibrium early warning indicators based on time irreversibility of cross correlations, entropy production rates, and average flux to predict critical transitions (tipping points) in cell fate decision making. These indicators demonstrated superior predictive capability compared to traditional critical slowing down methods (Xu and Wang, 2024). This work is directly relevant to Project Confluence because it provides a rigorous physical basis for the Unified Complexity Profile's temporal entropy and cross system coupling dimensions.

### 2.2 Adaptive Therapy and Evolutionary Game Theory

The therapeutic philosophy of Project Confluence draws heavily on the adaptive therapy paradigm pioneered by Gatenby, Silva, and colleagues at the Moffitt Cancer Center (Gatenby et al., 2009; Gatenby and Brown, 2020). The Moffitt group models the oncologist as a Stackelberg game leader who anticipates and influences the tumor's evolutionary response, maintaining sensitive cell populations to competitively suppress resistant clones. Long term clinical data from their pilot trial in metastatic castrate resistant prostate cancer (mCRPC) using abiraterone has demonstrated significantly improved time to progression and overall survival, with patients spending approximately half their time off treatment (Zhang et al., 2017). Recent expansion of the Moffitt program includes testing adaptive protocols in breast cancer using gemcitabine and capecitabine, and in ovarian cancer using de escalated PARP inhibitor maintenance (Moffitt Center of Excellence for Evolutionary Therapy, 2024).

Project Confluence's Monte Carlo validation achieving zero resistant takeover across two hundred biologically uncertain scenarios is consistent with these findings and extends the theoretical argument by embedding the adaptive policy within a full complexity restoration framework rather than treating evolutionary containment as an isolated objective (Ogbonna, 2026).

### 2.3 Minimum Action Pathways and Large Deviation Theory

The mathematical formalism for computing optimal transition pathways between attractor states originates from Freidlin Wentzell large deviation theory (Freidlin and Wentzell, 2012). Given a stochastic dynamical system of the form dz = F(z)dt + epsilon dW, where F is the deterministic drift (the ODE right hand side in Project Confluence) and epsilon dW represents biological noise, the most probable transition path between two attractors minimizes the Freidlin Wentzell action functional:

S[phi] = (1/2) integral from 0 to T of the squared norm of (d phi/dt minus F(phi)) dt

The minimizer of this functional, known as the instanton or minimum action path (MAP), represents the path of least resistance through the energy landscape separating the two attractors. Computational methods for finding these paths include the String Method (E et al., 2002; E et al., 2007), which evolves a discretized curve (string of images) iteratively toward the minimum energy path, and the Geometric Minimum Action Method (gMAM) (Heymann and Vanden Eijnden, 2008), which directly minimizes the action functional using a variational approach on the path space.

Recent applications (2024 to 2025) have extended these methods to analyze single cell transcriptomic data, mapping how cancer cells shift identities during epithelial mesenchymal transition (EMT) or when acquiring drug resistance. By identifying the MAP, researchers can pinpoint temporary windows of vulnerability during these transitions that may serve as novel therapeutic targets (Li et al., 2024). This represents a critical shift from targeting stable cancer states to targeting cells while they are in transition, a strategy that the geometric enrichment of Project Confluence is designed to exploit.

### 2.4 Information Geometry and Fisher Information

Information geometry, formalized by Amari (Amari and Nagaoka, 2000), treats the space of probability distributions as a Riemannian manifold with the Fisher Information Matrix (FIM) serving as the natural metric tensor. When applied to biological systems, each physiological state (defined by its probability distribution over gene expression, metabolite concentrations, or ODE state variables) becomes a point on this manifold, and the FIM measures how distinguishable neighboring states are based on observable measurements.

Transtrum and colleagues at Brigham Young University have demonstrated that complex biological models are typically "sloppy," meaning their behavior is controlled by a few stiff parameter combinations while most parameter directions are poorly constrained by data (Transtrum et al., 2015; Transtrum and Qiu, 2014). The FIM defines the geometry of this model manifold, revealing it to have a highly anisotropic "hyper ribbon" structure. Transtrum's Manifold Boundary Approximation Method (MBAM) follows geodesics along the sloppiest directions until the manifold reaches a boundary, at which point the model simplifies into a lower dimensional effective theory (Transtrum et al., 2014). This is directly applicable to Project Confluence's fifteen dimensional ODE system, where identifying which parameter combinations are stiff (and therefore represent high leverage drug targets) versus sloppy (and therefore therapeutically irrelevant) would dramatically focus the drug optimization search space.

In the context of therapeutic distance, the geodesic between the cancer state distribution and the healthy state distribution on the Fisher manifold provides the minimum information theoretic "cost" of transitioning between states. This geodesic distance is invariant to the choice of parameterization, making it a robust measure of therapeutic difficulty that does not depend on arbitrary coordinate choices.

### 2.5 Ricci Curvature on Biological Networks

Discrete Ricci curvature, particularly the Ollivier Ricci curvature (ORC), has emerged as a powerful tool for analyzing the robustness and vulnerability of biological networks (Sandhu et al., 2015; Ni et al., 2015). Applied to gene regulatory or protein interaction networks, ORC assigns a curvature value to each edge: positively curved edges indicate dense, redundant, robust connectivity (functional cooperation), while negatively curved edges represent fragile information bottlenecks (Sandhu et al., 2015). Disrupting negatively curved edges can effectively destabilize disease related network modules.

In 2024, the ORCO (Ollivier Ricci Curvature Omics) tool was released as an open source Python package for computing ORC on biological networks, integrating omics data to analyze system level robustness and identify high risk patient groups or prognostic biomarkers (Pouryahya et al., 2024). Studies using dynamic network curvature analysis have revealed novel therapeutic targets in sarcoma and multiple myeloma. Furthermore, Ricci flow, the process of deforming a network by its curvature, has been applied to understand how biological networks rewire during transitions from healthy states to malignancy, providing geometric insight into drug resistance mechanisms (Sandhu et al., 2015).

---

## 3. Practical Assessment and Alignment with Project Confluence

### 3.1 What is Practically Implementable

The review of current computational biology laboratories reveals that all three proposed geometric enrichments have mature theoretical foundations and functioning software implementations. However, their practical applicability to Project Confluence's specific architecture varies. This section provides an honest assessment.

**Minimum Action Pathways (High Practicality).** This is the most immediately implementable enrichment. Project Confluence already defines the deterministic drift F(z) as the right hand side of its fifteen dimensional ODE system. The healthy attractor is explicitly defined (the `healthy_initial_state()` method returns z0 in R^15), and disease states are computed by integrating the ODE with disease specific parameter sets (TNBCParams, GlioblastomaParams, and others). The boundary conditions for the minimum action path are therefore already available. The String Method can be implemented using SciPy's optimization routines, discretizing the path into 50 to 100 images and iteratively evolving them. The Geometric Minimum Action Method is available via PyGMAM. For a fifteen dimensional system with smooth Michaelis Menten kinetics, the computational cost is manageable on standard hardware. Wang's group routinely applies these methods to gene regulatory networks of comparable dimensionality (Li and Wang, 2014).

**Fisher Information Geometry (Moderate Practicality).** Computing the Fisher Information Matrix for the ODE system requires evaluating the sensitivity of observable outputs to parameter perturbations. For the fifteen dimensional system with approximately forty parameters (as defined in ExtendedParams), this involves computing a 40 by 40 matrix via finite difference Jacobians or adjoint sensitivity methods. SciPy's `solve_ivp` with sensitivity analysis, or libraries such as SALib, can handle this. Transtrum's MBAM has been applied to systems of similar scale (Transtrum et al., 2015). The primary challenge is computational: each entry of the FIM requires a perturbation simulation, yielding on the order of 40 squared (1600) additional integrations. This is feasible but requires careful implementation, potentially using adjoint methods for efficiency.

**Ricci Curvature Analysis (High Practicality for Network Layer).** The existing Project Confluence architecture represents metabolic interactions as a generator matrix A (10 by 10 or 15 by 15), which can be interpreted as a weighted directed graph. Computing Ollivier Ricci curvature on this graph is computationally inexpensive and can be implemented using the NetworkX and GraphRicciCurvature Python packages. The ORCO framework is directly applicable. However, the full power of Ricci curvature analysis is realized when applied to genome scale networks (e.g., TCGA derived gene regulatory networks), which is the scale at which Sandhu and colleagues have demonstrated clinical utility (Sandhu et al., 2015). Integrating this with Project Confluence's bioinformatics miner module (Module 4) would extend the geometric analysis from the ODE state space to the underlying regulatory network.

### 3.2 What Requires Caution

Several aspects of the proposed geometric framework warrant careful attention. First, the application of Kramers escape rate theory and Freidlin Wentzell theory assumes that the noise in the biological system is relatively small compared to the deterministic dynamics (the so called weak noise limit). In real tumors, stochastic fluctuations can be substantial, particularly in small cell populations during treatment. The computed minimum action paths should therefore be interpreted as the most probable trajectories under ideal conditions, not as deterministic prescriptions.

Second, the Fisher Information analysis identifies parametric sensitivities in the mathematical model, which may not perfectly correspond to real druggable targets. A parameter identified as "stiff" (high therapeutic leverage) in the ODE model, such as the glucose uptake rate, must still be mapped to specific molecular interventions (e.g., GLUT1 inhibitors) through the existing gene to parameter mapping in the validation pipeline.

Third, while stochastic resonance theory suggests that noise can be therapeutically useful (amplifying immune detection of tumor cells at optimal noise levels), the clinical application of "resonance therapies" remains controversial and lacks clinical validation (Gammaitoni et al., 1998). The entropic resonance frequencies computed by the current geometric optimization module should be understood as theoretical predictions, not clinical recommendations.

---

## 4. Proposed Geometric Enrichment Architecture

### 4.1 Module Design

The proposed implementation creates a new module, `geometric_pathways.py`, containing three primary classes:

**FreidlinWentzellOptimizer.** This class computes the minimum action path between any two attractor states in the fifteen dimensional phase space. It accepts the ODE right hand side function F(z), a start state (e.g., the TNBC attractor), and an end state (e.g., the healthy attractor), and returns a discretized path in R^15 alongside the action value (total energetic cost of the transition). The implementation uses a simplified string method with reparameterization, adapted from the algorithms described by E, Ren, and Vanden Eijnden (2007).

**FisherManifoldAnalyzer.** This class computes the Fisher Information Matrix for the ODE system's observable outputs with respect to the model parameters, identifies stiff and sloppy parameter combinations via eigenvalue decomposition of the FIM, and computes the geodesic distance between disease and healthy states on the resulting Riemannian manifold. The implementation follows the methodology of Transtrum and colleagues (2014, 2015).

**NetworkCurvatureAnalyzer.** This class constructs a weighted directed graph from the ODE generator matrix and computes Ollivier Ricci curvature for each edge. Edges with strongly negative curvature are flagged as structural bottlenecks representing candidate drug targets. This analysis extends the existing `CoherenceAnalyzer` by adding a geometric network layer.

### 4.2 Integration with Existing Therapeutic Protocol Optimizer

The existing `TherapeuticProtocolOptimizer` in `geometric_optimization.py` implements a three phase Flatten Heat Push protocol. The geometric pathway enrichment modifies this optimizer to accept a precomputed minimum action path as input. When a pathway is provided, Phase 1 (Flatten) selects interventions that not only reduce basin curvature but explicitly push the system's state vector along the tangent direction of the minimum action path at each time step. Phase 2 (Heat) concentrates entropic noise at the frequencies identified by the resonance analysis, specifically targeting the saddle point (highest energy point on the MAP) where the system is most susceptible to noise assisted escape. Phase 3 (Push) applies immune rectifiers aligned with the final descent direction of the MAP into the healthy basin.

### 4.3 Drug Target Identification via Geometric Analysis

The combined geometric analysis produces a ranked list of drug targets through the following convergent evidence framework. First, the minimum action path identifies which state variables undergo the steepest gradients during the transition, highlighting them as variables whose dynamics must be altered for successful realignment. Second, the Fisher Information analysis identifies which model parameters have the highest leverage over the trajectory, distinguishing between stiff (high impact) and sloppy (low impact) parameters. Third, the Ricci curvature analysis identifies which network edges are structurally fragile and therefore most amenable to pharmacological disruption. Targets that appear across all three geometric lenses receive the highest priority ranking. This convergent approach addresses a key limitation of single method target identification, where false positives arise from the particular assumptions of any one mathematical framework.

---

## 5. Validation Strategy

### 5.1 Computational Validation

The minimum action pathway implementation will be validated using a synthetic bistable system (a two dimensional double well potential) for which the analytical MAP is known. The computed path must converge to within numerical tolerance of the known solution. Subsequently, the method will be applied to the full fifteen dimensional ODE system across all ten cancer subtypes currently supported by Project Confluence. The resulting pathways will be compared for biological plausibility: for example, the MAP from TNBC to healthy should show a monotonic decrease in glycolytic flux and a recovery of immune effector function, consistent with known TNBC biology.

### 5.2 Retrospective Validation

The existing TCGA retrospective validation pipeline (Track A) provides survival data for 240 synthetic patients across cancer subtypes. The geometric enrichment will be validated by testing whether the action (total energetic cost) of the minimum action path for each patient's disease state correlates with observed survival time. A higher action value (deeper, more distant attractor) should predict shorter survival. This analysis extends the framework's existing Spearman correlation validation (current rho = negative 0.7937) by adding a geometrically grounded predictor.

### 5.3 Alignment with Regulatory Frameworks

Project Confluence is explicitly aligned with the FDA Model Informed Drug Development (MIDD) guidelines (FDA, 2018). The geometric enrichment maintains this alignment by producing outputs that are interpretable within existing regulatory categories. The minimum action path provides a mechanistic trajectory that can be compared against clinical pharmacokinetic and pharmacodynamic models. The Fisher Information analysis identifies parameter identifiability issues, directly supporting the regulatory requirement for model qualification and uncertainty characterization. The Ricci curvature analysis produces ranked target lists that map to existing drug libraries through the framework's gene to parameter mapping.

---

## 6. Conclusion

This research proposes a principled geometric enrichment of the Project Confluence computational oncology framework, grounded in three mathematically rigorous methods that have been validated by leading computational biology laboratories worldwide. The Freidlin Wentzell minimum action pathway, pioneered in biological applications by Wang's group at Stony Brook and validated through single cell transcriptomic studies, provides the global transition trajectory that local eigenvalue analysis cannot reveal. The Fisher Information Metric, operationalized for biological model reduction by Transtrum's group at BYU, distinguishes therapeutically relevant parameters from noise, focusing the drug discovery search space. The Ollivier Ricci curvature, validated as a prognostic and target identification tool by Sandhu and colleagues and implemented in the 2024 ORCO toolkit, adds a structural network layer that connects ODE dynamics to molecular targets. Together, these methods transform the framework's therapeutic optimization from a problem of local attractor well flattening into a problem of global geometric navigation through the disease state space, enabling the principled discovery of realignment pathways for complex disease.

---

## References

Amari, S. and Nagaoka, H. (2000). *Methods of Information Geometry*. Translations of Mathematical Monographs, Vol. 191. American Mathematical Society.

Cancer Genome Atlas Research Network. (2014). The Cancer Genome Atlas Pan Cancer analysis project. *Nature Genetics*, 45(10), 1113 to 1120.

Costa, M., Goldberger, A.L. and Peng, C.K. (2005). Multiscale entropy analysis of biological signals. *Physical Review E*, 71(2), 021906.

E, W., Ren, W. and Vanden Eijnden, E. (2002). String method for the study of rare events. *Physical Review B*, 66(5), 052301.

E, W., Ren, W. and Vanden Eijnden, E. (2007). Simplified and improved string method for computing the minimum energy path in barrier crossing events. *Journal of Chemical Physics*, 126(16), 164103.

Food and Drug Administration. (2018). *Model Informed Drug Development Pilot Program*. U.S. Department of Health and Human Services.

Freidlin, M.I. and Wentzell, A.D. (2012). *Random Perturbations of Dynamical Systems*. 3rd Edition. Springer.

Gammaitoni, L., Hanggi, P., Jung, P. and Marchesoni, F. (1998). Stochastic resonance. *Reviews of Modern Physics*, 70(1), 223 to 287.

Gatenby, R.A., Silva, A.S., Gillies, R.J. and Frieden, B.R. (2009). Adaptive therapy. *Cancer Research*, 69(11), 4894 to 4903.

Gatenby, R.A. and Brown, J.S. (2020). Integrating evolutionary dynamics into cancer therapy. *Nature Reviews Clinical Oncology*, 17(11), 675 to 686.

Goldberger, A.L., Amaral, L.A.N., Hausdorff, J.M., Ivanov, P.C., Peng, C.K. and Stanley, H.E. (2002). Fractal dynamics in physiology: alterations with disease and aging. *Proceedings of the National Academy of Sciences*, 99(Suppl 1), 2466 to 2472.

Heymann, M. and Vanden Eijnden, E. (2008). The geometric minimum action method: a least action principle on the space of curves. *Communications on Pure and Applied Mathematics*, 61(8), 1052 to 1117.

Huang, S., Ernberg, I. and Kauffman, S. (2009). Cancer attractors: a systems view of tumors from a gene network dynamics and developmental perspective. *Seminars in Cell and Developmental Biology*, 20(7), 869 to 876.

Huang, S. (2013). Genetic and non genetic instability in tumor progression: link between the fitness landscape and the epigenetic landscape of cancer cells. *Cancer and Metastasis Reviews*, 32(3 to 4), 423 to 448.

Huang, S. (2024). Cancer as a tipping point: exploring cell state transitions using attractor landscape theory. Institute for Systems Biology Research Program.

Kembro, J.M., Aon, M.A., Winslow, R.L., O'Rourke, B. and Bhatt, H.N. (2014). Loss of complexity in cancer: multiscale entropy analysis of cardiac interbeat interval dynamics. *Frontiers in Physiology*, 5, 547.

Kramers, H.A. (1940). Brownian motion in a field of force and the diffusion model of chemical reactions. *Physica*, 7(4), 284 to 304.

Li, C. and Wang, J. (2014). Quantifying the landscape for development and cancer from a core cancer regulatory circuit. *Cancer Research*, 75(13), 2607 to 2618.

Li, C., Zhang, L. and Wang, J. (2024). Mapping cancer cell fate transitions using minimum action pathway analysis of single cell omics data. *BioRxiv* preprint.

National Cancer Institute. (2017). *Common Terminology Criteria for Adverse Events (CTCAE) Version 5.0*. U.S. Department of Health and Human Services.

Ni, C.C., Lin, Y.Y., Gao, J., Gu, X.D. and Saucan, E. (2015). Ricci curvature of the Internet topology. *2015 IEEE Conference on Computer Communications (INFOCOM)*, 2758 to 2766.

Ogbonna, K. (2026). *Project Confluence: Complexity Restoring Precision Oncology Framework* [Computer software]. GitHub. https://github.com/cloudynirvana/project-confluence

Pouryahya, M., Oh, J.H., Tannenbaum, A. and Deasy, J.O. (2024). ORCO: Ollivier Ricci Curvature Omics, an open source tool for biological network robustness analysis. *Bioinformatics*, 40(3), btae112.

Rosenstein, M.T., Collins, J.J. and De Luca, C.J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D: Nonlinear Phenomena*, 65(1 to 2), 117 to 134.

Sandhu, R., Georgiou, T., Reznik, E., Zhu, L., Kolesov, I., Senbabaoglu, Y. and Tannenbaum, A. (2015). Graph curvature for differentiating cancer networks. *Scientific Reports*, 5, 12323.

Strogatz, S.H. (2015). *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering*. 2nd Edition. Westview Press.

Transtrum, M.K. and Qiu, P. (2014). Model reduction by manifold boundaries. *Physical Review Letters*, 113(9), 098701.

Transtrum, M.K., Machta, B.B., Brown, K.S., Daniels, B.C., Myers, C.R. and Sethna, J.P. (2015). Perspective: sloppiness and emergent theories in physics, biology, and beyond. *Journal of Chemical Physics*, 143(1), 010901.

Wang, J., Xu, L., Wang, E. and Huang, S. (2008). The potential landscape of genetic circuits imposes the arrow of time in stem cell differentiation. *Biophysical Journal*, 99(1), 29 to 39.

Warburg, O. (1956). On the origin of cancer cells. *Science*, 123(3191), 309 to 314.

Xu, L. and Wang, J. (2024). Nonequilibrium early warning indicators for cell fate decision making: entropy production and curl flux based tipping point detection. *BioRxiv* preprint.

Zhang, J., Cunningham, J.J., Brown, J.S. and Gatenby, R.A. (2017). Integrating evolutionary dynamics into treatment of metastatic castrate resistant prostate cancer. *Nature Communications*, 8, 1816.
