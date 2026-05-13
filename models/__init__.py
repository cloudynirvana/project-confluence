"""
Project Confluence — Core Computational Models
================================================

Modules:
    complexity_profiler : 5D Φ vector computation (UCP Module 1)
    patient_fitter      : Bayesian digital twin creation (UCP Module 2)
                          Now supports Nigeria guideline priors via NSTG 2022
    drug_optimization_engine : RADO engine (UCP Module 3)
    ode_system          : 15D SAEM nonlinear ODE system
    immune_dynamics     : Multi-compartment immune force field
    intervention        : Drug library with PK/PD modeling
    alphafold_client    : AlphaFold DB REST client + pocket detection
    structure_bridge    : Protein structure → ODE parameter mapping
    structural_docking  : Geometric drug-target docking heuristics
    complexity_calibrator : Staged calibration pipeline (Genotype→SIS→Δθ→Φ)
    adaptive_controller : Closed-loop controller with NSTG 2022 guideline-aware safety
    geometric_pathways  : Freidlin-Wentzell minimum action pathway (MAP) optimizer
    fisher_geometry     : Fisher Information Matrix and stiff/sloppy decomposition
    network_curvature   : Forman-Ricci curvature for structural bottleneck detection
    biologic_operator   : Biologic agents as geometric operators on Φ-space
"""
