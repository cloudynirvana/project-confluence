"""Five disease-class archetypes with distinct X and Y signatures.

Tissue archetypes (GBM / PDAC / melanoma) remain and are tagged malignant.
These class modes exist so controllers and the validation suite can
discriminate benign / malignant / occult / dormant / terminal dynamics.
"""

from confluence.cancer_env.ode_system import ArchetypeParams


class BenignArchetype(ArchetypeParams):
    def __init__(self):
        super().__init__(
            name="benign",
            display_name="Benign (growth-limited)",
            disease_class="benign",
            r_s=0.045,
            r_r=0.020,
            r_f=0.018,
            k_carry=0.85,
            kappa_immune=0.58,
            kappa_kinase=0.20,
            rho_immune=0.28,
            gamma0=0.04,
            p_tgfb=0.04,
            sigma_stroma=0.04,
            p_lactate=0.08,
            kappa_burden=0.012,
            invasion_factor=0.35,
            immune_evasion=0.30,
            clinical_visibility_k=0.08,
            k_junction_shed=0.25,
            occult_af_leak=0.05,
            growth_awake_gate=False,
            fusion_id="none_benign",
            fusion_display="no chimeric-driver emphasis (benign mode)",
            x0=(0.22, 0.02, 0.55, 0.06, 0.08, 0.18, 0.90, 1.20, 0.10, 0.40, 0.96, 0.02, 0.20, 0.10, 1.0),
            notes=(
                "Growth-limited lesion: low r, low carrying capacity, high immune "
                "kill, low TGF-β / invasion. Y is high-visibility and stays quiet. "
                "Research mode, not a histopathology diagnosis."
            ),
        )


class MalignantClassArchetype(ArchetypeParams):
    """Named malignant mode — GBM-like aggressive TME (class, not a new tissue)."""

    def __init__(self):
        super().__init__(
            name="malignant",
            display_name="Malignant (aggressive)",
            disease_class="malignant",
            r_s=0.22,
            r_r=0.14,
            k_carry=4.2,
            kappa_immune=0.18,
            kappa_kinase=0.32,
            rho_immune=0.12,
            gamma0=0.11,
            p_tgfb=0.18,
            p_lactate=0.30,
            kappa_burden=0.055,
            invasion_factor=1.0,
            immune_evasion=1.0,
            clinical_visibility_k=0.12,
            k_junction_shed=0.85,
            fusion_id="fgfr3_tacc3_like",
            fusion_display="FGFR3–TACC3-class chimeric oncoprotein",
            r_f=0.17,
            tki_imatinib_weight=0.25,
            tki_alk_weight=0.90,
            x0=(0.95, 0.12, 0.18, 0.22, 0.20, 0.70, 0.45, 1.30, 0.45, 0.16, 0.90, 0.28, 0.10, 0.06, 1.0),
            notes="Aggressive bulk + evasion. Same family as the GBM tissue table.",
        )


class OccultArchetype(ArchetypeParams):
    def __init__(self):
        super().__init__(
            name="occult",
            display_name="Occult / hidden",
            disease_class="occult",
            r_s=0.16,
            r_r=0.10,
            r_f=0.14,
            k_carry=3.4,
            kappa_immune=0.22,
            rho_immune=0.14,
            p_tgfb=0.12,
            clinical_visibility_k=1.35,
            k_junction_shed=2.20,
            occult_af_leak=0.55,
            invasion_factor=0.85,
            immune_evasion=0.90,
            fusion_id="occult_fusion_like",
            fusion_display="occult chimeric-junction leak (research)",
            x0=(0.12, 0.02, 0.28, 0.10, 0.10, 0.22, 0.80, 1.15, 0.18, 0.22, 0.94, 0.08, 0.08, 0.05, 1.0),
            notes=(
                "Low bulk burden; clinical Y_burden is attenuated until late. "
                "Junction neoantigen and occult AF leak earlier (research proxy)."
            ),
        )


class DormantArchetype(ArchetypeParams):
    def __init__(self):
        super().__init__(
            name="dormant",
            display_name="Dormant (stochastic wake)",
            disease_class="dormant",
            r_s=0.18,
            r_r=0.12,
            r_f=0.15,
            k_carry=3.6,
            kappa_immune=0.26,
            growth_awake_gate=True,
            awaken_hazard=0.07,
            awaken_jump=0.50,
            clinical_visibility_k=0.20,
            k_junction_shed=1.10,
            occult_af_leak=0.20,
            invasion_factor=0.90,
            immune_evasion=0.80,
            fusion_id="dormant_seed_like",
            fusion_display="dormant chimeric-seed (research)",
            x0=(0.16, 0.03, 0.32, 0.08, 0.10, 0.20, 0.85, 1.15, 0.16, 0.24, 0.95, 0.05, 0.12, 0.08, 0.05),
            notes=(
                "Near-zero net growth while awake≈0. Stochastic awakening "
                "(placeholder hazard) raises dormancy_exit in Y."
            ),
        )


class TerminalArchetype(ArchetypeParams):
    def __init__(self):
        super().__init__(
            name="terminal",
            display_name="Terminal (host-failure)",
            disease_class="terminal",
            r_s=0.24,
            r_r=0.16,
            r_f=0.18,
            k_carry=5.0,
            kappa_immune=0.10,
            kappa_burden=0.14,
            kappa_tox=0.18,
            r_host=0.015,
            p_tgfb=0.22,
            p_lactate=0.32,
            invasion_factor=1.15,
            immune_evasion=1.20,
            clinical_visibility_k=0.08,
            k_junction_shed=0.90,
            fusion_id="terminal_burden_like",
            fusion_display="high-burden chimeric clone (research)",
            x0=(2.10, 0.70, 0.08, 0.35, 0.55, 1.10, 0.28, 0.70, 0.70, 0.08, 0.30, 0.55, 0.04, 0.03, 1.0),
            notes=(
                "High burden and H already near failure. Host-health dynamics "
                "dominate. Research mode, not a hospice protocol."
            ),
        )
