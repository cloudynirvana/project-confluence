"""PDAC archetype — dense stroma, TGF-β driven, immune-excluded."""

from confluence.cancer_env.ode_system import ArchetypeParams


class PancreaticPDACArchetype(ArchetypeParams):
    def __init__(self):
        super().__init__(
            name="pancreatic_pdac",
            display_name="PDAC / pancreatic",
            r_s=0.14,
            r_r=0.10,
            k_carry=3.6,
            alpha_rs=0.90,
            alpha_sr=0.60,
            kappa_immune=0.14,
            kappa_kinase=0.28,
            resist_immune_factor=0.35,
            resist_kinase_factor=0.15,
            eps0=0.010,
            lambda_lactate=1.2,
            rho_immune=0.10,
            gamma0=0.10,
            sigma_stroma=0.22,
            mu_stroma=0.025,
            p_lactate=0.20,
            cl_lactate=0.14,
            supply_o2=0.20,
            consume_o2=0.14,
            p_tgfb=0.28,
            cl_tgfb=0.08,
            kappa_burden=0.050,
            fusion_id="nrg1_ntrk_like",
            fusion_display="NRG1 / NTRK-class chimeric oncoprotein",
            r_f=0.13,
            tki_imatinib_weight=0.15,
            tki_alk_weight=1.0,
            kappa_fusion=0.75,
            x0=(0.80, 0.18, 0.14, 0.20, 0.62, 0.55, 0.38, 1.00, 0.75, 0.12, 0.88, 0.10),
            notes=(
                "Desmoplastic PDAC-like TME: high baseline fibrosis, TGF-β production, "
                "and stromal shielding of immune kill. Builds on the v1 PDAC rogue-"
                "closure intuition (KRAS persistence + exclusion) without claiming "
                "patient-level calibration. Fusion clone is NRG1/NTRK-class "
                "(KRAS-WT PDAC-inspired; research mapping)."
            ),
        )
