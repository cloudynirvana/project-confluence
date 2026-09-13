"""Melanoma persister archetype — immune-hot, fast phenotypic switching."""

from confluence.cancer_env.ode_system import ArchetypeParams


class MelanomaPersisterArchetype(ArchetypeParams):
    def __init__(self):
        super().__init__(
            name="melanoma_persister",
            display_name="Melanoma (persister)",
            r_s=0.20,
            r_r=0.13,
            k_carry=3.8,
            alpha_rs=0.75,
            alpha_sr=0.45,
            kappa_immune=0.42,
            kappa_kinase=0.55,
            resist_immune_factor=0.55,
            resist_kinase_factor=0.10,
            eps0=0.028,
            lambda_lactate=1.0,
            rho_immune=0.26,
            gamma0=0.07,
            sigma_stroma=0.07,
            p_lactate=0.16,
            cl_lactate=0.20,
            supply_o2=0.40,
            consume_o2=0.10,
            p_tgfb=0.10,
            p_ifng=0.30,
            kappa_burden=0.035,
            fusion_id="alk_braf_fusion_like",
            fusion_display="ALK / BRAF-fusion-class chimeric oncoprotein",
            r_f=0.15,
            tki_imatinib_weight=0.35,
            tki_alk_weight=0.85,
            kappa_fusion=0.70,
            x0=(0.78, 0.10, 0.42, 0.10, 0.12, 0.28, 0.80, 1.05, 0.22, 0.40, 0.93, 0.14),
            notes=(
                "Immune-hot cutaneous melanoma with a fast drug-tolerant persister "
                "switch (HDAC-sensitive). Kinase channel maps onto MEK pressure "
                "(trametinib catalog entry). Checkpoint occupancy is more potent here. "
                "Fusion clone is ALK/BRAF-fusion-class (rare but documented; research mapping)."
            ),
        )
