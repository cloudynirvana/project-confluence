"""Glioblastoma archetype — immune-cold, glycolytic, BBB-shielded."""

from confluence.cancer_env.ode_system import ArchetypeParams


class GlioblastomaArchetype(ArchetypeParams):
    def __init__(self):
        super().__init__(
            name="glioblastoma",
            display_name="Glioblastoma",
            r_s=0.22,
            r_r=0.14,
            k_carry=4.2,
            alpha_rs=0.80,
            alpha_sr=0.50,
            kappa_immune=0.18,
            kappa_kinase=0.32,
            resist_immune_factor=0.40,
            resist_kinase_factor=0.18,
            eps0=0.012,
            lambda_lactate=1.8,
            rho_immune=0.12,
            gamma0=0.11,
            sigma_stroma=0.10,
            p_lactate=0.30,
            cl_lactate=0.12,
            supply_o2=0.22,
            consume_o2=0.16,
            p_tgfb=0.18,
            kappa_burden=0.055,
            x0=(0.95, 0.12, 0.18, 0.22, 0.20, 0.70, 0.45, 1.30, 0.45, 0.16, 0.90),
            notes=(
                "IDH-wildtype-like metabolic reprogramming: extreme Warburg flux, "
                "lactate-driven immunosuppression, and a BBB-like drop in immune "
                "kill. Kinase channel is a stand-in for EGFR-adjacent pressure."
            ),
        )
