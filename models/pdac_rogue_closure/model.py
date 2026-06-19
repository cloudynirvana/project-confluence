"""Executable PDAC rogue-closure model.

The model is intentionally compact: it turns the current Confluence synthesis
into falsifiable state variables rather than trying to be a patient simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class PDACParameters:
    carrying_capacity: float = 1.4
    base_growth: float = 0.021
    kras_drive: float = 1.35
    immune_kill_rate: float = 0.030
    ras_drug_effect: float = 0.030
    egfr_drug_effect: float = 0.012
    stat3_drug_effect: float = 0.016
    glyco_disruption_effect: float = 0.060
    stroma_modulation_effect: float = 0.038
    immune_boost_effect: float = 0.020
    resistance_selection: float = 0.018
    resistance_cost: float = 0.004
    glyco_production: float = 0.020
    glyco_decay: float = 0.010
    stroma_production: float = 0.013
    stroma_decay: float = 0.008
    immune_recovery: float = 0.018
    immune_suppression: float = 0.020
    stress_decay: float = 0.050
    stress_from_therapy: float = 0.065
    toxicity_per_dose: float = 0.007


@dataclass(frozen=True)
class PDACState:
    tumor: float = 0.62
    resistance: float = 0.08
    immune: float = 0.46
    glyco_shield: float = 0.58
    stroma: float = 0.64
    metabolic_stress: float = 0.18


@dataclass(frozen=True)
class TherapyDose:
    ras: float = 0.0
    egfr: float = 0.0
    stat3: float = 0.0
    glyco: float = 0.0
    stroma: float = 0.0
    immune_boost: float = 0.0

    @property
    def total(self) -> float:
        return self.ras + self.egfr + self.stat3 + self.glyco + self.stroma + self.immune_boost


@dataclass(frozen=True)
class ModelPoint:
    day: float
    scenario: str
    state: PDACState
    dose: TherapyDose
    rogue_closure: float
    host_access: float
    driver_signal: float
    bypass_signal: float


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def host_access(state: PDACState) -> float:
    """Approximate immune/drug access through stroma plus glycocalyx shielding."""

    return 1.0 / (1.0 + 1.75 * state.glyco_shield + 1.35 * state.stroma)


def driver_signal(state: PDACState, dose: TherapyDose, p: PDACParameters) -> float:
    ras_sensitive = 1.0 - state.resistance
    ras_blockade = p.ras_drug_effect * dose.ras * ras_sensitive
    return max(0.15, p.kras_drive * (1.0 - ras_blockade))


def bypass_signal(state: PDACState, dose: TherapyDose) -> float:
    raw = 0.42 * state.resistance + 0.25 * state.glyco_shield + 0.25 * state.stroma
    blocked = 0.34 * dose.egfr + 0.42 * dose.stat3
    return clamp(raw * (1.0 - blocked), 0.0, 1.0)


def rogue_closure_score(state: PDACState, dose: TherapyDose, p: PDACParameters) -> float:
    access = host_access(state)
    driver = driver_signal(state, dose, p) / max(p.kras_drive, 1e-12)
    bypass = bypass_signal(state, dose)
    shield = 0.52 * state.glyco_shield + 0.48 * state.stroma
    score = (
        0.34 * state.tumor
        + 0.20 * driver
        + 0.17 * bypass
        + 0.17 * shield
        + 0.12 * state.resistance
        - 0.14 * access
        - 0.10 * state.immune
        - 0.07 * state.metabolic_stress
    )
    return clamp(score, 0.0, 1.5)


def derivative(state: PDACState, dose: TherapyDose, p: PDACParameters) -> PDACState:
    access = host_access(state)
    driver = driver_signal(state, dose, p)
    bypass = bypass_signal(state, dose)
    shield_survival = 0.65 + 0.35 * (state.glyco_shield + state.stroma) / 2.0

    growth = p.base_growth * state.tumor * (1.0 - state.tumor / p.carrying_capacity)
    growth *= driver * (1.0 + 0.55 * bypass) * shield_survival
    immune_kill = p.immune_kill_rate * state.immune * access * state.tumor
    drug_delivery = 0.55 + 0.45 * access
    therapy_kill = state.tumor * drug_delivery * (
        p.ras_drug_effect * dose.ras * (1.0 - state.resistance)
        + p.egfr_drug_effect * dose.egfr
        + p.stat3_drug_effect * dose.stat3
    )
    stress_kill = 0.010 * state.metabolic_stress * state.tumor

    selection_pressure = dose.ras + 0.45 * dose.egfr + 0.45 * dose.stat3
    shield_pressure = 0.5 * (state.glyco_shield + state.stroma)
    d_resistance = (
        p.resistance_selection * selection_pressure * shield_pressure * (1.0 - state.resistance)
        - p.resistance_cost * state.resistance
        - 0.010 * (dose.egfr + dose.stat3) * state.resistance
    )

    d_glyco = (
        p.glyco_production * state.tumor * (0.7 + 0.3 * driver)
        + 0.010 * bypass
        - p.glyco_decay * state.glyco_shield
        - p.glyco_disruption_effect * dose.glyco * state.glyco_shield
    )
    d_stroma = (
        p.stroma_production * state.tumor * (1.0 + 0.4 * state.glyco_shield)
        - p.stroma_decay * state.stroma
        - p.stroma_modulation_effect * dose.stroma * state.stroma
    )
    toxicity = p.toxicity_per_dose * dose.total
    d_immune = (
        p.immune_recovery * (1.0 - state.immune)
        + p.immune_boost_effect * dose.immune_boost
        - p.immune_suppression * state.tumor * (state.glyco_shield + state.stroma) / 2.0
        - toxicity * state.immune
    )
    d_stress = (
        p.stress_from_therapy * (dose.ras + dose.egfr + dose.stat3)
        + 0.015 * (1.0 - access)
        - p.stress_decay * state.metabolic_stress
    )

    return PDACState(
        tumor=growth - immune_kill - therapy_kill - stress_kill,
        resistance=d_resistance,
        immune=d_immune,
        glyco_shield=d_glyco,
        stroma=d_stroma,
        metabolic_stress=d_stress,
    )


def add_scaled(state: PDACState, delta: PDACState, scale: float) -> PDACState:
    return PDACState(
        tumor=max(0.0, state.tumor + scale * delta.tumor),
        resistance=clamp(state.resistance + scale * delta.resistance),
        immune=clamp(state.immune + scale * delta.immune),
        glyco_shield=clamp(state.glyco_shield + scale * delta.glyco_shield),
        stroma=clamp(state.stroma + scale * delta.stroma),
        metabolic_stress=clamp(state.metabolic_stress + scale * delta.metabolic_stress),
    )


def rk4_step(state: PDACState, dose: TherapyDose, dt: float, p: PDACParameters) -> PDACState:
    k1 = derivative(state, dose, p)
    k2 = derivative(add_scaled(state, k1, dt / 2.0), dose, p)
    k3 = derivative(add_scaled(state, k2, dt / 2.0), dose, p)
    k4 = derivative(add_scaled(state, k3, dt), dose, p)
    return PDACState(
        tumor=max(0.0, state.tumor + dt * (k1.tumor + 2 * k2.tumor + 2 * k3.tumor + k4.tumor) / 6.0),
        resistance=clamp(state.resistance + dt * (k1.resistance + 2 * k2.resistance + 2 * k3.resistance + k4.resistance) / 6.0),
        immune=clamp(state.immune + dt * (k1.immune + 2 * k2.immune + 2 * k3.immune + k4.immune) / 6.0),
        glyco_shield=clamp(state.glyco_shield + dt * (k1.glyco_shield + 2 * k2.glyco_shield + 2 * k3.glyco_shield + k4.glyco_shield) / 6.0),
        stroma=clamp(state.stroma + dt * (k1.stroma + 2 * k2.stroma + 2 * k3.stroma + k4.stroma) / 6.0),
        metabolic_stress=clamp(state.metabolic_stress + dt * (k1.metabolic_stress + 2 * k2.metabolic_stress + 2 * k3.metabolic_stress + k4.metabolic_stress) / 6.0),
    )


Schedule = Callable[[float, PDACState, PDACParameters], TherapyDose]


def scenario_schedule(name: str) -> Schedule:
    if name == "untreated":
        return lambda day, state, p: TherapyDose()
    if name == "ras_only":
        return lambda day, state, p: TherapyDose(ras=0.90)
    if name == "triple_driver":
        return lambda day, state, p: TherapyDose(ras=0.82, egfr=0.48, stat3=0.52)
    if name == "closure_stack":
        return lambda day, state, p: TherapyDose(
            ras=0.76,
            egfr=0.42,
            stat3=0.48,
            glyco=0.46,
            stroma=0.34,
            immune_boost=0.22,
        )
    if name == "adaptive_closure":
        def adaptive(day: float, state: PDACState, p: PDACParameters) -> TherapyDose:
            closure = rogue_closure_score(state, TherapyDose(), p)
            pulse = 1.0 if int(day // 21) % 2 == 0 else 0.62
            intensity = clamp((closure - 0.32) / 0.45, 0.0, 1.0) * pulse
            immune_guard = 0.55 + 0.45 * state.immune
            return TherapyDose(
                ras=0.78 * intensity,
                egfr=0.42 * intensity,
                stat3=0.46 * intensity,
                glyco=0.50 * intensity,
                stroma=0.30 * intensity,
                immune_boost=0.25 * immune_guard,
            )
        return adaptive
    raise ValueError(f"unknown scenario: {name}")


def simulate(
    scenario: str,
    days: float = 180.0,
    dt: float = 0.5,
    initial: PDACState | None = None,
    parameters: PDACParameters | None = None,
) -> list[ModelPoint]:
    p = parameters or PDACParameters()
    state = initial or PDACState()
    schedule = scenario_schedule(scenario)
    points: list[ModelPoint] = []
    steps = int(math.ceil(days / dt))

    for step in range(steps + 1):
        day = min(step * dt, days)
        dose = schedule(day, state, p)
        points.append(
            ModelPoint(
                day=day,
                scenario=scenario,
                state=state,
                dose=dose,
                rogue_closure=rogue_closure_score(state, dose, p),
                host_access=host_access(state),
                driver_signal=driver_signal(state, dose, p),
                bypass_signal=bypass_signal(state, dose),
            )
        )
        if step < steps:
            state = rk4_step(state, dose, dt, p)

    return points


def summarize(points: list[ModelPoint]) -> dict[str, float | str]:
    first = points[0]
    final = points[-1]
    peak_tumor = max(point.state.tumor for point in points)
    min_access = min(point.host_access for point in points)
    mean_closure = sum(point.rogue_closure for point in points) / len(points)
    return {
        "scenario": final.scenario,
        "days": final.day,
        "initial_tumor": first.state.tumor,
        "final_tumor": final.state.tumor,
        "peak_tumor": peak_tumor,
        "initial_rogue_closure": first.rogue_closure,
        "final_rogue_closure": final.rogue_closure,
        "mean_rogue_closure": mean_closure,
        "final_resistance": final.state.resistance,
        "final_glyco_shield": final.state.glyco_shield,
        "final_stroma": final.state.stroma,
        "final_immune": final.state.immune,
        "final_host_access": final.host_access,
        "min_host_access": min_access,
    }


def flatten_point(point: ModelPoint) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "day": point.day,
        "scenario": point.scenario,
        "rogue_closure": point.rogue_closure,
        "host_access": point.host_access,
        "driver_signal": point.driver_signal,
        "bypass_signal": point.bypass_signal,
    }
    row.update({f"state_{key}": value for key, value in asdict(point.state).items()})
    row.update({f"dose_{key}": value for key, value in asdict(point.dose).items()})
    return row


def write_outputs(all_points: Iterable[ModelPoint], output_dir: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    points = list(all_points)

    csv_path = directory / "pdac_timeseries.csv"
    rows = [flatten_point(point) for point in points]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[ModelPoint]] = {}
    for point in points:
        grouped.setdefault(point.scenario, []).append(point)
    summary = {
        "schema": "pdac-rogue-closure/v1",
        "claim": "PDAC persistence is modeled as rogue closure maintained by RAS drive, bypass signaling, surface shielding, stromal shielding, immune exclusion, and resistance.",
        "guardrail": "Hypothesis generator only; not a clinical predictor or treatment recommendation.",
        "summaries": [summarize(grouped[name]) for name in sorted(grouped)],
    }
    json_path = directory / "pdac_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    svg_path = directory / "pdac_closure_report.svg"
    svg_path.write_text(render_svg(grouped), encoding="utf-8")
    return csv_path, json_path, svg_path


def _scale(value: float, lower: float, upper: float, pixels: float) -> float:
    return pixels * (value - lower) / max(upper - lower, 1e-12)


def _polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}" />'


def render_svg(grouped: dict[str, list[ModelPoint]]) -> str:
    colors = {
        "untreated": "#6b7280",
        "ras_only": "#c2410c",
        "triple_driver": "#2563eb",
        "closure_stack": "#059669",
        "adaptive_closure": "#7c3aed",
    }
    width = 1080
    height = 640
    left = 72
    top = 86
    panel_w = 420
    panel_h = 210
    all_points = [point for points in grouped.values() for point in points]
    max_day = max(point.day for point in all_points)

    def panel(title: str, metric: Callable[[ModelPoint], float], x0: float, y0: float) -> str:
        values = [metric(point) for point in all_points]
        low = min(values)
        high = max(values)
        pad = max((high - low) * 0.08, 0.05)
        low -= pad
        high += pad
        lines = [
            f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfaf7" stroke="#111827" />',
            f'<text x="{x0 + 14}" y="{y0 + 26}" font-size="17" font-family="Georgia" fill="#111827">{title}</text>',
        ]
        for scenario, points in grouped.items():
            scaled = [
                (
                    x0 + _scale(point.day, 0.0, max_day, panel_w),
                    y0 + panel_h - _scale(metric(point), low, high, panel_h),
                )
                for point in points
            ]
            lines.append(_polyline(scaled, colors.get(scenario, "#111827")))
        return "\n".join(lines)

    legend = []
    for index, scenario in enumerate(grouped):
        y = 42 + index * 20
        legend.append(f'<line x1="760" y1="{y - 4}" x2="788" y2="{y - 4}" stroke="{colors.get(scenario, "#111827")}" stroke-width="3" />')
        legend.append(f'<text x="798" y="{y}" font-size="13" font-family="Consolas" fill="#111827">{scenario}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f2efe8" />
<text x="72" y="42" font-size="24" font-family="Georgia" fill="#111827">PDAC Rogue Closure Executable Model</text>
<text x="72" y="64" font-size="13" font-family="Consolas" fill="#4b5563">RAS drive + EGFR/STAT3 bypass + glyco/stromal shielding + immune access.</text>
{chr(10).join(legend)}
{panel("Tumor Burden", lambda p: p.state.tumor, left, top)}
{panel("Rogue Closure", lambda p: p.rogue_closure, left + 520, top)}
{panel("Host Access", lambda p: p.host_access, left, top + 290)}
{panel("Resistance", lambda p: p.state.resistance, left + 520, top + 290)}
</svg>
"""


SCENARIOS = ["untreated", "ras_only", "triple_driver", "closure_stack", "adaptive_closure"]
