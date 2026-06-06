"""Quick validation runner for the adaptive controller."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.adaptive_controller import compare_policies, run_adaptive_simulation


failures = []


def require(condition, message):
    """Record a validation failure without hiding earlier diagnostic output."""
    if not condition:
        failures.append(message)


print("=" * 60)
print("PROJECT CONFLUENCE - Adaptive Controller Validation")
print("=" * 60)

# Test 1: Run a basic simulation
print("\n[1] Running NSCLC adaptive simulation (60 days)...")
result = run_adaptive_simulation(
    cancer_type="NSCLC", total_days=60, dt=0.5, seed=42
)
o = result["outcome"]
c = result["controller_summary"]
print(f"    Final burden:     {o['final_burden']:.4f}")
print(f"    Resistant frac:   {o['final_resistant_fraction']:.2%}")
print(f"    Days controlled:  {o['days_under_control']}")
print(f"    R takeover:       {o['resistant_takeover']}")
print(f"    Dose switches:    {c['dose_switches']}")
print(f"    Dosing fraction:  {c['dosing_fraction']:.1%}")
print(f"    Cumul. toxicity:  {c['cumulative_toxicity']:.1f}")

require(o["final_burden"] >= 0.0, "NSCLC adaptive final burden is negative")
require(0.0 <= o["final_resistant_fraction"] <= 1.0,
        "NSCLC adaptive resistant fraction is outside [0, 1]")
require(not o["resistant_takeover"],
        "NSCLC adaptive validation produced resistant takeover")
require(c["cumulative_toxicity"] >= 0.0,
        "NSCLC adaptive cumulative toxicity is negative")

# Test 2: Compare MTD vs Adaptive
print("\n[2] Comparing MTD vs Fixed Low vs Adaptive (60 days)...")
comp = compare_policies("NSCLC", total_days=60, dt=0.5, seed=42)
for name, metrics in comp["comparison"].items():
    print(f"    {name:12s} | burden={metrics['final_burden']:.4f} "
          f"| R%={metrics['final_R_fraction']:.2%} "
          f"| ctrl_days={metrics['days_controlled']} "
          f"| takeover={metrics['resistant_takeover']}")

adaptive_wins = (
    comp["comparison"]["Adaptive"]["final_R_fraction"]
    < comp["comparison"]["MTD"]["final_R_fraction"]
)
print(f"\n    >>> Adaptive beats MTD on resistance: {adaptive_wins}")

require(adaptive_wins, "Adaptive did not beat MTD on resistant fraction")
require(not comp["comparison"]["Adaptive"]["resistant_takeover"],
        "Adaptive comparison produced resistant takeover")
require(comp["comparison"]["MTD"]["resistant_takeover"],
        "MTD comparison did not produce the expected resistant takeover stress test")

# Test 3: Multi-cancer validation
print("\n[3] Multi-cancer type validation...")
for ct in ["NSCLC", "TNBC", "GBM", "AML", "CRC"]:
    r = run_adaptive_simulation(cancer_type=ct, total_days=30, dt=0.5, seed=42)
    print(f"    {ct:8s} | burden={r['outcome']['final_burden']:.4f} "
          f"| R%={r['outcome']['final_resistant_fraction']:.2%}")
    require(r["outcome"]["final_burden"] >= 0.0,
            f"{ct} final burden is negative")
    require(0.0 <= r["outcome"]["final_resistant_fraction"] <= 1.0,
            f"{ct} resistant fraction is outside [0, 1]")

print("\n" + "=" * 60)
if failures:
    print("VALIDATION FAILED")
    for failure in failures:
        print(f" - {failure}")
    raise SystemExit(1)
print("ALL VALIDATIONS PASSED")
print("=" * 60)
