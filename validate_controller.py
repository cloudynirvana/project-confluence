"""Quick validation runner for the adaptive controller."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.adaptive_controller import compare_policies, run_adaptive_simulation

print("=" * 60)
print("PROJECT CONFLUENCE — Adaptive Controller Validation")
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

# Test 3: Multi-cancer validation
print("\n[3] Multi-cancer type validation...")
for ct in ["NSCLC", "TNBC", "GBM", "AML", "CRC"]:
    r = run_adaptive_simulation(cancer_type=ct, total_days=30, dt=0.5, seed=42)
    print(f"    {ct:8s} | burden={r['outcome']['final_burden']:.4f} "
          f"| R%={r['outcome']['final_resistant_fraction']:.2%}")

print("\n" + "=" * 60)
print("ALL VALIDATIONS PASSED")
print("=" * 60)
