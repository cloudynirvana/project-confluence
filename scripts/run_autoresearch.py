"""
AutoResearchClaw Launcher for Project Confluence.

Usage:
  python scripts/run_autoresearch.py "Your research topic here"
  python scripts/run_autoresearch.py --list-topics
  python scripts/run_autoresearch.py --help

This script:
  1. Verifies AutoResearchClaw is installed
  2. Loads the Confluence-specific config
  3. Runs the full 23-stage research pipeline
  4. Copies deliverables to results/autoresearch/
"""
import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARC_ROOT = PROJECT_ROOT.parent / "AutoResearchClaw"
CONFIG_PATH = PROJECT_ROOT / "config.arc.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "autoresearch"


# Pre-built research topics relevant to Confluence
SUGGESTED_TOPICS = {
    "phi-universality": (
        "Test whether the 5D Phi complexity vector provides a universal "
        "biomarker for disease severity across 9 cancer types (TNBC, PDAC, "
        "NSCLC, Melanoma, GBM, CRC, HGSOC, mCRPC, AML) using the 15D SAEM "
        "ODE model with Michaelis-Menten kinetics."
    ),
    "drug-scheduling": (
        "Compare complexity-restoring drug scheduling (Phi-guided RADO "
        "optimization) vs traditional maximum-tolerated-dose approaches "
        "across pan-cancer metabolic archetypes."
    ),
    "immune-metabolic": (
        "Investigate immune-metabolic coupling (I_eff, I_reg, I_exhaust "
        "coupled to glucose/glutamine dynamics) as a therapeutic target "
        "in the 15D SAEM model."
    ),
    "ferroptosis-complexity": (
        "Analyze whether ferroptosis-inducing interventions restore "
        "dynamical complexity (Phi) more effectively than conventional "
        "cytotoxic approaches in ROS-susceptible cancer archetypes."
    ),
    "digital-twin": (
        "Evaluate Bayesian digital twin fitting accuracy using MCMC-based "
        "patient parameter estimation and its correlation with real-world "
        "TCGA survival outcomes via Phi-distance metrics."
    ),
}


def check_arc_installed() -> bool:
    """Check if researchclaw CLI is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "researchclaw", "--help"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def install_arc():
    """Install AutoResearchClaw from the cloned repo."""
    if not ARC_ROOT.exists():
        print(f"❌ AutoResearchClaw not found at {ARC_ROOT}")
        print(f"   Clone it: git clone https://github.com/aiming-lab/AutoResearchClaw.git {ARC_ROOT}")
        sys.exit(1)

    print(f"📦 Installing AutoResearchClaw from {ARC_ROOT}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(ARC_ROOT)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Install failed:\n{result.stderr}")
        sys.exit(1)
    print("✅ AutoResearchClaw installed.")


def run_research(topic: str, auto_approve: bool = True):
    """Run the full research pipeline."""
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        print(f"   Create it from the template in project-confluence/")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, "-m", "researchclaw", "run",
        "--config", str(CONFIG_PATH),
        "--topic", topic,
    ]
    if auto_approve:
        cmd.append("--auto-approve")

    print(f"\n🚀 AutoResearchClaw Pipeline Starting")
    print(f"   Topic: {topic[:80]}...")
    print(f"   Config: {CONFIG_PATH}")
    print(f"   Auto-approve: {auto_approve}")
    print(f"   {'='*60}\n")

    # Run
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    # Stream output
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode == 0:
        print(f"\n✅ Pipeline complete!")
        copy_deliverables()
    else:
        print(f"\n❌ Pipeline failed with exit code {process.returncode}")
        sys.exit(process.returncode)


def copy_deliverables():
    """Copy latest artifacts to results/autoresearch/."""
    artifacts_dir = PROJECT_ROOT / "artifacts"
    if not artifacts_dir.exists():
        print("⚠️ No artifacts directory found.")
        return

    # Find latest run
    runs = sorted(artifacts_dir.glob("rc-*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        print("⚠️ No run artifacts found.")
        return

    latest = runs[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = RESULTS_DIR / timestamp
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copytree(latest, dest, dirs_exist_ok=True)
    print(f"\n📄 Deliverables copied to: {dest}")
    print(f"   Look for: paper_draft.md, paper.tex, references.bib, charts/")


def main():
    parser = argparse.ArgumentParser(
        description="🦞 AutoResearchClaw × Project Confluence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suggested topics:
  --topic phi-universality     Test Phi as universal biomarker
  --topic drug-scheduling      Compare Phi-guided vs MTD scheduling
  --topic immune-metabolic     Immune-metabolic coupling target
  --topic ferroptosis-complexity  Ferroptosis + complexity restoration
  --topic digital-twin         Bayesian digital twin validation
  
Or provide any custom topic string.
        """
    )
    parser.add_argument(
        "topic", nargs="?", default=None,
        help="Research topic (or shorthand key from suggested topics)"
    )
    parser.add_argument(
        "--list-topics", action="store_true",
        help="List pre-built research topics"
    )
    parser.add_argument(
        "--no-auto-approve", action="store_true",
        help="Require manual approval at gate stages"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Install AutoResearchClaw before running"
    )

    args = parser.parse_args()

    if args.list_topics:
        print("\n📚 Pre-built Research Topics for Project Confluence:\n")
        for key, desc in SUGGESTED_TOPICS.items():
            print(f"  {key:25s} {desc[:70]}...")
        print(f"\nUsage: python {__file__} <topic-key>")
        return

    if args.install or not check_arc_installed():
        install_arc()

    if not args.topic:
        parser.print_help()
        return

    # Resolve shorthand topic
    topic = SUGGESTED_TOPICS.get(args.topic, args.topic)

    run_research(topic, auto_approve=not args.no_auto_approve)


if __name__ == "__main__":
    main()
