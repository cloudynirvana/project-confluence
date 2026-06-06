"""Prepare local directories and dependency checks for external validation.

This script intentionally does not download biomedical data. It creates the
expected local folder structure, checks optional loader dependencies, and prints
the next validation tasks. Raw or restricted data should remain outside Git.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "validation" / "external_data_manifest.json"


OPTIONAL_PACKAGES = {
    "numpy": "core numerical arrays",
    "scipy": "ODEs and signal processing",
    "mne": "Sleep-EDF EDF loading and annotation handling",
    "wfdb": "PhysioNet WFDB record and annotation loading",
    "requests": "GDC/cBioPortal API queries",
    "pandas": "tabular cohort metadata",
    "datasets": "Hugging Face dataset-card/prototype loading",
}


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def ensure_directories(manifest: dict) -> list[Path]:
    created = []
    for rel_path in manifest["local_paths"].values():
        path = PROJECT_ROOT / rel_path
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)

    for source in manifest["sources"]:
        source_dir = PROJECT_ROOT / manifest["local_paths"]["raw_data_root"] / source["id"]
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / ".gitkeep").touch()
        created.append(source_dir)

    return created


def print_dependency_report() -> None:
    print("Optional dependency check")
    print("-" * 32)
    for package, purpose in OPTIONAL_PACKAGES.items():
        status = "OK" if package_available(package) else "MISSING"
        print(f"{package:10s} {status:8s} {purpose}")


def print_source_plan(manifest: dict) -> None:
    print("\nExternal validation source plan")
    print("-" * 32)
    for source in sorted(manifest["sources"], key=lambda item: item["priority"]):
        credential = source["credentialing_required"]
        print(f"[P{source['priority']}] {source['id']}")
        print(f"  provider: {source['provider']}")
        print(f"  access:   {source['access']} | credentialing: {credential}")
        print(f"  question: {source['validation_question']}")


def print_guardrails(manifest: dict) -> None:
    print("\nDo not commit")
    print("-" * 32)
    for item in manifest["do_not_commit"]:
        print(f"- {item}")


def main() -> None:
    manifest = load_manifest()
    created = ensure_directories(manifest)

    print("External validation preparation complete")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Directories ensured: {len(created)}")
    print_dependency_report()
    print_source_plan(manifest)
    print_guardrails(manifest)


if __name__ == "__main__":
    main()
