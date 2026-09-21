#!/usr/bin/env python3
"""Assemble and push Thesis #1 and Thesis #3 dedicated GitHub repos.

Source of truth remains this checkout (project-confluence). The dedicated
repos are publication copies: README, THESIS.md, THESIS.pdf, CITATION.cff,
DISCLAIMER.md, and refs/.

    python3 scripts/publish_dedicated_thesis_repos.py

Requires write access to:
  - github.com/cloudynirvana/thesis-01-confluence-onco
  - github.com/cloudynirvana/thesis-03-disease-profile

The Cloud Agent GitHub App installation for this workspace is scoped to
project-confluence only; grant those two repositories to the same
installation (or run this script with a PAT that can push) before expecting
a successful push.

Research only. Not a medical device. Not clinical decision support.
No document DOI is invented.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST_ROOT = Path("/tmp/thesis-publish")

T1_REMOTE = "https://github.com/cloudynirvana/thesis-01-confluence-onco.git"
T3_REMOTE = "https://github.com/cloudynirvana/thesis-03-disease-profile.git"

AUTHOR = "Kelechi Emeka Ogbonna"
EMAIL = "kelechiogbonna300@gmail.com"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)


def _extract_refs(markdown: str, heading: str) -> list[str]:
    _, _, rest = markdown.partition(heading)
    out: list[str] = []
    for line in rest.splitlines():
        if line.startswith("## "):
            break
        if line.strip().startswith("<!--"):
            continue
        if re.match(r"^\d+\.\s", line.strip()):
            out.append(line.strip())
    return out


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def assemble_thesis_01(dest: Path) -> None:
    ms = (REPO / "docs/manuscript/thesis_01_confluence_onco.md").read_text(encoding="utf-8")
    pdf = REPO / "evidence/thesis.pdf"
    bib_json = REPO / "docs/manuscript/thesis_01_bibliography.json"
    dest.mkdir(parents=True, exist_ok=True)
    _write(dest / "THESIS.md", ms)
    shutil.copy2(pdf, dest / "THESIS.pdf")
    (dest / "refs").mkdir(exist_ok=True)
    shutil.copy2(bib_json, dest / "refs/thesis_01_bibliography.json")
    refs = _extract_refs(ms, "## References")
    _write(
        dest / "refs/references.md",
        "# Thesis #1 — Vancouver references\n\n"
        "Numbered Vancouver list matching THESIS.md. Journal DOIs are Crossref/PubMed-verified. "
        "No DOI is invented. This document has no registered DOI.\n\n"
        + "\n".join(refs)
        + "\n",
    )
    _write(
        dest / "DISCLAIMER.md",
        f"""# Disclaimer — Thesis #1 (research only)

This repository publishes a **computational research manuscript**.

- **NOT** a medical device
- **NOT** a clinical decision-support system
- **NOT** a diagnostic, prognostic, or therapeutic product
- **NOT** a protocol, dose, or cure
- **NOT** a claim that OnCo knowledge, OnCo confidence, or a legacy gene-to-parameter map is an identified parameter Θ
- **NOT** approved by the FDA, EMA, NAFDAC, or any regulator

Simulated trajectories, adapter outputs, and architectural findings are research artefacts. They are not patient outcomes.

OnCo data (onco.cc) is cited under CC BY-NC 4.0; commercial use needs a licence.
CONFLUENCE software (project-confluence) is MIT.

Author: {AUTHOR} — {EMAIL}
""",
    )
    _write(
        dest / "CITATION.cff",
        """cff-version: 1.2.0
title: "CONFLUENCE × OnCo: An Evidence-Gated Dynamical Framework for Integrating Oncology Knowledge Graphs with Adaptive Cancer-State Models"
message: "If you cite this manuscript, please use the metadata below. This record is the working thesis, not a journal article. No DOI is registered for this document. Do not reuse the Project Confluence software Zenodo DOI."
type: article
authors:
  - family-names: Ogbonna
    given-names: Kelechi Emeka
    email: kelechiogbonna300@gmail.com
repository-code: "https://github.com/cloudynirvana/thesis-01-confluence-onco"
url: "https://confluence-research.vercel.app/thesis"
license: MIT
date-released: "2026-09-20"
abstract: "Cancer research now produces knowledge graphs, multi-omic assays and dynamical simulators in parallel. The failure mode is collapsing those layers. This study asks whether a provenance-controlled pipeline can keep the layers apart while still allowing testable predictions. Scientific success is a staged chain — traceability, mathematical validity, identifiability, out-of-sample prediction, experimental falsification — not disease eradication. Research only. Not a medical device. Not clinical decision support."
keywords:
  - computational oncology
  - knowledge graphs
  - dynamical systems
  - evidence gates
  - identifiability
  - OnCo
  - CONFLUENCE
  - research-only
""",
    )
    _write(
        dest / "README.md",
        f"""# CONFLUENCE × OnCo: An Evidence-Gated Dynamical Framework for Integrating Oncology Knowledge Graphs with Adaptive Cancer-State Models

**Thesis #1** — working computational-research manuscript.

**Author:** {AUTHOR}  
**Email:** {EMAIL}  
**Affiliation:** Independent computational research / Project Confluence  
**Date:** 20 September 2026 (revision 21 September 2026: Problem / Justification / Significance sections)

## Non-claims

This thesis is **research only**. It is not a medical device, not clinical decision support, not a protocol, not a dose, and not a cure. OnCo knowledge is not an identified parameter Θ. No DOI is registered for this document; do not invent one and do not reuse the Project Confluence software Zenodo record.

See [DISCLAIMER.md](DISCLAIMER.md).

## Live copies

- HTML (Scholar landing): https://confluence-research.vercel.app/thesis
- Citeable PDF: https://confluence-research.vercel.app/thesis.pdf
- Parent codebase: https://github.com/cloudynirvana/project-confluence
- This dedicated manuscript repo: https://github.com/cloudynirvana/thesis-01-confluence-onco

## Files

| Path | Role |
|---|---|
| `THESIS.md` | Full manuscript (Vancouver citations; Problem / Justification / Significance before Methods) |
| `THESIS.pdf` | Regenerated citeable PDF |
| `refs/references.md` | Numbered Vancouver bibliography |
| `refs/thesis_01_bibliography.json` | Machine-readable bibliography source |
| `CITATION.cff` | Citation metadata (no document DOI) |
| `DISCLAIMER.md` | Research-only disclaimer |

## How to cite

Ogbonna KE. CONFLUENCE × OnCo: an evidence-gated dynamical framework for integrating oncology knowledge graphs with adaptive cancer-state models [Internet]. Thesis #1 working manuscript. 20 September 2026 [cited YYYY Mon DD]. Available from: https://github.com/cloudynirvana/thesis-01-confluence-onco and https://confluence-research.vercel.app/thesis.pdf

Prefer `CITATION.cff` for machine-readable citation. When a document DOI is later minted, add it there only after it exists.

## Licence

Manuscript text in this repository is provided for scholarly reuse with attribution. OnCo data cited in the text remains CC BY-NC 4.0.
""",
    )


def assemble_thesis_03(dest: Path) -> None:
    ms = (REPO / "docs/manuscript/thesis_03_disease_profile_method.md").read_text(encoding="utf-8")
    pdf = REPO / "docs/manuscript/thesis_03_disease_profile_method.pdf"
    dest.mkdir(parents=True, exist_ok=True)
    _write(dest / "THESIS.md", ms)
    shutil.copy2(pdf, dest / "THESIS.pdf")
    refs = _extract_refs(ms, "## 8. References")
    _write(
        dest / "refs/references.md",
        "# Thesis #3 — Vancouver references\n\n"
        "Numbered Vancouver list matching THESIS.md (n = 77). Journal DOIs are Crossref-verified. "
        "No DOI is invented. This document has no registered DOI.\n\n"
        + "\n".join(refs)
        + "\n",
    )
    _write(
        dest / "DISCLAIMER.md",
        f"""# Disclaimer — Thesis #3 (research only)

This repository publishes a **computational research method manuscript**.

**This document is not personalized medicine as a clinical product.**

- **NOT** a medical device
- **NOT** clinical decision support
- **NOT** a patient chart, care plan, or dose
- **NOT** a diagnostic, prognostic, or therapeutic tool
- **NOT** a claim of cure, disease eradication, or a treatment path
- **NOT** a claim that OnCo knowledge, OnCo confidence, Idea maturity, or a legacy gene-to-parameter map is identified Θ
- **NOT** an execution of Nigeria Standard Treatment Guidelines
- **NOT** approved by the FDA, EMA, NAFDAC, or any regulator

Disease Profiles, thinking-lab boards, and HypothesisObjects are research artefacts. Qualitative boards are not simulated patient benefit. Public-dataset names are experimental bindings, not completed analyses.

Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence.

Author: {AUTHOR} — {EMAIL}
""",
    )
    _write(
        dest / "CITATION.cff",
        """cff-version: 1.2.0
title: "Disease Profiles for Complex Pathologies: A Gated Method for Systemic Personalized-Medicine Research Objects"
message: "If you cite this manuscript, please use the metadata below. This record is a working method thesis, not a journal article. No DOI is registered for this document. Do not invent a DOI."
type: article
authors:
  - family-names: Ogbonna
    given-names: Kelechi Emeka
    email: kelechiogbonna300@gmail.com
repository-code: "https://github.com/cloudynirvana/thesis-03-disease-profile"
url: "https://confluence-research.vercel.app/thesis-03"
license: MIT
date-released: "2026-09-20"
abstract: "This thesis defines the Disease Profile as a versioned research object for systemic personalized-medicine research. A profile is not a patient chart, not clinical decision support, and not personalized medicine as a clinical product. Admission is gated: Knowledge ≠ Evidence ≠ Mechanism ≠ Parameter ≠ Prediction. Worked examples are qualitative boards, not patient outcomes. Research only."
keywords:
  - Disease Profile
  - research object
  - personalized-medicine research
  - knowledge gates
  - identifiability
  - OnCo
  - CONFLUENCE
  - not clinical decision support
""",
    )
    _write(
        dest / "README.md",
        f"""# Disease Profiles for Complex Pathologies: A Gated Method for Systemic Personalized-Medicine Research Objects

**Thesis #3** — working method manuscript (research / in-silico only).

**Author:** {AUTHOR}  
**Email:** {EMAIL}  
**Affiliation:** Project Confluence (computational research)  
**Date:** 20 September 2026 (revision 21 September 2026: Problem / Justification / Significance sections)

## Non-claims

**This thesis is not personalized medicine as a clinical product.** It is not a medical device, not clinical decision support, not a patient chart, not a dose, and not a cure. OnCo knowledge is not Θ. No DOI is registered for this document; do not invent one.

See [DISCLAIMER.md](DISCLAIMER.md).

## Live copies

- HTML (Scholar landing): https://confluence-research.vercel.app/thesis-03
- PDF on the evidence site: https://confluence-research.vercel.app/papers/thesis_03_disease_profile_method.pdf
- Parent codebase: https://github.com/cloudynirvana/project-confluence
- This dedicated manuscript repo: https://github.com/cloudynirvana/thesis-03-disease-profile

## Files

| Path | Role |
|---|---|
| `THESIS.md` | Full manuscript (Vancouver citations; Problem / Justification / Significance before Methods) |
| `THESIS.pdf` | Regenerated method PDF |
| `refs/references.md` | Numbered Vancouver bibliography (77 entries) |
| `CITATION.cff` | Citation metadata (no document DOI) |
| `DISCLAIMER.md` | Research-only disclaimer |

## How to cite

Ogbonna KE. Disease profiles for complex pathologies: a gated method for systemic personalized-medicine research objects [Internet]. Thesis #3 working method manuscript. 20 September 2026 [cited YYYY Mon DD]. Available from: https://github.com/cloudynirvana/thesis-03-disease-profile and https://confluence-research.vercel.app/papers/thesis_03_disease_profile_method.pdf

Prefer `CITATION.cff` for machine-readable citation. When a document DOI is later minted, add it there only after it exists.

## Licence

Manuscript text in this repository is provided for scholarly reuse with attribution. OnCo data cited in the text remains CC BY-NC 4.0.
""",
    )


def _origin_token() -> str | None:
    try:
        url = _run(["git", "remote", "get-url", "origin"], cwd=REPO).stdout.strip()
    except subprocess.CalledProcessError:
        return None
    match = re.search(r"x-access-token:([^@]+)@", url)
    return match.group(1) if match else None


def push_tree(dest: Path, remote: str, message: str, push: bool) -> str:
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-b", "main"], cwd=dest)
        _run(["git", "remote", "add", "origin", remote], cwd=dest)
    token = _origin_token()
    authed = remote
    if token and remote.startswith("https://github.com/"):
        authed = remote.replace("https://github.com/", f"https://x-access-token:{token}@github.com/")
        _run(["git", "remote", "set-url", "origin", authed], cwd=dest)
    _run(["git", "add", "-A"], cwd=dest)
    status = _run(["git", "status", "--porcelain"], cwd=dest).stdout.strip()
    if status:
        _run(["git", "commit", "-m", message], cwd=dest)
    sha = _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
    if push:
        result = _run(["git", "push", "-u", "origin", "main"], cwd=dest, check=False)
        if result.returncode != 0:
            # Strip token from any echoed URL.
            err = re.sub(r"x-access-token:[^@]+@", "x-access-token:<redacted>@", result.stderr)
            print(f"push failed for {remote}:\n{err}", file=sys.stderr)
            raise SystemExit(result.returncode)
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-push", action="store_true", help="Assemble and commit locally only")
    parser.add_argument("--out", type=Path, default=DEST_ROOT)
    args = parser.parse_args()

    t1 = args.out / "thesis-01-confluence-onco"
    t3 = args.out / "thesis-03-disease-profile"
    assemble_thesis_01(t1)
    assemble_thesis_03(t3)
    sha1 = push_tree(
        t1,
        T1_REMOTE,
        "Publish Thesis #1 manuscript with Problem, Justification, and Significance",
        push=not args.no_push,
    )
    sha3 = push_tree(
        t3,
        T3_REMOTE,
        "Publish Thesis #3 Disease Profile method manuscript",
        push=not args.no_push,
    )
    print(f"thesis-01-confluence-onco {sha1}")
    print(f"thesis-03-disease-profile {sha3}")


if __name__ == "__main__":
    main()
