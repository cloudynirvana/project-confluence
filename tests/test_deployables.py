"""Clinical briefing + evidence lab reel + hosted-sim deployables (research only)."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_evidence_site_is_static_and_honest() -> None:
    index = (REPO / "evidence" / "index.html").read_text(encoding="utf-8")
    lowered = index.lower()
    assert "simulation / research" in lowered
    assert "disclaimer" in lowered
    assert "connectome" in lowered
    assert "cancer ode" in lowered or "cancer ode" in index.lower()
    assert "flybody" in lowered
    assert "pip install -e ." in index
    assert "python -m confluence" in index
    compact = " ".join(lowered.split())
    assert "vercel serverless" in compact
    assert "fastapi" in compact and "websocket" in compact
    assert "awaiting-clinical-validation" in lowered
    assert "DISCLAIMER.md" in index
    assert 'href="/thinking"' in index
    assert 'href="/thesis"' in index
    assert 'href="/thesis-03"' in index
    assert "thinking lab" in lowered
    assert "research thesis" in lowered
    thesis03 = (REPO / "evidence" / "thesis-03.html").read_text(encoding="utf-8")
    assert 'name="citation_title"' in thesis03
    assert 'name="citation_author"' in thesis03
    assert 'name="citation_pdf_url"' in thesis03
    assert "not personalized medicine as a clinical product" in thesis03.lower()
    assert (REPO / "docs" / "manuscript" / "thesis_03_disease_profile_method.md").is_file()
    assert (REPO / "docs" / "manuscript" / "thesis_03_disease_profile_method.pdf").is_file()
    assert (REPO / "evidence" / "papers" / "thesis_03_disease_profile_method.pdf").is_file()
    assert (REPO / "evidence" / "sitemap-thesis03.xml").is_file()
    for phrase in (
        "this is a cure",
        "fda-approved",
        "phase ii result",
        "clinically validated treatment",
    ):
        assert phrase not in lowered
    vercel = (REPO / "evidence" / "vercel.json").read_text(encoding="utf-8")
    assert '"outputDirectory": "."' in vercel
    assert "/thesis-03" in vercel
    assert not (REPO / "evidence" / "public").exists()
    assert (REPO / "evidence" / "assets" / "still_hero.png").is_file()
    assert (REPO / "evidence" / "assets" / "cinematic.mp4").is_file()
    assert (REPO / "evidence" / "assets" / "blender" / "blender_still.png").is_file()
    assert (REPO / "evidence" / "disclaimer.html").is_file()
    assert (REPO / "evidence" / "awaiting-clinical-validation.html").is_file()


def test_clinical_site_is_static_and_honest() -> None:
    index_path = REPO / "clinical" / "index.html"
    assert index_path.is_file()
    index = index_path.read_text(encoding="utf-8")
    lowered = index.lower()
    assert "research" in lowered
    assert "disclaimer" in lowered
    assert "not a medical device" in lowered
    assert "clinical decision support" in lowered
    assert "not" in lowered
    assert "falsif" in lowered
    assert "critique" in lowered or "mentor" in lowered
    assert "DISCLAIMER.md" in index
    assert "AWAITING_CLINICAL_VALIDATION.md" in index
    assert "github.com/cloudynirvana/project-confluence" in lowered
    assert "observation" in lowered and "decision" in lowered
    assert "simulated infusion" in lowered
    assert "mapped endpoints" in lowered
    assert "what this is not" in lowered
    compact = " ".join(lowered.split())
    assert "not a completed tcga" in compact or "not a completed tcga (gdc)" in compact
    for phrase in (
        "this is a cure",
        "we cured",
        "cured cancer",
        "fda-approved",
        "fda-ready",
        "phase ii result",
        "clinically validated treatment",
        "tcga retrospective validation is complete",
        "tcga validation is complete",
        "completed tcga validation",
    ):
        assert phrase not in lowered
    vercel = (REPO / "clinical" / "vercel.json").read_text(encoding="utf-8")
    assert '"outputDirectory": "."' in vercel
    assert not (REPO / "clinical" / "public").exists()
    assert (REPO / "clinical" / "README.md").is_file()
    assert (REPO / "clinical" / "styles.css").is_file()
    assert "<img" not in index.lower()
    assert "<video" not in index.lower()
    assert "fruitfly.xml" not in lowered


def test_hosting_docs_split_vercel_and_container() -> None:
    hosting = (REPO / "docs" / "HOSTING.md").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Root Directory" in readme and "clinical" in readme
    assert "Deploy the clinical briefing to Vercel" in readme
    assert "docs/HOSTING.md" in readme
    assert "Root Directory" in hosting and "`clinical`" in hosting
    assert "Railway" in hosting and "Fly" in hosting
    assert "CONFLUENCE_CORS_ORIGINS" in hosting
    assert "free" in hosting.lower() and "sleep" in hosting.lower()
    assert "not a medical device" in hosting.lower() or "research" in hosting.lower()
    assert "serverless" in hosting.lower()
    assert "DISCLAIMER.md" in hosting
    assert "AWAITING_CLINICAL_VALIDATION.md" in hosting
    for secret_name in ("CAVE_TOKEN=", "FLYWIRE_TOKEN=", "sk-"):
        assert secret_name not in hosting


def test_dockerfile_is_lightweight_web_target() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "uvicorn confluence.telemetry.websocket_server:app" in dockerfile
    assert "--host 0.0.0.0" in dockerfile
    assert "${PORT:-8765}" in dockerfile
    assert "/health" in dockerfile
    assert "FROM python:" in dockerfile
    assert "pip install --no-cache-dir ." in dockerfile
    runs = "\n".join(ln for ln in dockerfile.splitlines() if ln.strip().startswith("RUN")).lower()
    assert "flybody" not in runs
    assert "mujoco" not in runs
    mesh = (REPO / "Dockerfile.mesh").read_text(encoding="utf-8")
    assert "FROM confluence-sim" in mesh
    assert "flybody" in mesh.lower()
    ignore = (REPO / ".dockerignore").read_text(encoding="utf-8")
    assert "evidence" in ignore
    assert (REPO / "fly.toml").is_file()
    assert (REPO / "railway.toml").is_file()
    assert (REPO / "Procfile").is_file()
    proc = (REPO / "Procfile").read_text(encoding="utf-8")
    assert "websocket_server:app" in proc


def test_health_and_cors_without_flybody() -> None:
    from confluence.telemetry.websocket_server import app, health

    body = asyncio.run(health())
    assert body["status"] == "ok"
    assert body["research_only"] is True
    assert body["embodiment"]["available"] in {True, False}
    if not body["embodiment"]["available"]:
        assert body["embodiment"]["backend"] == "unavailable"
    names = [getattr(m, "cls", type("x", (), {"__name__": ""})).__name__ for m in app.user_middleware]
    assert "CORSMiddleware" in names


def test_websocket_server_has_no_hard_flybody_import() -> None:
    src = (REPO / "confluence" / "telemetry" / "websocket_server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "flybody" not in imported
    assert "mujoco" not in imported
