# Lab reel (optional Vercel site, static)

Cinematic flybody / connectome HUD for computational reviewers.
**SIMULATION / RESEARCH.**

This folder is a static export (HTML/CSS + stills/films). It does **not** host
the FastAPI + WebSocket interactive session. Do not configure Vercel as a
Python/serverless runtime for `confluence.telemetry.websocket_server`.

The **default public face** for teaching-hospital mentors is
[`clinical/`](../clinical/). Point production **Root Directory** at `clinical`.
Keep this reel only if you want a second Vercel project for the films.

Public thesis evidence (Google Scholar): `/thesis` and `/thesis.pdf`.
Owner checklist: [docs/SCHOLAR_INDEXING.md](../docs/SCHOLAR_INDEXING.md).

## Deploy as a second Vercel project

One-click:
https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fcloudynirvana%2Fproject-confluence&root-directory=evidence&project-name=confluence-evidence

Dashboard:

1. Import [cloudynirvana/project-confluence](https://github.com/cloudynirvana/project-confluence).
2. Set **Root Directory** to `evidence` (second project — not production default).
3. Framework Preset: **Other**. Leave the build command empty.
4. Output Directory: `.` (override if Vercel suggests `public`). Films are in `assets/`, not a Vercel `public/` output root.
5. No environment secrets are required. Optional: `XAI_API_KEY` only if you want the thesis Grok auditor. Without it the ledger still loads and the auditor soft-fails offline. Never commit an xAI key.

See the README section **Deploy the clinical briefing to Vercel** and [docs/HOSTING.md](../docs/HOSTING.md) for Railway / Fly.

## Local preview

```bash
cd evidence
python -m http.server 4174
# open http://127.0.0.1:4174
```

Assets in `assets/` are copies of `docs/demo/` and `docs/demo/blender/renders/`.
