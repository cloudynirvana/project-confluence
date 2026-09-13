# Evidence site (Vercel, static)

Static research evidence for Project Confluence. **SIMULATION / RESEARCH.**

This folder is a static export (HTML/CSS + stills/films). It does **not** host
the FastAPI + WebSocket interactive session. Do not configure Vercel as a
Python/serverless runtime for `confluence.telemetry.websocket_server`.

## Deploy to Vercel (shareable `*.vercel.app` URL)

One-click:
https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fcloudynirvana%2Fproject-confluence&root-directory=evidence&project-name=confluence-evidence

Dashboard:

1. Import [cloudynirvana/project-confluence](https://github.com/cloudynirvana/project-confluence).
2. Set **Root Directory** to `evidence`.
3. Framework Preset: **Other**. Leave the build command empty.
4. Output Directory: `.` (override if Vercel suggests `public`). Films are in `assets/`, not a Vercel `public/` output root.
5. No environment secrets are required.

See the README section **Deploy evidence to Vercel** and [docs/HOSTING.md](../docs/HOSTING.md) for Railway / Fly.

## Local preview

```bash
cd evidence
python -m http.server 4173
# open http://127.0.0.1:4173
```

Assets in `assets/` are copies of `docs/demo/` and `docs/demo/blender/renders/`.
