# Evidence site (Vercel, static)

Static research evidence for Project Confluence. **SIMULATION / RESEARCH.**

This folder is a static export (HTML/CSS + stills/films). It does **not** host
the FastAPI + WebSocket interactive session. Do not configure Vercel as a
Python/serverless runtime for `confluence.telemetry.websocket_server`.

## Local preview

```bash
cd evidence
python -m http.server 4173
# open http://127.0.0.1:4173
```

## Deploy to Vercel

1. Import [cloudynirvana/project-confluence](https://github.com/cloudynirvana/project-confluence).
2. Set **Root Directory** to `evidence`.
3. Framework Preset: **Other** (static). Leave build empty; `vercel.json` already sets `outputDirectory` to `.`.
4. No environment secrets are required.

See the README section **Deploy evidence to Vercel** and [docs/HOSTING.md](../docs/HOSTING.md) for the separate Railway / Fly sim. Optional flybody mesh is `Dockerfile.mesh`, not the default image.

Assets in `public/` are copies of `docs/demo/` and `docs/demo/blender/renders/`.
