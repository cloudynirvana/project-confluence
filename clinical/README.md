# Clinical briefing (Vercel, static)

The **public-facing** mentor page for Project Confluence. Written for a
60-second read by a haematologist, surgeon, or pathologist.
**SIMULATION / RESEARCH.** Not a medical device. Not clinical decision support.

This folder is static HTML/CSS. It does **not** host the FastAPI + WebSocket
interactive session. Do not configure Vercel as a Python/serverless runtime
for `confluence.telemetry.websocket_server`.

The cinematic flybody / connectome HUD remains in [`evidence/`](../evidence/)
as an optional second static site (“lab reel”), linked from this page.

## Deploy to Vercel (shareable `*.vercel.app` URL)

Set the project **Root Directory** to `clinical`.

One-click:
https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fcloudynirvana%2Fproject-confluence&root-directory=clinical&project-name=confluence-clinical

Dashboard:

1. Import [cloudynirvana/project-confluence](https://github.com/cloudynirvana/project-confluence).
2. Set **Root Directory** to `clinical`.
3. Framework Preset: **Other**. Leave the build command empty.
4. Output Directory: `.` (override if Vercel suggests `public`).
5. No environment secrets are required.

See the README section **Deploy the clinical briefing to Vercel** and
[docs/HOSTING.md](../docs/HOSTING.md). To keep the cinematic reel live,
create a **second** Vercel project with Root Directory `evidence`.

## Local preview

```bash
cd clinical
python -m http.server 4173
# open http://127.0.0.1:4173
```
