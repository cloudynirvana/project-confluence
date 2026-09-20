# Hosting Project Confluence

**SIMULATION / RESEARCH.** A public URL is still a research demo. It is not a
medical device, not clinical decision support, and not FDA / EMA / Phase II
evidence. Link [DISCLAIMER.md](../DISCLAIMER.md) and
[AWAITING_CLINICAL_VALIDATION.md](AWAITING_CLINICAL_VALIDATION.md) from any
page you put in front of reviewers.

There are **three deployables**. Do not collapse the interactive sim onto Vercel serverless.

| Deployable | Where | What it is |
|------------|--------|------------|
| Clinical briefing | **Vercel** (static) | 60-second mentor page. **Root Directory `clinical`** (production public face). |
| Lab reel + thesis evidence | **Vercel** (static + one serverless function) | Flybody / cinematic HUD, thinking lab, thesis evidence auditor. Root Directory `evidence`. |
| Interactive sim | **Railway or Fly.io** (container / dyno) | Long-lived FastAPI + WebSockets. |

The interactive session is `uvicorn confluence.telemetry.websocket_server:app`.
It needs a process that stays up and speaks HTTP **and** WebSockets. That is
not a Vercel serverless function.

## 1. Clinical briefing on Vercel (public face)

**One-line production change:** if an existing project still uses Root Directory `evidence`, set it to `clinical` and redeploy.

**One-click:** [Deploy clinical briefing on Vercel](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fcloudynirvana%2Fproject-confluence&root-directory=clinical&project-name=confluence-clinical)

Dashboard (import this GitHub repo — no secrets):

1. Open [vercel.com/new](https://vercel.com/new) and import `cloudynirvana/project-confluence`.
2. Set **Root Directory** to `clinical`.
3. Framework Preset: **Other**. Leave the build command empty.
   Output Directory: `.` — not `public`.
   `clinical/vercel.json` sets `framework: null` and `outputDirectory: "."`.
4. Deploy. No environment variables.

There is no in-repo Vercel preview URL until the owner connects the GitHub app and deploys. The first production URL will look like `https://confluence-clinical.vercel.app`.

### Optional lab reel + thesis evidence (Root Directory `evidence`)

Deploy a **second** project with Root Directory `evidence`. One-click: [Deploy lab reel](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fcloudynirvana%2Fproject-confluence&root-directory=evidence&project-name=confluence-evidence). `evidence/vercel.json` already sets `framework: null` and `outputDirectory: "."`.

Static routes on that project:

- `/` — cinematic lab reel
- `/thinking` — disease-specific thinking lab (not a Scholar article URL)
- `/thesis` — thesis evidence page + ledger (Highwire `citation_*` tags)
- `/thesis.pdf` — citeable thesis PDF (same directory as `/thesis`; `Content-Type: application/pdf`)
- `/robots.txt`, `/sitemap.xml` — allow Googlebot / Googlebot-Scholar on `/thesis` and `/thesis.pdf`
- `GET`/`POST /api/grok-review` — server-side Grok auditor (`evidence/api/grok-review.js`)

Scholar owner checklist: [SCHOLAR_INDEXING.md](SCHOLAR_INDEXING.md).

**Environment variable (evidence project only):** set `XAI_API_KEY` in the Vercel dashboard for Preview and Production. The key must never be committed, pasted into HTML, or shipped in client JavaScript. Without it, `/thesis` still renders the ledger. The auditor **soft-fails** with `{ ok:false, reason:"auditor_offline", message }` and the page shows *Evidence auditor offline (no API key) — citations still load from ledger* instead of a dead audit button. Do not add an xAI key to the repository.

Local preview:

```bash
cd clinical && python -m http.server 4173
# from repo root, optional lab reel / thesis:
cd evidence && python -m http.server 4174
# Grok auditor requires `vercel dev` (or equivalent) plus XAI_API_KEY
```

If you later add a “Try the live sim” button, point it at the Railway/Fly
origin (HTTPS → `wss`). Do not proxy `/ws/sim` through Vercel serverless.

## 2. Interactive sim on Railway

**One-click (GitHub → Railway, no secrets):**
[Deploy on Railway](https://railway.app/new?template=https://github.com/cloudynirvana/project-confluence)

Dashboard:

1. New project → deploy from the GitHub repo.
2. Railway detects `railway.toml` + `Dockerfile` (numpy / scipy / fastapi —
   **no MuJoCo / flybody**). Mesh is `Dockerfile.mesh` only. Do not change the
   Dockerfile path to `Dockerfile.mesh` unless you want the heavier image.
3. Railway injects `PORT`. The image already runs
   `uvicorn … --host 0.0.0.0 --port ${PORT:-8765}`.
4. Health check: `GET /health` (configured in `railway.toml`).
5. Optional Nixpacks fallback: `Procfile` has the same `web:` process.

No API keys are required for the demo. Do not commit `.env` files.

## 3. Interactive sim on Fly.io

**One-click-ish (Fly CLI, no secrets in git):**

```bash
# once, from a clone
fly launch --no-deploy --copy-config --yes
fly deploy
```

`fly.toml` is already in the repo. `fly launch` will rewrite the placeholder
`app` name. Do not add API tokens to the repository.

`fly.toml` publishes `internal_port = 8765`, HTTPS, and `GET /health`.
`auto_stop_machines = "stop"` means a free/hobby machine **will sleep**.
The first WebSocket after idle can take tens of seconds.

```bash
fly status
curl -fsS "https://<your-app>.fly.dev/health"
```

## 4. Docker (any host)

```bash
docker build -t confluence-sim .
docker run --rm -p 8765:8765 -e PORT=8765 confluence-sim
curl -fsS http://127.0.0.1:8765/health
```

Optional fruitfly.xml viewport (heavier image; not required for the UI to boot).
Kept in a **separate** `Dockerfile.mesh` so Railway / Fly never clone flybody
on the default path:

```bash
docker build -t confluence-sim .
docker build -f Dockerfile.mesh -t confluence-sim:mesh .
```

`.dockerignore` keeps v1 `models/`, `docs/`, `evidence/`, and demo films out
of the build context so the default image stays small.

## Environment variables

Do not put tokens in the repo.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8765` | Listen port (Railway/Fly set this). |
| `CONFLUENCE_CORS_ORIGINS` | `*` | Comma-separated browser origins allowed to call HTTP APIs (`/health`, `/api/*`). Use your Vercel origin if the clinical briefing or lab reel fetches the sim. |
| `CONFLUENCE_SKIP_WARMUP` | unset | Set `1` to skip building a session on process start (faster boot; first WebSocket constructs the loop). |
| `MUJOCO_GL` | `osmesa` | Only relevant for the optional `mesh` image. |
| `CONFLUENCE_PPO_CKPT` | unset | Optional torch checkpoint for controller C. Not needed for the demo. |
| `XAI_API_KEY` | unset | **Evidence Vercel project only.** Server-side Grok auditor at `POST /api/grok-review`. Never expose to the browser. |

No `CAVE_TOKEN` / `FLYWIRE_TOKEN` is required. The connectome stays on the
structured stub unless you add credentials yourself (keep them in the host’s
secret store, never in git).

## CORS and WebSockets

- The **interactive UI is served by the same FastAPI process** (`GET /` +
  `GET /static/*` + `WS /ws/sim`). Same-origin sockets do not need CORS.
- Browser **HTTP** calls from another origin (for example the Vercel clinical
  briefing hitting `/health`) honor `CONFLUENCE_CORS_ORIGINS`.
- **WebSocket** handshakes are not fully covered by Starlette CORS middleware.
  Keep the live canvas on the Railway/Fly origin, or terminate TLS on a proxy
  that forwards `Upgrade: websocket` to the container.
- On HTTPS public URLs the page uses `wss://` automatically
  (`confluence/telemetry/static/app.js`).
- Fly: `http_service.http_options.idle_timeout = 300` so a paused socket is
  less likely to be cut at 60s. Railway’s edge also supports WebSockets on
  the `web` process; do not enable a “serverless” worker for this app.

## Free-tier caveats

Hobby Fly machines and idle Railway services **sleep**. A public reviewer
hitting the URL after idle will wait for a cold start. That is acceptable for
a research demo; it is not acceptable as implied clinical availability.

## Public-URL copy

Any hosted sim page already says “research · not clinical” in the HUD. Still:

- Do not title the deployment “FDA-ready,” “Phase II,” or “treatment planner.”
- Link [DISCLAIMER.md](../DISCLAIMER.md) and
  [AWAITING_CLINICAL_VALIDATION.md](AWAITING_CLINICAL_VALIDATION.md) from the
  clinical briefing (already linked), the optional evidence lab reel, and the
  repo README.
- Simulated burden / resistance / protein channels remain ODE research scores.
- The thesis auditor classifies claims. It does not prescribe treatment.

## Local equivalent (no container)

```bash
pip install -e .
python -m confluence --host 0.0.0.0 --port 8765
```
