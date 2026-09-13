# Hosting Project Confluence

**SIMULATION / RESEARCH.** A public URL is still a research demo. It is not a
medical device, not clinical decision support, and not FDA / EMA / Phase II
evidence. Link [DISCLAIMER.md](../DISCLAIMER.md) and
[AWAITING_CLINICAL_VALIDATION.md](AWAITING_CLINICAL_VALIDATION.md) from any
page you put in front of reviewers.

There are **two deployables**. Do not collapse them onto Vercel serverless.

| Deployable | Where | What it is |
|------------|--------|------------|
| Evidence site | **Vercel** (static) | Stills, films, honesty copy. Root directory `evidence`. |
| Interactive sim | **Railway or Fly.io** (container / dyno) | Long-lived FastAPI + WebSockets. |

The interactive session is `uvicorn confluence.telemetry.websocket_server:app`.
It needs a process that stays up and speaks HTTP **and** WebSockets. That is
not a Vercel serverless function.

## 1. Evidence site on Vercel

**One-click:** [Deploy evidence on Vercel](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fcloudynirvana%2Fproject-confluence&root-directory=evidence&project-name=confluence-evidence)

Dashboard (import this GitHub repo — no secrets):

1. Open [vercel.com/new](https://vercel.com/new) and import `cloudynirvana/project-confluence`.
2. Set **Root Directory** to `evidence`.
3. Framework Preset: **Other**. Leave the build command empty.
   Output Directory: `.` — not `public`. Stills live in `evidence/assets/`.
   `evidence/vercel.json` sets `framework: null` and `outputDirectory: "."`.
4. Deploy. No environment variables.

There is no in-repo Vercel preview URL until the owner connects the GitHub app and deploys. The first production URL will look like `https://confluence-evidence.vercel.app`.

Local preview:

```bash
cd evidence
python -m http.server 4173
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

None are secrets. Do not put tokens in the repo.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8765` | Listen port (Railway/Fly set this). |
| `CONFLUENCE_CORS_ORIGINS` | `*` | Comma-separated browser origins allowed to call HTTP APIs (`/health`, `/api/*`). Use your Vercel origin if the evidence site fetches the sim. |
| `CONFLUENCE_SKIP_WARMUP` | unset | Set `1` to skip building a session on process start (faster boot; first WebSocket constructs the loop). |
| `MUJOCO_GL` | `osmesa` | Only relevant for the optional `mesh` image. |
| `CONFLUENCE_PPO_CKPT` | unset | Optional torch checkpoint for controller C. Not needed for the demo. |

No `CAVE_TOKEN` / `FLYWIRE_TOKEN` is required. The connectome stays on the
structured stub unless you add credentials yourself (keep them in the host’s
secret store, never in git).

## CORS and WebSockets

- The **interactive UI is served by the same FastAPI process** (`GET /` +
  `GET /static/*` + `WS /ws/sim`). Same-origin sockets do not need CORS.
- Browser **HTTP** calls from another origin (for example the Vercel evidence
  site hitting `/health`) honor `CONFLUENCE_CORS_ORIGINS`.
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
  evidence site (already linked) and from the repo README.
- Simulated burden / resistance / protein channels remain ODE research scores.

## Local equivalent (no container)

```bash
pip install -e .
python -m confluence --host 0.0.0.0 --port 8765
```
