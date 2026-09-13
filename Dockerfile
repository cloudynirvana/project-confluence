# syntax=docker/dockerfile:1
# Lightweight FastAPI + WebSocket interactive UI.
# numpy / scipy / fastapi only — MuJoCo and flybody are NOT required to boot.
#
#   docker build -t confluence-sim .
#   docker run --rm -p 8765:8765 confluence-sim
#
# Optional fruitfly.xml image: see Dockerfile.mesh
#
# Research simulation only. Not a medical device.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8765 \
    MUJOCO_GL=osmesa \
    CONFLUENCE_CORS_ORIGINS=*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY confluence ./confluence

RUN pip install --no-cache-dir .

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=8s --start-period=25s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8765'))"

# Shell form so Railway/Fly ${PORT} is honored. Do not use a Vercel serverless CMD.
CMD ["sh", "-c", "uvicorn confluence.telemetry.websocket_server:app --host 0.0.0.0 --port ${PORT:-8765}"]
