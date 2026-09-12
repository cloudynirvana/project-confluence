"""FastAPI + WebSocket interactive session.

Start:
    python -m confluence
    # or
    uvicorn confluence.telemetry.websocket_server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from confluence.cancer_env.archetypes import ARCHETYPES, DISPLAY_NAMES
from confluence.controllers import make_controller
from confluence.embodiment.flybody_bridge import flybody_status
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.toxicity_constraints import load_drug_catalog
from confluence.telemetry.serializers import frame_to_dict

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Confluence v2",
    description="Interactive fly-MB / cancer-microenvironment simulation (research only).",
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/catalog")
async def catalog():
    return [d.model_dump() for d in load_drug_catalog()]


@app.get("/api/archetypes")
async def archetypes():
    return [
        {"id": key, "name": DISPLAY_NAMES[key], "notes": ARCHETYPES[key]().notes}
        for key in ARCHETYPES
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "confluence-v2", "embodiment": flybody_status()}


@app.get("/api/embodiment")
async def embodiment_info():
    return flybody_status()


class LiveSession:
    def __init__(self):
        self.controller_id = "E"
        self.mode = "both"  # cancer | embodiment | both
        self.sim = ClosedLoopSimulator(
            archetype="glioblastoma",
            controller=make_controller("E", n_kc=256, seed=7),
            dt=0.25,
            seed=7,
        )
        self.running = False
        self.hz = 12.0

    def _flags(self):
        return {
            "run_cancer": self.mode in {"cancer", "both"},
            "run_embodiment": self.mode in {"embodiment", "both"},
        }

    def reset(self, archetype: Optional[str] = None, controller: Optional[str] = None, seed: int = 7):
        if controller:
            self.controller_id = controller
            kwargs = {}
            if controller in {"D", "E"}:
                kwargs = {"n_kc": 256, "seed": seed}
            self.sim.set_controller(make_controller(controller, **kwargs))
        return self.sim.reset(archetype=archetype, seed=seed)


@app.websocket("/ws/sim")
async def sim_socket(ws: WebSocket):
    await ws.accept()
    session = LiveSession()
    frame = session.sim.history[-1] if session.sim.history else session.sim.reset()
    await ws.send_json({
        "type": "hello",
        "frame": frame_to_dict(frame),
        "controller": session.controller_id,
        "mode": session.mode,
        "embodiment": flybody_status(),
    })
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0 / session.hz)
            except asyncio.TimeoutError:
                msg = None
            except WebSocketDisconnect:
                break

            if msg:
                cmd = msg.get("cmd")
                if cmd == "play":
                    session.running = True
                elif cmd == "pause":
                    session.running = False
                elif cmd == "step":
                    frame = session.sim.step(**session._flags())
                    payload = frame_to_dict(frame)
                    payload["mode"] = session.mode
                    await ws.send_json(payload)
                elif cmd == "reset":
                    frame = session.reset(
                        archetype=msg.get("archetype"),
                        controller=msg.get("controller"),
                        seed=int(msg.get("seed", 7)),
                    )
                    session.running = False
                    await ws.send_json(frame_to_dict(frame))
                elif cmd == "set_controller":
                    session.reset(controller=msg.get("controller", "E"))
                    session.running = False
                    await ws.send_json(frame_to_dict(session.sim.history[-1]))
                elif cmd == "set_archetype":
                    session.reset(archetype=msg.get("archetype", "glioblastoma"))
                    session.running = False
                    await ws.send_json(frame_to_dict(session.sim.history[-1]))
                elif cmd == "override":
                    session.sim.set_manual(bool(msg.get("enabled", False)), msg.get("U"))
                elif cmd == "set_speed":
                    session.sim.dt = float(msg.get("dt", 0.25))
                    session.hz = float(msg.get("hz", session.hz))
                elif cmd == "set_mode":
                    mode = str(msg.get("mode", "both"))
                    if mode in {"cancer", "embodiment", "both"}:
                        session.mode = mode
                    await ws.send_json({
                        "type": "mode",
                        "mode": session.mode,
                        "embodiment": flybody_status(),
                    })
                elif cmd == "ping":
                    await ws.send_json({"type": "pong"})

            if session.running:
                frame = session.sim.step(**session._flags())
                payload = frame_to_dict(frame)
                payload["mode"] = session.mode
                await ws.send_json(payload)
                if frame.terminal:
                    session.running = False
                    await ws.send_json({"type": "halted", "reason": "terminal_toxicity"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            return
