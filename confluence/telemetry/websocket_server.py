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
from confluence.contracts import (
    DEMO_BRAIN_NEURONS,
    FULL_BRAIN_NEURONS,
    INTERACTIVE_BRAIN_NEURONS,
    PROTEIN_CHANNEL_IDS,
)
from confluence.controllers import make_controller
from confluence.controllers.full_brain import FullBrainController
from confluence.embodiment.flybody_bridge import flybody_status
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.toxicity_constraints import load_drug_catalog
from confluence.telemetry.serializers import frame_to_dict
from confluence.training.loop import run_live_episode

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
    return {
        "status": "ok",
        "service": "confluence-v2",
        "embodiment": flybody_status(),
        "full_brain_neurons": FULL_BRAIN_NEURONS,
        "protein_channels": list(PROTEIN_CHANNEL_IDS),
    }


@app.get("/api/embodiment")
async def embodiment_info():
    return flybody_status()


class LiveSession:
    BRAIN_MODES = {
        "demo": {"controller": "E", "n_neurons": DEMO_BRAIN_NEURONS, "n_kc": DEMO_BRAIN_NEURONS},
        "train": {"controller": "F", "n_neurons": INTERACTIVE_BRAIN_NEURONS},
        "full": {"controller": "F", "n_neurons": FULL_BRAIN_NEURONS},
    }

    def __init__(self):
        self.controller_id = "E"
        self.brain_mode = "demo"
        self.n_neurons = DEMO_BRAIN_NEURONS
        self.mode = "both"  # cancer | embodiment | both
        self.sim = ClosedLoopSimulator(
            archetype="glioblastoma",
            controller=make_controller("E", n_kc=DEMO_BRAIN_NEURONS, seed=7),
            dt=0.25,
            seed=7,
        )
        self.running = False
        self.hz = 12.0
        self.training = {
            "episodes": 0,
            "last_reward": 0.0,
            "last_da": 0.0,
            "last_burden": 0.0,
            "last_toxicity": 0.0,
            "protein_active": [],
            "best_reward": None,
        }

    def _flags(self):
        return {
            "run_cancer": self.mode in {"cancer", "both"},
            "run_embodiment": self.mode in {"embodiment", "both"},
        }

    def _controller_kwargs(self, controller: str, seed: int) -> dict:
        if controller in {"D", "E"}:
            n_kc = DEMO_BRAIN_NEURONS if self.brain_mode == "demo" else min(self.n_neurons, 2048)
            return {"n_kc": n_kc, "seed": seed}
        if controller in {"F", "full_brain"}:
            return {"n_neurons": self.n_neurons, "seed": seed}
        return {}

    def reset(self, archetype: Optional[str] = None, controller: Optional[str] = None, seed: int = 7):
        if controller:
            self.controller_id = controller
            self.sim.set_controller(make_controller(controller, **self._controller_kwargs(controller, seed)))
        return self.sim.reset(archetype=archetype, seed=seed)

    def set_brain_mode(self, mode: str, seed: int = 7):
        if mode not in self.BRAIN_MODES:
            raise ValueError(f"unknown brain mode '{mode}'")
        spec = self.BRAIN_MODES[mode]
        self.brain_mode = mode
        self.n_neurons = int(spec["n_neurons"])
        self.controller_id = spec["controller"]
        self.sim.set_controller(
            make_controller(self.controller_id, **self._controller_kwargs(self.controller_id, seed))
        )
        return self.sim.reset(seed=seed)

    def train_episode(self, days: float = 80.0) -> dict:
        if not isinstance(self.sim.controller, FullBrainController):
            self.set_brain_mode("train" if self.n_neurons < FULL_BRAIN_NEURONS else "full")
        metrics = run_live_episode(self.sim, days=days)
        self.training["episodes"] += 1
        self.training["last_reward"] = metrics["reward"]
        self.training["last_da"] = metrics["mean_da"]
        self.training["last_burden"] = metrics["mean_burden"]
        self.training["last_toxicity"] = metrics["mean_toxicity"]
        self.training["protein_active"] = list(metrics["protein_active"])
        best = self.training["best_reward"]
        if best is None or metrics["reward"] > best:
            self.training["best_reward"] = metrics["reward"]
        return {**metrics, **self.training, "n_neurons": self.n_neurons, "brain_mode": self.brain_mode}

    def session_meta(self) -> dict:
        return {
            "controller": self.controller_id,
            "mode": self.mode,
            "brain_mode": self.brain_mode,
            "n_neurons": self.n_neurons,
            "protein_channels": list(PROTEIN_CHANNEL_IDS),
            "training": dict(self.training),
            "embodiment": flybody_status(),
        }


@app.websocket("/ws/sim")
async def sim_socket(ws: WebSocket):
    await ws.accept()
    session = LiveSession()
    frame = session.sim.history[-1] if session.sim.history else session.sim.reset()
    await ws.send_json({
        "type": "hello",
        "frame": frame_to_dict(frame),
        **session.session_meta(),
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
                    payload.update(session.session_meta())
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
                elif cmd == "set_brain_mode":
                    session.running = False
                    frame = session.set_brain_mode(str(msg.get("mode", "demo")))
                    payload = frame_to_dict(frame)
                    payload.update({"type": "brain_mode", **session.session_meta()})
                    await ws.send_json(payload)
                elif cmd == "train_episode":
                    session.running = False
                    days = float(msg.get("days", 80.0))
                    metrics = await asyncio.to_thread(session.train_episode, days)
                    frame = session.sim.history[-1]
                    payload = frame_to_dict(frame)
                    payload.update({"type": "train_result", **session.session_meta(), "metrics": metrics})
                    await ws.send_json(payload)
                elif cmd == "ping":
                    await ws.send_json({"type": "pong"})

            if session.running:
                frame = session.sim.step(**session._flags())
                payload = frame_to_dict(frame)
                payload.update(session.session_meta())
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
