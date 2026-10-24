"""Web server: the control page (/), the window display (/display), a JSON API and a websocket."""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import settings as settings_mod
from .engine import IMAGES_DIR, Engine
from .generator import MfluxBackend
from .glass import make_glass
from .settings import DATA_DIR

STATIC = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Belle Verrière")
    clients: set[WebSocket] = set()
    loop_holder: dict = {}
    changed = {"event": asyncio.Event()}  # set (and replaced) on every broadcast; wakes long polls

    def emit(msg: dict):
        loop = loop_holder.get("loop")
        if loop:
            asyncio.run_coroutine_threadsafe(_broadcast(msg), loop)

    async def _broadcast(msg: dict):
        changed["event"].set()
        changed["event"] = asyncio.Event()
        if msg.get("type") == "status":
            msg = {"type": "status", **engine.status_dict()}
        data = json.dumps(msg)
        for ws in list(clients):
            try:
                await ws.send_text(data)
            except Exception:
                clients.discard(ws)

    s = settings_mod.load()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = Engine(s, MfluxBackend(DATA_DIR), emit)
    app.state.engine = engine

    @app.on_event("startup")
    async def _start():
        loop_holder["loop"] = asyncio.get_running_loop()
        threading.Thread(target=engine.run, daemon=True, name="engine").start()

    # ---- pages
    @app.get("/")
    async def control_page():
        return FileResponse(STATIC / "control.html")

    @app.get("/display")
    async def display_page():
        return FileResponse(STATIC / "display.html")

    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

    # ---- API
    @app.get("/api/state")
    async def state():
        return {"settings": asdict(engine.settings), **engine.status_dict()}

    def _display() -> dict:
        d = {**engine.current_message(), **engine.display_config()}
        key = json.dumps([d.get("type"), d.get("url"), d.get("crossfade"), d.get("rotation")])
        d["token"] = hashlib.sha1(key.encode()).hexdigest()[:12]
        return d

    @app.get("/api/display")
    async def display_state(seen: str = ""):
        """What the window should show, for clients that poll (the Roku app). Long polling: pass
        the last answer's token as ?seen= and this waits, up to 25s, until something changes."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 25
        while True:
            wake = changed["event"]  # taken before reading, so a change in between isn't missed
            d = _display()
            remaining = deadline - loop.time()
            if d["token"] != seen or remaining <= 0:
                return d
            try:
                await asyncio.wait_for(wake.wait(), remaining)
            except asyncio.TimeoutError:
                pass

    @app.post("/api/settings")
    async def update_settings(changes: dict):
        new, changed = settings_mod.update(engine.settings, changes)
        await asyncio.to_thread(engine.apply_settings, new, changed)
        await _broadcast({"type": "status"})
        return {"changed": sorted(changed), "settings": asdict(new)}

    @app.get("/api/history")
    async def history(limit: int = 60, starred: bool = False):
        items = [e for e in engine.history if e.get("starred") or not starred]
        return list(reversed(items[-limit:]))

    @app.post("/api/action/{name}")
    async def action(name: str, body: dict | None = None):
        body = body or {}
        try:
            if name == "next":
                engine.next_now()
            elif name == "pause":
                await update_settings({"paused": True})
            elif name == "resume":
                await update_settings({"paused": False})
            elif name == "show":
                await asyncio.to_thread(engine.show_entry, body["id"])
            elif name == "restart_from":
                engine.restart_from(body["id"])
            elif name == "reset_drift":
                engine.reset_drift()
            elif name == "star":
                engine.star(body["id"], bool(body.get("value", True)))
            elif name == "surprise":
                idea = await asyncio.to_thread(engine.invent)
                await update_settings({"base_prompt": idea["idea"], "drift_guidance": idea["rules"]})
                engine.next_now()
                return {"ok": True, **idea}
            elif name == "delete":
                engine.delete(body["id"])
            elif name == "preset":
                preset = settings_mod.PRESETS[body["name"]]
                await update_settings({**preset, "preset": body["name"]})
            else:
                raise HTTPException(404, f"unknown action {name}")
        except KeyError as e:
            raise HTTPException(404, f"no such image {e}")
        except ValueError as e:
            raise HTTPException(409, str(e))
        await _broadcast({"type": "status"})
        return {"ok": True}

    @app.post("/api/preview")
    async def preview(changes: dict):
        """Render the current image with trial glass settings without saving them."""
        if not engine.current:
            raise HTTPException(409, "nothing generated yet")
        trial, _ = settings_mod.update(engine.settings, changes)
        e = engine.current

        def render():
            from PIL import Image
            img = make_glass(Image.open(IMAGES_DIR / e["raw"]), out_size=(trial.out_width, trial.out_height),
                             frame=trial.frame, strength=trial.glass_strength, colors=trial.colors,
                             cell_smoothing=trial.cell_smoothing, lead_width=trial.lead_width,
                             saturation=trial.saturation, contrast=trial.contrast, texture=trial.texture,
                             saddle_bars=trial.saddle_bars, trace_leads=trial.trace_leads,
                             shard_smoothing=trial.shard_smoothing, seed=e["seed"])
            img.save(IMAGES_DIR / "preview.jpg", quality=90)
        await asyncio.to_thread(render)
        await _broadcast({"type": "show", "url": f"/images/preview.jpg?t={asyncio.get_running_loop().time()}",
                          "crossfade": 1.5, "entry": None})
        return {"ok": True}

    @app.get("/api/llm_models")
    async def llm_models():
        return await asyncio.to_thread(engine.drifter.list_models)

    # ---- live updates
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        try:
            await websocket.send_text(json.dumps({"type": "config", **engine.display_config()}))
            await websocket.send_text(json.dumps(engine.current_message()))
            await websocket.send_text(json.dumps({"type": "status", **engine.status_dict()}))
            while True:
                await websocket.receive_text()  # keepalive pings from the pages
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    return app
