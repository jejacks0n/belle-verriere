"""The main loop: generate the next window while the current one is up, then crossfade to it."""
from __future__ import annotations

import json
import random
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from PIL import Image

from . import daylight
from . import settings as settings_mod
from .drift import Drifter, PromptState, compose, describe
from .generator import GenRequest
from .glass import make_glass
from .settings import DATA_DIR, Settings

IMAGES_DIR = DATA_DIR / "images"
HISTORY_FILE = DATA_DIR / "history.jsonl"
PROMPT_STATE_FILE = DATA_DIR / "prompt_state.json"
MAX_HISTORY = 600  # older, un-starred images are deleted

GLASS_KEYS = {"frame", "glass_strength", "colors", "cell_smoothing", "lead_width", "saturation",
              "contrast", "texture", "saddle_bars", "out_width", "out_height", "trace_leads", "shard_smoothing"}
CONTENT_KEYS = {"style_prompt", "negative_prompt", "model_path", "quantize", "float16_compute", "steps",
                "gen_width", "gen_height", "loras", "shape_strength", "use_controlnet", "controlnet_strength", "controlnet_steps"}


class Engine:
    def __init__(self, settings: Settings, backend, emit: Callable[[dict], None]):
        self.settings = settings
        self.backend = backend
        self.emit = emit
        self.drifter = Drifter(settings.ollama_url, settings.llm_model)
        self.rnd = random.Random()
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stop_flag = False

        self.current: dict | None = None     # history entry on the display
        self.pending: dict | None = None     # generated, waiting its turn
        self.shown_at = 0.0
        self.skip = False
        self.blank = False
        self.status = "starting"
        self.last_error: str | None = None
        self.last_gen_seconds: float | None = None
        self.gen_started: float | None = None  # set while an image is generating
        self.gen_step = (0, 0)                 # (done, total) denoise steps
        self.epoch = 0                        # bumped when queued work becomes stale
        self.invented: list[str] = []         # recent "Surprise me" ideas, so they don't repeat
        self.recent_poses: list[str] = []

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = self._load_history()
        self.prompt = self._load_prompt_state()
        if self.history:
            self.current = self.history[-1]
            self.shown_at = time.time()

    # ---------------------------------------------------------------- persistence
    def _load_history(self) -> list[dict]:
        if not HISTORY_FILE.exists():
            return []
        out = []
        for line in HISTORY_FILE.read_text().splitlines():
            try:
                e = json.loads(line)
                if (IMAGES_DIR / e["file"]).exists():
                    out.append(e)
            except Exception:
                pass
        return out

    def _save_history(self):
        tmp = HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in self.history))
        tmp.replace(HISTORY_FILE)

    def _load_prompt_state(self) -> PromptState:
        try:
            d = json.loads(PROMPT_STATE_FILE.read_text())
            if d.get("base_prompt") == self.settings.base_prompt:
                return PromptState(d["original"], d["current"], d.get("generation", 0), d.get("recent_subjects", []))
        except Exception:
            pass
        return PromptState()

    def _save_prompt_state(self):
        PROMPT_STATE_FILE.write_text(json.dumps({"base_prompt": self.settings.base_prompt, **asdict(self.prompt)}))

    def _prune(self):
        if len(self.history) <= MAX_HISTORY:
            return
        keep_from = len(self.history) - MAX_HISTORY
        survivors = []
        for i, e in enumerate(self.history):
            if i < keep_from and not e.get("starred") and e is not self.current:
                for f in (e["file"], e.get("raw")):
                    if f:
                        (IMAGES_DIR / f).unlink(missing_ok=True)
            else:
                survivors.append(e)
        self.history = survivors

    # ---------------------------------------------------------------- public controls
    def apply_settings(self, new: Settings, changed: set[str]):
        with self.lock:
            self.settings = new
            settings_mod.save(new)
            self.drifter.ollama_url = new.ollama_url.rstrip("/")
            self.drifter.model = new.llm_model
            if "base_prompt" in changed:
                self.prompt = PromptState()
                self._discard_pending()
            elif changed & CONTENT_KEYS or "frame" in changed:  # the prompt describes the frame's shape
                self._discard_pending()
            if changed & GLASS_KEYS:
                for e in (self.current, self.pending):
                    if e:
                        self._render_glass(e)
                if self.current:
                    self._show(self.current, crossfade=4)
            if "display_rotation" in changed or "crossfade_seconds" in changed:
                self.emit({"type": "config", **self.display_config()})
        self.wake.set()

    def _discard_pending(self):
        self.pending = None
        self.epoch += 1

    def next_now(self):
        with self.lock:
            self.skip = True
            self.blank = False
        self.wake.set()

    def show_entry(self, entry_id: str):
        with self.lock:
            e = self._find(entry_id)
            self.blank = False
            self._show(e)

    def restart_from(self, entry_id: str):
        """Make an earlier image's prompt the new starting point: it becomes the idea, the anchor
        the drift is pulled back to, and exactly what the next image is made from."""
        with self.lock:
            slots = dict(self._find(entry_id)["slots"])
            self.settings = replace(self.settings, base_prompt=describe(slots))
            settings_mod.save(self.settings)
            self.prompt = PromptState(dict(slots), dict(slots), 0, [slots["subject"]] if slots.get("subject") else [])
            self._save_prompt_state()
            self._discard_pending()
        self.wake.set()

    def reset_drift(self):
        with self.lock:
            self.prompt.current = dict(self.prompt.original) if self.prompt.original else {}
            self._save_prompt_state()
            self._discard_pending()
        self.wake.set()

    def invent(self) -> dict[str, str]:
        """Ask the LLM for a fresh idea and drift rules (doesn't apply them)."""
        idea = self.drifter.invent(self.rnd, self.invented)
        self.invented = (self.invented + [idea["idea"]])[-8:]
        return idea

    def star(self, entry_id: str, value: bool):
        with self.lock:
            self._find(entry_id)["starred"] = value
            self._save_history()

    def delete(self, entry_id: str):
        """Remove an image and its files for good. The one on the window can't be deleted."""
        with self.lock:
            e = self._find(entry_id)
            if e is self.current:
                raise ValueError("That image is on the window right now")
            for f in (e["file"], e.get("raw")):
                if f:
                    (IMAGES_DIR / f).unlink(missing_ok=True)
            self.history.remove(e)
            self._save_history()

    def _find(self, entry_id: str) -> dict:
        for e in self.history:
            if e["id"] == entry_id:
                return e
        raise KeyError(entry_id)

    # ---------------------------------------------------------------- display
    def display_config(self) -> dict:
        return {"rotation": self.settings.display_rotation, "crossfade": self.settings.crossfade_seconds}

    def current_message(self) -> dict:
        if self.blank or not self.current:
            return {"type": "blank"}
        return {"type": "show", "url": self._url(self.current), "crossfade": 3, "entry": self.current}

    @staticmethod
    def _url(e: dict) -> str:
        return f"/images/{e['file']}?v={e.get('v', 0)}"

    def _show(self, e: dict, crossfade: float | None = None):
        self.current = e
        self.shown_at = time.time()
        self.emit({"type": "show", "url": self._url(e),
                   "crossfade": self.settings.crossfade_seconds if crossfade is None else crossfade,
                   "entry": e})

    def status_dict(self) -> dict:
        s = self.settings
        remaining = None
        if self.current and not s.paused:
            remaining = max(0, int(s.display_seconds - (time.time() - self.shown_at)))
        return {
            "status": self.status, "error": self.last_error, "llm_error": self.drifter.last_error,
            "backend": self.backend.name, "current": self.current, "pending": self.pending is not None,
            "seconds_left": remaining, "last_gen_seconds": self.last_gen_seconds,
            "gen_elapsed": round(time.time() - self.gen_started) if self.gen_started else None,
            "gen_step": self.gen_step[0], "gen_steps": self.gen_step[1],
            "prompt_slots": self.prompt.current, "original_slots": self.prompt.original,
            "daylight": daylight.status(s),
        }

    # ---------------------------------------------------------------- generation
    def _render_glass(self, e: dict):
        s = self.settings
        raw = Image.open(IMAGES_DIR / e["raw"])
        out = make_glass(raw, out_size=(s.out_width, s.out_height), frame=s.frame, strength=s.glass_strength,
                         colors=s.colors, cell_smoothing=s.cell_smoothing, lead_width=s.lead_width,
                         saturation=s.saturation, contrast=s.contrast, texture=s.texture,
                         saddle_bars=s.saddle_bars, trace_leads=s.trace_leads,
                         shard_smoothing=s.shard_smoothing, seed=e["seed"])
        out.save(IMAGES_DIR / e["file"], quality=93)
        e["v"] = e.get("v", 0) + 1

    def _next_prompt(self) -> tuple[dict, str]:
        s = self.settings
        ps = self.prompt
        if not ps.original:
            self.status = "reading your prompt"
            ps.original = self.drifter.decompose(s.base_prompt, s.drift_guidance)
            ps.current = dict(ps.original)
            ps.generation = 0
        elif not ps.current:
            ps.current = dict(ps.original)
        elif s.drift_enabled and ps.generation > 0 and ps.generation % max(1, s.mutate_every) == 0:
            self.status = "drifting the prompt"
            ps.current = self.drifter.mutate(ps, s.drift, s.tether, self.rnd, s.drift_slots, s.drift_guidance)
        ps.generation += 1
        subj = ps.current.get("subject", "")
        if subj and (not ps.recent_subjects or ps.recent_subjects[-1] != subj):
            ps.recent_subjects = (ps.recent_subjects + [subj])[-12:]
        self._save_prompt_state()
        return dict(ps.current), compose(ps.current, s.style_prompt, daylight.time_hint(s), s.frame,
                                         self._next_pose() if s.vary_pose else None)

    def _next_pose(self) -> str:
        """A pose and framing not used in the last few images."""
        from .drift import FRAMINGS, POSES
        pose = self.rnd.choice([p for p in POSES if p not in self.recent_poses])
        self.recent_poses = (self.recent_poses + [pose])[-8:]
        return f"{pose}, {self.rnd.choice(FRAMINGS)}"

    def _generate(self) -> dict | None:
        with self.lock:
            epoch = self.epoch
            s = self.settings
        slots, prompt = self._next_prompt()
        seed = self.rnd.randrange(2**31)
        req = GenRequest(prompt=prompt, negative_prompt=s.negative_prompt, seed=seed,
                         width=s.gen_width // 16 * 16, height=s.gen_height // 16 * 16, steps=s.steps)
        self.status = "generating"
        self.gen_step = (0, req.steps)
        self.gen_started = t0 = time.time()
        self.emit({"type": "status"})
        try:
            img = self.backend.generate(req, s, on_step=self._on_step)
        finally:
            self.gen_started = None
        self.last_gen_seconds = round(time.time() - t0, 1)
        t = round(self.last_gen_seconds)
        print(f"[engine] image generated in {t // 60}:{t % 60:02d}")

        with self.lock:
            if epoch != self.epoch:
                return None  # settings changed mid-generation; this image no longer fits
            eid = time.strftime("%Y%m%d-%H%M%S") + f"-{seed % 10000:04d}"
            e = {"id": eid, "file": f"{eid}.jpg", "raw": f"{eid}_raw.png", "prompt": prompt, "slots": slots,
                 "seed": seed, "time": time.strftime("%Y-%m-%d %H:%M"), "gen_seconds": self.last_gen_seconds,
                 "starred": False, "v": 0}
            img.save(IMAGES_DIR / e["raw"])
            self.status = "leading the glass"
            self._render_glass(e)
            self.history.append(e)
            self._prune()
            self._save_history()
            return e

    def _on_step(self, step: int, total: int):
        self.gen_step = (step, total)
        self.emit({"type": "status"})

    def run(self):
        while not self.stop_flag:
            try:
                self._tick()
            except Exception as ex:
                self.last_error = f"{type(ex).__name__}: {ex}"
                self.status = "error"
                traceback.print_exc()
                self.emit({"type": "status"})
                self.wake.wait(30)
                self.wake.clear()

    def _tick(self):
        s = self.settings
        if not daylight.is_awake(s):
            if not self.blank:
                self.blank = True
                self.emit({"type": "blank"})
            self.status = "sleeping until sunrise"
            self.wake.wait(60)
            self.wake.clear()
            return
        if self.blank:
            self.blank = False
            if self.current:
                self._show(self.current)
        if s.paused:
            self.status = "paused"
            self.wake.wait(5)
            self.wake.clear()
            return
        if self.pending is None:
            e = self._generate()
            self.last_error = None
            with self.lock:
                self.pending = e
            if e is None:
                return
        with self.lock:
            due = self.current is None or self.skip or \
                  (time.time() - self.shown_at) >= self.settings.display_seconds
            if due and self.pending:
                self._show(self.pending)
                self.pending = None
                self.skip = False
                return
        self.status = "waiting"
        self.emit({"type": "status"})
        self.wake.wait(5)
        self.wake.clear()
