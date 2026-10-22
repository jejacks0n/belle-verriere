"""All tunable settings, persisted to data/settings.json so they survive restarts."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"


@dataclass
class LoraSpec:
    path: str  # Hugging Face repo id or a local .safetensors path
    scale: float = 0.8
    enabled: bool = True


@dataclass
class Settings:
    # --- What to make -------------------------------------------------------
    base_prompt: str = "a woman with a sword standing among lilies"
    style_prompt: str = (
        "fractured stained glass mosaic artwork, the whole image built from hundreds of irregular "
        "shards of glass, each shard hand-painted with soft realistic shading, thin dark bronze lead "
        "lines between every shard, sweeping concentric arcs of glass fragments around the subject, "
        "luminous backlit glass, warm amber, burnt orange and crimson set against deep teal and cobalt "
        "blue, dramatic, highly detailed, no text"
    )
    negative_prompt: str = "photo, 3d render, blurry, text, watermark, signature, gradient sky"

    # --- Prompt drift -------------------------------------------------------
    drift_enabled: bool = True
    drift: float = 0.35          # 0 = tiny tweaks, 1 = wild leaps
    tether: float = 0.3          # chance per mutation of pulling a slot back toward the original
    mutate_every: int = 1        # mutate the prompt every N images
    drift_guidance: str = ""     # rules the drift LLM must follow, e.g. "settings stay bright"
    drift_slots: list[str] = field(default_factory=lambda: [   # which prompt parts may change
        "subject", "action", "setting", "palette", "motif", "mood"])
    llm_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    time_of_day_palette: bool = True  # cooler colors in the morning, warmer toward sunset

    # --- Timing -------------------------------------------------------------
    display_seconds: int = 900   # how long each image stays up
    crossfade_seconds: int = 25
    paused: bool = False         # hold the current image, stop generating

    # --- Daylight -----------------------------------------------------------
    daylight_only: bool = False
    latitude: float | None = None
    longitude: float | None = None
    minutes_before_sunrise: int = 20
    minutes_after_sunset: int = 10

    # --- Model --------------------------------------------------------------
    model_path: str = "filipstrand/Z-Image-Turbo-mflux-4bit"
    quantize: int | None = None  # leave None for pre-quantized repos; 8 or 4 for full-precision repos
    float16_compute: bool = True  # recommended for M1-family GPUs
    steps: int = 9
    gen_width: int = 720         # multiples of 16; upscaled to the panel afterwards
    gen_height: int = 1280
    loras: list[LoraSpec] = field(default_factory=list)
    shape_strength: float = 0.0  # mix the frame's shape into the starting noise; 0 = off, 0.03 tested, 0.05+ blurs
    use_controlnet: bool = False  # bend the composition to the window frame (slower: ~20 steps)
    controlnet_strength: float = 0.75
    controlnet_steps: int = 20

    # --- Window frame & glass look -------------------------------------------
    preset: str = "between"      # last preset applied (informational)
    frame: str = "none"          # none | lancet | rose | triptych | tiffany
    trace_leads: float = 0.85    # deepen the lead lines the model painted
    shard_smoothing: float = 0.5  # even out color inside each shard
    glass_strength: float = 0.0  # re-cut into new flat panes (classic church glass)
    colors: int = 18             # palette size for the panes
    cell_smoothing: int = 7      # bigger = larger, simpler panes
    lead_width: int = 4          # lead line thickness in panel pixels
    saturation: float = 1.3
    contrast: float = 1.1
    texture: float = 0.2         # hand-blown glass mottling inside each pane
    saddle_bars: bool = False

    # --- Panel --------------------------------------------------------------
    out_width: int = 1080
    out_height: int = 1920
    display_rotation: int = 0    # extra CSS rotation on the display page: 0, 90, 180, 270


CLASSIC_STYLE = (
    "leaded stained glass window, flat panes of richly colored cathedral glass, "
    "thick black lead came lines outlining every shape, backlit by sunlight, "
    "jewel tones, art nouveau, symmetrical composition, no text"
)

# Each preset sets the look in one tap; everything stays adjustable afterwards.
PRESETS: dict[str, dict] = {
    "painted": {  # closest to the model's own painting
        "style_prompt": Settings.style_prompt, "frame": "none", "trace_leads": 0.5, "shard_smoothing": 0.25,
        "glass_strength": 0.0, "lead_width": 3, "saturation": 1.3, "contrast": 1.1, "texture": 0.15,
        "saddle_bars": False,
    },
    "between": {  # painterly shards with real lead: the default
        "style_prompt": Settings.style_prompt, "frame": "none", "trace_leads": 0.85, "shard_smoothing": 0.5,
        "glass_strength": 0.0, "lead_width": 4, "saturation": 1.3, "contrast": 1.1, "texture": 0.2,
        "saddle_bars": False,
    },
    "classic": {  # flat panes in a gothic window
        "style_prompt": CLASSIC_STYLE, "frame": "lancet", "trace_leads": 0.6, "shard_smoothing": 0.0,
        "glass_strength": 0.85, "colors": 18, "cell_smoothing": 7, "lead_width": 5, "saturation": 1.45,
        "contrast": 1.15, "texture": 0.35, "saddle_bars": True,
    },
}

_lock = threading.RLock()


def _from_dict(d: dict) -> Settings:
    names = {f.name for f in fields(Settings)}
    clean = {k: v for k, v in d.items() if k in names}
    if "loras" in clean:
        clean["loras"] = [LoraSpec(**l) if isinstance(l, dict) else l for l in clean["loras"]]
    return Settings(**clean)


def load() -> Settings:
    with _lock:
        if SETTINGS_FILE.exists():
            try:
                return _from_dict(json.loads(SETTINGS_FILE.read_text()))
            except Exception as e:  # corrupt file: fall back to defaults, keep the bad copy
                SETTINGS_FILE.rename(SETTINGS_FILE.with_suffix(".bad.json"))
                print(f"[settings] could not read settings ({e}); using defaults")
        return Settings()


def save(s: Settings) -> None:
    with _lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(s), indent=2))
        tmp.replace(SETTINGS_FILE)


def update(s: Settings, changes: dict) -> tuple[Settings, set[str]]:
    """Apply a partial update; returns the new settings and the names that changed."""
    current = asdict(s)
    merged = {**current, **{k: v for k, v in changes.items() if k in current}}
    new = _from_dict(merged)
    changed = {k for k in current if asdict(new)[k] != current[k]}
    return new, changed
