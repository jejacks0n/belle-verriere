#!/usr/bin/env python3
"""Try the glass look on any image, without the server or the model.

    python try_look.py examples/reference.png                 # all three presets side by side
    python try_look.py my.png --preset classic --frame rose   # one look, saved next to the input
"""
import argparse
from pathlib import Path

from PIL import Image

from belleverriere.glass import make_glass
from belleverriere.settings import PRESETS, Settings

KEYS = ["frame", "glass_strength", "colors", "cell_smoothing", "lead_width", "saturation", "contrast",
        "texture", "saddle_bars", "trace_leads", "shard_smoothing"]


def render(img, preset: str, frame: str | None, size):
    s = {**{k: getattr(Settings(), k) for k in KEYS}, **{k: v for k, v in PRESETS[preset].items() if k in KEYS}}
    if frame:
        s["frame"] = frame
    s["strength"] = s.pop("glass_strength")
    return make_glass(img, out_size=size, **s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--preset", choices=list(PRESETS), help="one preset (default: all, side by side)")
    p.add_argument("--frame", help="override the frame: none, lancet, rose, triptych, tiffany")
    p.add_argument("--size", default="1080x1920")
    a = p.parse_args()
    size = tuple(int(x) for x in a.size.lower().split("x"))
    src = Image.open(a.image)
    stem = Path(a.image).with_suffix("")
    if a.preset:
        out = render(src, a.preset, a.frame, size)
        path = f"{stem}_{a.preset}.jpg"
    else:
        tiles = [src.convert("RGB").resize(size)] + [render(src, k, a.frame, size) for k in PRESETS]
        gap = 20
        out = Image.new("RGB", (len(tiles) * size[0] + gap * (len(tiles) - 1), size[1]), "white")
        for i, t in enumerate(tiles):
            out.paste(t, (i * (size[0] + gap), 0))
        path = f"{stem}_looks.jpg"
    out.save(path, quality=92)
    print("wrote", path)
