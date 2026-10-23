"""Window-frame templates.

Each frame is a mask the size of the panel: 255 where the panel should be opaque
(stonework, lead, iron bars) and 0 where the glass is. On a transparent LCD, black
blocks the sunlight, so the frame reads as a solid window opening.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

FRAMES = ["none", "lancet", "rose", "triptych", "tiffany"]

# How each frame's opening is described to the image model, so it composes for that shape.
FRAME_SHAPES = {
    "lancet": "a tall pointed gothic arch window",
    "rose": "a tall pointed gothic arch window with a round rose window at the top",
    "triptych": "three tall pointed gothic arch windows side by side",
    "tiffany": "a tall rectangular window",
}


def _grid(w: int, h: int):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x, y


def _pointed_arch(x, y, left, right, spring_y, bottom):
    """Inside of an equilateral pointed (gothic) arch spanning left..right."""
    span = right - left
    below = (y >= spring_y) & (y <= bottom) & (x >= left) & (x <= right)
    d_left = np.hypot(x - right, y - spring_y)   # left curve is centered on the right springing point
    d_right = np.hypot(x - left, y - spring_y)
    above = (y < spring_y) & (d_left <= span) & (d_right <= span) & (x >= left) & (x <= right)
    return below | above


def _arch_shape(w, h, margin, bottom_margin=None):
    left, right = margin, w - margin
    span = right - left
    apex = margin
    spring_y = apex + span * math.sin(math.radians(60))
    bottom = h - (bottom_margin if bottom_margin is not None else margin)
    return left, right, spring_y, bottom


def _band_divisions(band: np.ndarray, x, y, spacing: int, lead: int) -> np.ndarray:
    """Cut a border band into small panes with short leads every `spacing` px."""
    stripes = ((y.astype(int) % spacing) < lead) | ((x.astype(int) % spacing) < lead)
    return band & stripes


def _saddle_bars(glass: np.ndarray, h: int, count: int, width: int) -> np.ndarray:
    bars = np.zeros_like(glass)
    for i in range(1, count + 1):
        yy = int(h * i / (count + 1))
        bars[max(0, yy - width // 2): yy + (width + 1) // 2, :] = True
    return bars & glass


def render_frame(name: str, w: int, h: int, lead: int = 5, saddle_bars: bool = True,
                 silhouette: bool = False) -> Image.Image:
    """Return an 'L' mask: 255 = opaque, 0 = glass.

    silhouette=True instead returns just the main glass opening (255 = glass), without the
    decorative border band, tracery or bars.
    """
    x, y = _grid(w, h)
    unit = min(w, h)
    border = max(lead * 2, int(unit * 0.008))     # thin outer lead edge; more just wastes screen
    band = max(lead * 4, int(unit * 0.06))         # decorative pane band inside the border

    opaque = np.zeros((h, w), dtype=bool)

    if name == "none":
        opaque |= (x < border) | (x >= w - border) | (y < border) | (y >= h - border)
        shape = ~opaque

    elif name in ("lancet", "rose"):
        # one layer: the glass fills the whole arch, with a single lead line around its edge
        l, r, sy, b = _arch_shape(w, h, border)
        inner = _pointed_arch(x, y, l, r, sy, b)
        span2, sy2 = r - l, sy
        shape = inner
        opaque |= ~inner
        opaque |= inner & ~_pointed_arch(x, y, l + lead, r - lead, sy + lead * 0.6, b - lead)

        if name == "rose":
            cx, cy = w / 2, sy2 - span2 * 0.02
            rad = span2 * 0.42
            d = np.hypot(x - cx, y - cy)
            ang = np.arctan2(y - cy, x - cx)
            in_rose = d <= rad
            opaque |= (np.abs(d - rad) < lead) | (np.abs(d - rad * 0.32) < lead * 0.8)
            step = 2 * np.pi / 12
            m = np.mod(ang, step)
            perp = d * np.sin(np.minimum(m, step - m))   # distance to the nearest spoke
            opaque |= in_rose & (d > rad * 0.32) & (perp < lead * 0.6)
            # petal tips
            opaque |= in_rose & (np.abs(d - rad * 0.72) < lead * 0.6)
            # bar under the rose separating it from the lancet body
            opaque |= inner & (np.abs(y - (cy + rad + lead * 3)) < lead)
            glass_body = inner & (y > cy + rad + lead * 4)
        else:
            glass_body = inner
        if saddle_bars:
            opaque |= _saddle_bars(glass_body, h, 4, max(2, lead // 2))

    elif name == "triptych":
        gap = max(lead * 3, int(w * 0.03))
        col_w = (w - 2 * border - 2 * gap) / 3
        glass = np.zeros_like(opaque)
        for i in range(3):
            l = border + i * (col_w + gap)
            r = l + col_w
            top = border + (0 if i == 1 else int(h * 0.08))
            span = r - l
            sy = top + span * math.sin(math.radians(60))
            g = _pointed_arch(x, y, l, r, sy, h - border)
            glass_inner = _pointed_arch(x, y, l + lead, r - lead, sy + lead * 0.6, h - border - lead)
            opaque |= g & ~glass_inner
            if saddle_bars:
                opaque |= _saddle_bars(glass_inner, h, 5, max(2, lead // 2))
            glass |= g
        opaque |= ~glass
        shape = glass

    elif name == "tiffany":
        opaque |= (x < border) | (x >= w - border) | (y < border) | (y >= h - border)
        inner = (x >= border + band) & (x < w - border - band) & (y >= border + band) & (y < h - border - band)
        ring = ~inner & ~opaque
        shape = ~opaque  # the picture runs under the border band, so the band isn't left dark
        edge = (np.abs(x - (border + band)) < lead / 2) | (np.abs(x - (w - border - band)) < lead / 2) | \
               (np.abs(y - (border + band)) < lead / 2) | (np.abs(y - (h - border - band)) < lead / 2)
        opaque |= (edge & ~opaque & (ring | inner) & _near_rect(x, y, w, h, border + band, lead))
        opaque |= _band_divisions(ring, x, y, max(band, 40), lead)

    else:
        raise ValueError(f"unknown frame '{name}'")

    if silhouette:
        return Image.fromarray((shape * 255).astype(np.uint8), "L")
    return Image.fromarray((opaque * 255).astype(np.uint8), "L")


def _near_rect(x, y, w, h, inset, lead):
    in_x = (x >= inset - lead) & (x <= w - inset + lead)
    in_y = (y >= inset - lead) & (y <= h - inset + lead)
    return in_x & in_y


def glass_area(name: str, w: int, h: int) -> Image.Image:
    """The plain opening the subject should fill: 255 = glass, no border band or tracery."""
    return render_frame(name, w, h, lead=max(2, w // 200), saddle_bars=False, silhouette=True)


def control_image(name: str, w: int, h: int) -> Image.Image:
    """A picture the ControlNet's canny detector can read: a mid-gray silhouette on black,
    so the only edge is the window's outline and the frame's own detail is left to the overlay."""
    img = Image.new("RGB", (w, h), (10, 10, 10))
    img.paste((150, 150, 150), mask=glass_area(name, w, h))
    return img


def shape_start_image(name: str, w: int, h: int) -> Image.Image:
    """Dark stone outside the window opening and random colored glass shards inside. A little of
    this is mixed into the starting noise so the model composes the figure within the shape.
    Shards rather than a smooth glow: the model copies smoothness, and a glow came out blurry."""
    from scipy.spatial import cKDTree

    rnd = np.random.default_rng(0)
    n = max(50, w * h // 2048)                               # 450 shards at 720x1280
    x, y = _grid(w, h)
    dist, idx = cKDTree(rnd.uniform([0, 0], [w, h], (n, 2))).query(np.c_[x.ravel(), y.ravel()], k=2)
    palette = np.array([[214, 150, 84], [200, 90, 40], [170, 40, 40], [40, 110, 120], [40, 70, 150],
                        [230, 190, 120], [90, 50, 110]], np.float32)
    colors = palette[rnd.integers(0, len(palette), n)] * rnd.uniform(0.7, 1.1, (n, 1))
    glass = colors[idx[:, 0]].reshape(h, w, 3)
    glass[(dist[:, 1] - dist[:, 0]).reshape(h, w) < 3] = 25   # lead between shards
    inside = np.asarray(glass_area(name, w, h), np.float32)[..., None] / 255.0
    out = glass * inside + 12 * (1 - inside)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGB")


if __name__ == "__main__":  # quick visual check: python -m belleverriere.frames
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "frames_preview.png"
    tiles = [render_frame(f, 270, 480).convert("RGB") for f in FRAMES]
    sheet = Image.new("RGB", (270 * len(tiles) + 10 * (len(tiles) - 1), 480), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * 280, 0))
    ImageDraw.Draw(sheet)
    sheet.save(out)
    print("wrote", out)
