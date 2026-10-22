"""Turn a generated image into something that reads as leaded glass on a sunlit LCD.

Steps: boost color -> flatten into a small palette of panes -> clean up specks ->
draw thick lead lines on every boundary between panes -> add hand-blown-glass
mottling -> apply the window frame (opaque stone/lead/iron).

A transparent LCD with no backlight is a color filter: black blocks the sun, white
lets it all through. Its filters are weak, so colors are pushed hard on purpose.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from .frames import render_frame


def _odd(n: int) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def _texture(w: int, h: int, seed: int) -> np.ndarray:
    """Low-frequency mottling in roughly [-1, 1], like seedy hand-rolled glass."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    for scale, weight in ((48, 0.6), (16, 0.3), (5, 0.1)):
        small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
        layer = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += (np.asarray(layer, np.float32) / 127.5 - 1.0) * weight
    return out


def make_glass(
    img: Image.Image,
    *,
    out_size: tuple[int, int],
    frame: str = "lancet",
    strength: float = 0.85,
    colors: int = 18,
    cell_smoothing: int = 7,
    lead_width: int = 5,
    saturation: float = 1.45,
    contrast: float = 1.15,
    texture: float = 0.35,
    saddle_bars: bool = True,
    trace_leads: float = 0.0,
    shard_smoothing: float = 0.0,
    seed: int = 0,
) -> Image.Image:
    """Two complementary ways to make glass, which can be mixed:

    - trace_leads / shard_smoothing keep the painting: the thin dark lines the model
      already drew between shards are deepened into lead, and color inside each shard is
      evened out without crossing those lines. Best for painterly, fractured-glass images.
    - strength (flatten) re-cuts the image into new flat panes of a small palette with
      lead on every boundary. Best for classic church-window looks.
    """
    w, h = out_size
    base = img.convert("RGB")
    if base.size != (w, h):
        base = _cover_resize(base, w, h)

    base = ImageEnhance.Color(base).enhance(saturation)
    base = ImageEnhance.Contrast(base).enhance(contrast)
    color = np.asarray(base, np.float32)
    s = float(np.clip(strength, 0.0, 1.0))
    t = float(np.clip(trace_leads, 0.0, 1.0))

    # A) trace the painting's own lead lines (continuous, so leads with highlights stay whole)
    traced = np.zeros((h, w), np.float32)
    if t > 0 or shard_smoothing > 0:
        traced = _trace_lines(color.mean(axis=2), lead_width)
    if shard_smoothing > 0:
        color = _smooth_within(color, traced, float(np.clip(shard_smoothing, 0, 1)))

    # B) re-cut into flat panes
    if s > 0:
        labels, panes = _segment(base, colors, cell_smoothing)
        edge = np.zeros(labels.shape, bool)
        edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edge[1:, :] |= labels[1:, :] != labels[:-1, :]
        lead_img = Image.fromarray((edge * 255).astype(np.uint8))
        if lead_width > 1:
            lead_img = lead_img.filter(ImageFilter.MaxFilter(_odd(lead_width)))
        pane_lead = np.asarray(lead_img.filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255.0
        color = panes * s + color * (1.0 - s)
        color = color * (1.0 - (pane_lead * s)[..., None])

    # glass mottling
    if texture > 0:
        tex = _texture(w, h, seed)
        color = color * (1.0 + 0.22 * texture * tex)[..., None]

    if t > 0:
        color = color * (1.0 - (traced * (0.25 + 0.7 * t))[..., None])

    # 5) window frame: fully opaque
    fmask = np.asarray(render_frame(frame or "none", w, h, lead=lead_width, saddle_bars=saddle_bars),
                       np.float32) / 255.0
    color = color * (1.0 - fmask[..., None])

    return Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB")


def _trace_lines(lum: np.ndarray, width: int) -> np.ndarray:
    """0..1 map of thin dark lines (the leads a model paints between shards).

    A grey closing fills in anything thinner than the kernel, so closing - original lights
    up thin dark lines but not broad dark areas like hair or shadow. Pre-smoothing and
    dropping tiny isolated bits keeps surface texture from turning into black speckle.
    """
    smooth = ndimage.gaussian_filter(lum, 1.2)
    bh = ndimage.grey_closing(smooth, size=(7, 7)) - smooth
    d = np.clip((bh - 7.0) / 32.0, 0.0, 1.0)
    mask = d > 0.25
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n:
        sl = ndimage.find_objects(lab)
        span = np.array([max(a.stop - a.start, b.stop - b.start) for a, b in sl])
        keep = np.isin(lab, np.nonzero(span >= 18)[0] + 1)
        d = d * ndimage.binary_dilation(keep, iterations=2)
    k = max(1, int(width) - 2)
    if k > 1:
        d = ndimage.grey_dilation(d, size=(k, k))
    return ndimage.gaussian_filter(d, 0.8).astype(np.float32)


def _smooth_within(color: np.ndarray, walls: np.ndarray, amount: float, sigma: float = 6.0) -> np.ndarray:
    """Even out color inside each shard without bleeding across lead lines."""
    m = (1.0 - np.clip(walls * 2.0, 0, 1)).astype(np.float32)
    den = ndimage.gaussian_filter(m, sigma) + 1e-4
    out = color.copy()
    for c in range(3):
        out[..., c] = color[..., c] * (1 - amount) + ndimage.gaussian_filter(color[..., c] * m, sigma) / den * amount
    return out


def _segment(base: Image.Image, colors: int, smoothing: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the image into panes of flat color.

    Works at a third of the panel resolution so panes come out glass-sized, merges
    fragments smaller than a minimum area into their neighbors, then scales back up
    with smooth (not stair-stepped) boundaries.
    """
    w, h = base.size
    sw, sh = max(64, w // 3), max(64, h // 3)
    small = base.resize((sw, sh), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.0))
    pal = small.quantize(colors=max(2, colors), method=Image.Quantize.MEDIANCUT, kmeans=3)
    k = _odd(max(3, smoothing))
    for _ in range(2):
        pal = pal.filter(ImageFilter.ModeFilter(k))
    labels = np.asarray(pal).copy()

    min_area = max(4, smoothing * smoothing * 3)
    too_small = np.zeros(labels.shape, bool)
    for c in np.unique(labels):
        comp, n = ndimage.label(labels == c)
        if n == 0:
            continue
        sizes = ndimage.sum_labels(np.ones_like(comp), comp, index=np.arange(1, n + 1))
        tiny = np.nonzero(sizes < min_area)[0] + 1
        if tiny.size:
            too_small |= np.isin(comp, tiny)
    if too_small.any() and not too_small.all():
        iy, ix = ndimage.distance_transform_edt(too_small, return_distances=False, return_indices=True)
        labels = labels[iy, ix]

    palette = np.array(pal.getpalette()[: 3 * 256], np.uint8).reshape(-1, 3)
    smooth_up = Image.fromarray(palette[labels]).resize((w, h), Image.BICUBIC)
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(pal.getpalette())
    q = smooth_up.quantize(palette=pal_img, dither=Image.Dither.NONE).filter(ImageFilter.ModeFilter(5))
    return np.asarray(q), np.asarray(q.convert("RGB"), np.float32)


def _cover_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to cover (w, h) and center-crop, keeping aspect ratio."""
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    nw, nh = round(sw * scale), round(sh * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def calibration(w: int, h: int, frame: str = "lancet", lead_width: int = 5) -> Image.Image:
    """A test pattern for judging colors on the actual window: hues x brightness, plus grays."""
    hues = 8
    rows = 10
    arr = np.zeros((h, w, 3), np.float32)
    cell_w, cell_h = w / hues, h / rows
    for r in range(rows):
        for c in range(hues):
            y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
            x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
            if r == rows - 1:  # gray ramp
                v = 255 * c / (hues - 1)
                rgb = (v, v, v)
            else:
                hue = c / hues
                val = 1.0 - 0.09 * r
                rgb = _hsv(hue, 1.0, val)
            arr[y0:y1, x0:x1] = rgb
    img = Image.fromarray(arr.astype(np.uint8))
    leads = np.zeros((h, w), bool)
    for c in range(1, hues):
        x = int(c * cell_w)
        leads[:, max(0, x - lead_width // 2): x + lead_width // 2 + 1] = True
    for r in range(1, rows):
        y = int(r * cell_h)
        leads[max(0, y - lead_width // 2): y + lead_width // 2 + 1, :] = True
    arr = np.asarray(img, np.float32) * (~leads)[..., None]
    fmask = np.asarray(render_frame(frame, w, h, lead=lead_width), np.float32) / 255.0
    arr = arr * (1.0 - fmask[..., None])
    return Image.fromarray(arr.astype(np.uint8))


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return (r * 255, g * 255, b * 255)
