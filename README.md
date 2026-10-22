# Stained Window

Generates stained glass images on your M1 MacBook Pro and shows them on a transparent LCD in
your window. The sun is the backlight. The prompt drifts slowly over hours, so the window keeps
surprising you while staying close to what you asked for. You control it from your phone.

```
phone ──► control page ─┐
                        ▼
              run.py (FastAPI) ── prompt drift ──► Ollama (local LLM)
                        │
                        ├── Z-Image Turbo via mflux (Apple Silicon, MLX)
                        ├── glass post-processing (leads, shard smoothing, frame)
                        ▼
              display page (Chrome kiosk on the panel, crossfades)
```

## Setup (once)

```bash
cd stained-window
./setup.sh
```

This creates a Python virtual environment, installs everything, and pulls a small Ollama
model for the prompt drift (llama3.1:8b, about 5 GB) if Ollama is installed. The image model
(Z-Image Turbo, 4-bit, several GB) downloads the first time an image is generated.

Set the panel to portrait first: **System Settings → Displays → (the panel) → Rotation: 90° or
270°.** 180° flips the picture upside down but keeps it landscape. If you'd rather leave macOS
alone, the control page has a rotation setting for the display page itself.

## Run

```bash
./start.sh            # generate for real
./start.sh --mock     # fake images: try the controls without loading the model
```

`start.sh` starts the server, opens the display full screen on the second screen, and keeps the
Mac from sleeping while it runs. It prints the control page address. Open that on your phone
(same Wi-Fi), e.g. `http://192.168.1.20:8765/`.

The first image takes a while (model download and load). After that, check "Last image took …"
on the control page. Any generation time shorter than the display time is fine: the next window
is always generated while the current one is up.

## The control page

- **Now showing:** *Next now*, *Pause* (holds the current image), *Star*, and *Test pattern*
  (color swatches and grays to judge how colors look in actual sunlight).
- **The idea:** what the window should show. *Start from this* restarts the drift from new text.
  The chips show where the drift has wandered; gold ones have changed from your original.
- **Drift:** how far each step wanders, how often it pulls back toward the original, how often
  the prompt changes, and which Ollama model writes the changes.
- **Timing:** how long each window stays up (1 min – 2 h) and the crossfade length.
- **The glass:** three presets, then fine-tuning. Slider changes preview live on the window;
  tap *Keep* to save them.
  - *Painted:* the model's painting, lightly leaded.
  - *Painted glass* (default): the look between your reference image and real glass. The lines
    the model paints between shards are deepened into solid lead and each shard's color is
    evened out, so the painting stays but it reads as glass.
  - *Classic:* re-cuts the image into flat panes with a gothic lancet frame. It switches the style
    prompt to church glass as well, because it needs flat, graphic images to work well.
- **Style & model:** the style text (which never drifts), model, steps, sizes, LoRAs, and the
  ControlNet option that shapes each picture to the window frame.
- **Daylight & display:** sleep at night (enter latitude/longitude for real sunrise/sunset),
  shift colors with the time of day, and display rotation.
- **History:** tap any image to show it again, star it, or *Drift from here* to continue the
  drift from that image's prompt.

## Trying looks on any image

```bash
source .venv/bin/activate
python try_look.py examples/reference.png        # all presets side by side
python try_look.py some.png --preset between --frame rose
```

`examples/reference.png` is the image you liked; `examples/reference_looks.jpg` shows it in
each preset.

## Tips for a transparent LCD in sunlight

- Black blocks light, white passes all of it. Dark areas in an image become dark areas in your
  window, and white areas glow.
- LCD color filters are weak, so saturation is boosted on purpose. If colors still look washed
  out in direct sun, raise Saturation and Contrast; the Test pattern helps judge this.
- Thin leads can disappear in bright light. If they do, raise Lead width or Trace.

## LoRAs

Add them under *Style & model → LoRAs* as a Hugging Face repo id or a local `.safetensors`
path, with a strength around 0.5–0.9. The model reloads before the next image. mflux can also
train a LoRA on this Mac (`mflux-train`, see the mflux Z-Image docs). Twenty to forty images in
the style of your reference would be a good first dataset; on an M1 expect an overnight run.

## Files

```
run.py                 start the server (python run.py [--mock])
try_look.py            the glass look on any image, no server needed
stainedwindow/
  settings.py          every setting, presets; saved to data/settings.json
  engine.py            main loop: generate next, wait, crossfade; history
  generator.py         mflux (Z-Image Turbo, optional ControlNet) and the mock backend
  drift.py             prompt slots and the LLM drift
  glass.py             post-processing: traced leads, shard smoothing, panes, texture
  frames.py            window frame shapes
  daylight.py          sunrise/sunset and time-of-day colors
static/control.html    phone control page
static/display.html    the window
data/                  settings, history and images (created on first run)
```

Up to 600 images are kept; older ones are deleted unless starred.

## Troubleshooting

- **Display stays black:** check the server is running and open `http://localhost:8765/display`
  in a normal browser. At night with *Sleep at night* on, black is expected.
- **"Ollama not reachable":** start the Ollama app. Until then the drift uses a built-in word
  bank, so it keeps working, just less inventively.
- **Out of memory / very slow:** keep the 4-bit model, leave ControlNet off, and generate at
  720×1280 (the default). Close other heavy apps.
- **Phone can't reach the control page:** both devices need to be on the same network, and macOS
  may ask to allow incoming connections for Python the first time; allow it.
- **No authentication:** anyone on your home network can open the control page. Don't expose
  port 8765 to the internet.
