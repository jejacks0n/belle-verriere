# Belle Verrière

> “For me a stained glass window is a transparent partition between my heart and the world’s heart.” — Marc Chagall

I built Belle Verrière to explore a modern take on stained glass. Blending an ancient medium with generative tools to create an ever-changing light installation for my workspace.

Mounted directly within a window, a transparent LCD panel sits between the room and the sky outside. There is no artificial backlight -- only the natural sun, passing clouds, and shifting daylight and shadows from the trees casting colored silhouettes, refractions, and deep leaded shadows across my office floor and my desk throughout the day.

I’ve always loved the expressive, floating color fields of Marc Chagall’s windows and the timeless depth of cathedral glass like Notre-Dame de la Belle-Verrière at Chartres. With this project, I wanted to bring that tradition into a quiet dialogue with modern technology. Instead of treating a screen like a monitor you stare at, it lives as part of the architecture of the room.

# How It Works

The piece is designed around slow, wandering transformation rather than rapid generation:

- A Seed Idea: Everything begins with an idea. A phrase, a mood, or a visual prompt set from my phone.
- The Drift: A local language model slowly meditates on that idea over hours, nudging the prompt in subtle directions. It wanders into new colors, motifs, and compositions, drifting far enough to surprise me while staying loosely tethered to where it began, based on constraints provided.
- Leads and Shards: Images are generated on device and post-processed to feel like physical glass, smoothing colors into distinct panes and deepening contours into solid lead caming.
- Natural Light: Because the LCD is transparent, the weather and time of day dictate the mood. Direct afternoon sun burns bright colors onto the desk, while gray overcast days soften the window into gentle, moody tones.

It’s an ongoing experiment in ambient art: a small slice of glass, an idea, an LLM, and the sun working together to cast something new into the room every day.


```
phone ──► control page ─┐
                        ▼
              run.py (FastAPI) ── prompt drift ──► Ollama (local LLM)
                        │
                        ├── Z-Image Turbo via mflux (Apple Silicon, MLX)
                        ├── glass post-processing (leads, shard smoothing, frame)
                        ▼
              display page (open it full screen on the panel, crossfades)
```

## Setup (once)

```bash
cd belle-verriere
./setup.sh
```

This creates a Python virtual environment, installs everything, and pulls a small Ollama model for the prompt drift (llama3.1:8b, about 5 GB) if Ollama is installed. The image model (Z-Image Turbo, 4-bit, several GB) downloads the first time an image is generated.

## Run

```bash
./start.sh
```

`start.sh` starts the server and keeps the Mac from sleeping while it runs.

The first image takes a while (model download and load). After that, check "Last image took …" on the control page. Any generation time shorter than the display time is fine: the next window is always generated while the current one is up.

## Showing it on a Roku

`roku/` is a small Roku app that shows the window like the display page does: crossfades,
rotation and going dark at night. It asks this Mac what to show every few seconds, so the
server has to be running. It's for sideloading only.

1. Put the Roku in developer mode: on the remote press Home ×3, Up ×2, Right, Left, Right, Left, Right, then enable the installer and set a password. Note the IP address it shows.
2. Turn off the Roku screensaver (Settings → Theme → Screensaver → Change wait time → Disabled).
   Roku apps that only show pictures can't keep it away themselves. On Roku TVs, also check
   Settings → System → Power for automatic power-off.
3. Build and install:
   ```bash
   ROKU_IP=192.168.1.50 ROKU_PASS=your-password roku/build.sh
   ```
   Without `ROKU_IP` it only builds `roku/belle-verriere.zip`; upload that at
   `http://<roku-ip>` (user `rokudev`) instead.

The app has this Mac's address built in at build time. If the Mac's address changes, run
`roku/build.sh` again (or give the Mac a fixed address in your router). A portrait-mounted TV
uses the display rotation setting on the control page.

## Tips for a transparent LCD in sunlight

- Black blocks light, white passes all of it. Dark areas in an image become dark areas in your window, and white areas glow.
- LCD color filters are weak, so saturation is boosted on purpose. If colors still look washed out in direct sun, raise Saturation and Contrast.
- Thin leads can disappear in bright light. If they do, raise Lead width or Trace.

## LoRAs

Add them under *Style & model → LoRAs* as a Hugging Face repo id or a local `.safetensors` path, with a strength around 0.5–0.9. The model reloads before the next image. mflux can also train a LoRA on this Mac (`mflux-train`, see the mflux Z-Image docs). Twenty to forty images in the style of your reference would be a good first dataset; on an M1 expect an overnight run.
