"""Image generation backends.

MfluxBackend runs Z-Image Turbo natively on Apple Silicon through mflux/MLX.
MockBackend draws random colored shapes so everything else can be tested anywhere.
"""
from __future__ import annotations

import gc
import random
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter

from .frames import control_image, shape_start_image
from .settings import Settings


@dataclass
class GenRequest:
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int


def _model_key(s: Settings) -> tuple:
    loras = tuple((l.path, round(l.scale, 3)) for l in s.loras if l.enabled)
    return (s.model_path, s.quantize, s.float16_compute, s.use_controlnet, loras)


class _StepHook:
    """mflux in-loop callback that reports (step, total) after each denoise step."""
    on_step: Callable[[int, int], None] | None = None

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        if self.on_step:
            self.on_step(t + 1, config.num_inference_steps)


@contextmanager
def _shaped_noise(clean, amount: float):
    """Mix a little of the window-shape picture into the starting noise so the composition
    follows it. mflux's img2img can only skip whole steps, which is far too strong for Turbo:
    one skipped step (6% image) already comes out blurry. ~0.03 keeps full detail at 6-9 steps."""
    from mflux.models.common.latent_creator.latent_creator import LatentCreator
    from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator as Z

    # ponytail: patches an mflux internal; recheck this if mflux is upgraded
    original = Z.__dict__["create_noise"]

    def create_noise(seed, height, width):
        noise = original.__func__(seed, height, width)
        # the VAE returns float32; keep the model's dtype or every step runs slower in float32
        return LatentCreator.add_noise_by_interpolation(clean=clean, noise=noise, sigma=1 - amount).astype(noise.dtype)

    Z.create_noise = staticmethod(create_noise)
    try:
        yield
    finally:
        Z.create_noise = original


class MfluxBackend:
    name = "mflux"

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self._model = None
        self._key = None
        self._lock = threading.Lock()
        self._hook = _StepHook()
        # mflux's progress bars would otherwise create a multiprocessing lock, which leaks (and warns) on Ctrl-C
        from tqdm import tqdm
        tqdm.set_lock(threading.RLock())
        self._shape = (None, None)  # (frame, w, h) -> encoded shape picture, reused across images

    def _load(self, s: Settings):
        key = _model_key(s)
        if self._model is not None and key == self._key:
            return self._model
        import mlx.core as mx  # noqa: imported lazily so the server starts without MLX

        if self._model is not None:  # never hold two sets of weights at once
            self._model = None
            self._shape = (None, None)
            gc.collect()
            mx.clear_cache()

        loras = [l for l in s.loras if l.enabled]
        common = dict(
            quantize=s.quantize,
            lora_paths=[l.path for l in loras] or None,
            lora_scales=[l.scale for l in loras] or None,
        )
        print(f"[mflux] loading model (controlnet={s.use_controlnet}, loras={len(loras)}) ...")
        if s.use_controlnet:
            from mflux.models.z_image import ZImageTurboControlnet
            # The ControlNet checkpoint bundles its own Turbo transformer; quantize it on load.
            common["quantize"] = s.quantize or 8
            self._model = ZImageTurboControlnet(**common)
        else:
            from mflux.models.common.config import ModelConfig
            from mflux.models.z_image import ZImage
            extra = {}
            if s.float16_compute:
                extra["compute_precision"] = mx.float16
            self._model = ZImage(model_config=ModelConfig.z_image_turbo(),
                                 model_path=s.model_path, **common, **extra)
        self._model.callbacks.register(self._hook)
        self._key = key
        print("[mflux] model ready")
        return self._model

    def generate(self, req: GenRequest, s: Settings, on_step: Callable[[int, int], None] | None = None) -> Image.Image:
        with self._lock:
            model = self._load(s)
            self._hook.on_step = on_step
            if s.use_controlnet:
                from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
                hint = self.work_dir / f"shape_{s.frame}_{req.width}x{req.height}.png"
                if not hint.exists():
                    control_image(s.frame, req.width, req.height).save(hint)
                out = model.generate_image(
                    seed=req.seed, prompt=req.prompt,
                    controls=[ControlSpec(type=ControlType.canny, image_path=str(hint),
                                          strength=s.controlnet_strength)],
                    controlnet_strength=1.0,
                    num_inference_steps=s.controlnet_steps,
                    width=req.width, height=req.height,
                )
            else:
                shaped = s.shape_strength > 0 and s.frame != "none"
                with _shaped_noise(self._shape_latent(model, s.frame, req.width, req.height),
                                   s.shape_strength) if shaped else nullcontext():
                    out = model.generate_image(
                        seed=req.seed, prompt=req.prompt,
                        num_inference_steps=req.steps,
                        width=req.width, height=req.height,
                    )
            img = out.image.copy()
            del out
            import mlx.core as mx
            gc.collect()
            mx.clear_cache()
            return img

    def _shape_latent(self, model, frame: str, w: int, h: int):
        """The frame's shape picture, encoded once per frame and size. Encoding it lazily on every
        image added ~18s, mostly around the first and last steps."""
        if self._shape[0] != (frame, w, h):
            import mlx.core as mx
            from mflux.models.common.latent_creator.latent_creator import LatentCreator
            from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator as Z

            start = self.work_dir / "shape_start.png"
            shape_start_image(frame, w, h).save(start)
            clean = Z.pack_latents(LatentCreator.encode_image(vae=model.vae, image_path=start, height=h, width=w,
                                                              tiling_config=model.tiling_config), h, w)
            mx.eval(clean)
            self._shape = ((frame, w, h), clean)
        return self._shape[1]

    def unload(self):
        with self._lock:
            self._model = None
            self._key = None
            self._shape = (None, None)
            gc.collect()


class MockBackend:
    """Random bold shapes, for testing the pipeline without a GPU."""
    name = "mock"

    def __init__(self, work_dir: Path, delay: float = 2.0):
        self.delay = delay

    def generate(self, req: GenRequest, s: Settings, on_step: Callable[[int, int], None] | None = None) -> Image.Image:
        import time
        for i in range(req.steps):
            time.sleep(self.delay / req.steps)
            if on_step:
                on_step(i + 1, req.steps)
        rnd = random.Random(req.seed)
        img = Image.new("RGB", (req.width, req.height), tuple(rnd.randint(0, 80) for _ in range(3)))
        d = ImageDraw.Draw(img)
        for _ in range(40):
            x, y = rnd.randint(-100, req.width), rnd.randint(-100, req.height)
            r = rnd.randint(30, 260)
            col = tuple(rnd.randint(30, 255) for _ in range(3))
            if rnd.random() < 0.5:
                d.ellipse([x, y, x + r, y + r], fill=col)
            else:
                d.polygon([(x, y), (x + r, y + rnd.randint(-r, r)), (x + rnd.randint(-r, r), y + r)], fill=col)
        d.text((20, 20), req.prompt[:60], fill=(255, 255, 255))
        return img.filter(ImageFilter.GaussianBlur(2))

    def unload(self):
        pass


def make_backend(kind: str, work_dir: Path):
    if kind == "mock":
        return MockBackend(work_dir)
    return MfluxBackend(work_dir)
