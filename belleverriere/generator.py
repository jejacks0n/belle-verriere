"""Image generation: Z-Image Turbo running natively on Apple Silicon through mflux/MLX."""
from __future__ import annotations

import gc
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .frames import shape_start_image
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
    return (s.model_path, s.quantize, s.float16_compute, loras)


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
        print(f"[mflux] loading model (loras={len(loras)}) ...")
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image import ZImage
        extra = {}
        if s.float16_compute:
            extra["compute_precision"] = mx.float16
        # bake_lora=False: baking re-rounds a LoRA into the 4-bit weights and kept only 64% of the
        # stained-glass LoRA's effect; runtime LoRA layers apply all of it (and were no slower).
        self._model = ZImage(model_config=ModelConfig.z_image_turbo(),
                             model_path=s.model_path, bake_lora=False, **common, **extra)
        self._model.callbacks.register(self._hook)
        self._key = key
        print("[mflux] model ready")
        return self._model

    def generate(self, req: GenRequest, s: Settings, on_step: Callable[[int, int], None] | None = None) -> Image.Image:
        with self._lock:
            model = self._load(s)
            self._hook.on_step = on_step
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

