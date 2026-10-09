"""Disturbance stack coordinating physical, optical, and sensor perturbation pipelines."""

from __future__ import annotations

import numpy as np

from skylock.config.models import DisturbanceConfig
from skylock.simulation.disturbances.atmosphere import Atmosphere
from skylock.simulation.disturbances.base import DisturbanceContext
from skylock.simulation.disturbances.blur import OpticalBlur
from skylock.simulation.disturbances.jitter import CameraJitter
from skylock.simulation.disturbances.noise import (
    GaussianNoise,
    PoissonNoise,
    SaltPepperNoise,
)
from skylock.simulation.disturbances.platform import PlatformMotion


class DisturbanceStack:
    """Ordered pipeline of geometric and photometric disturbances.

    Pipeline execution order:
    1. Geometric offsets (Camera Jitter + Platform Motion) -> passed to VirtualCamera.render
    2. Atmosphere (Clear / Haze / Fog / Rain / Low-Light)
    3. Optical Blur (Defocus PSF)
    4. Poisson Shot Noise
    5. Gaussian Read Noise
    6. Salt & Pepper Impulse Noise
    7. Quantize / round and clip to uint8 [0, 255]
    """

    def __init__(
        self,
        config: DisturbanceConfig,
        seed: int,
        background_level: float = 20.0,
    ) -> None:
        self.config = config
        self.seed = seed
        self.background_level = float(background_level)

        self.camera_jitter = CameraJitter(config.camera_jitter, seed=seed)
        self.platform = PlatformMotion(config.platform, seed=seed)
        self.atmosphere = Atmosphere(
            config.atmosphere, seed=seed, background_level=self.background_level
        )
        self.blur = OpticalBlur(config.blur, seed=seed)
        self.poisson = PoissonNoise(config.poisson, seed=seed)
        self.gaussian = GaussianNoise(config.gaussian, seed=seed)
        self.salt_pepper = SaltPepperNoise(config.salt_pepper, seed=seed)

    def reset(self) -> None:
        """Reset all disturbance models to their initial deterministic state."""
        self.camera_jitter.reset()
        self.platform.reset()
        self.atmosphere.reset()
        self.blur.reset()
        self.poisson.reset()
        self.gaussian.reset()
        self.salt_pepper.reset()

    def compute_geometric_offset(self, frame_index: int, t: float) -> tuple[float, float]:
        """Compute aggregate line-of-sight shift (dx, dy) on the sensor focal plane."""
        dx_jit, dy_jit = self.camera_jitter.offset_px(frame_index, t)
        dx_plat, dy_plat = self.platform.offset_px(frame_index, t)
        return (dx_jit + dx_plat, dy_jit + dy_plat)

    def apply_photometric(self, image: np.ndarray, ctx: DisturbanceContext) -> np.ndarray:
        """Apply sequential photometric degradation operators in strict order."""
        # 1. Atmosphere
        out = self.atmosphere.apply(image, ctx)
        # 2. Blur
        out = self.blur.apply(out, ctx)
        # 3. Poisson shot noise
        out = self.poisson.apply(out, ctx)
        # 4. Gaussian read noise
        out = self.gaussian.apply(out, ctx)
        # 5. Salt & Pepper impulse noise
        out = self.salt_pepper.apply(out, ctx)
        return out

    def update_config(self, new_config: DisturbanceConfig) -> None:
        """Apply a new DisturbanceConfig without resetting stateful RNGs.

        Only config references are replaced; RNG state and history are preserved so
        seeded repeatability is not broken by live edits during a session.
        """
        import math as _math

        self.config = new_config
        # Update config references on each sub-model (preserves RNG / accumulator state)
        self.camera_jitter.config = new_config.camera_jitter
        self.camera_jitter._max_px = min(20.0, max(0.0, float(new_config.camera_jitter.max_px_frame)))
        self.camera_jitter._correlation = min(0.9999, max(0.0, float(new_config.camera_jitter.correlation)))
        self.platform.config = new_config.platform
        # Recompute effective velocity for platform (respects max_px_frame)
        v_raw = new_config.platform.velocity_px_frame
        if isinstance(v_raw, (int, float)):
            vx, vy = float(v_raw), 0.0
        else:
            vx, vy = float(v_raw[0]), float(v_raw[1])
        v_mag = _math.hypot(vx, vy)
        max_limit = 20.0
        if new_config.platform.max_px_frame > 0.0:
            max_limit = min(20.0, float(new_config.platform.max_px_frame))
        if v_mag > max_limit and v_mag > 0.0:
            scale = max_limit / v_mag
            self.platform._vx_eff = vx * scale
            self.platform._vy_eff = vy * scale
        else:
            self.platform._vx_eff = vx
            self.platform._vy_eff = vy
        self.atmosphere.config = new_config.atmosphere
        self.blur.config = new_config.blur
        self.poisson.config = new_config.poisson
        self.gaussian.config = new_config.gaussian
        self.salt_pepper.config = new_config.salt_pepper

    def quantize(self, image: np.ndarray) -> np.ndarray:
        """Quantize float32 image to 8-bit unsigned integer with rounding and clipping."""
        return np.clip(np.round(image), 0, 255).astype(np.uint8)
