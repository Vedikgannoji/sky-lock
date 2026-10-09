"""Orbital target beacon provider and frame source using true spherical geometry."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from skylock.config.models import GimbalConfig, SkyLockConfig, TargetConfig
from skylock.core.enums import InputKind
from skylock.core.geometry import wrap_deg
from skylock.core.interfaces import FrameSource, GimbalPlant
from skylock.core.orbital_world import (
    camera_tangent_offset,
    dir_to_pan_tilt,
    peer_bearing,
)
from skylock.core.rng import derive_rng
from skylock.core.sim_clock import SimClock, get_shared_clock
from skylock.core.types import Frame, GroundTruthSample, Pointing
from skylock.simulation.camera import VirtualCamera
from skylock.simulation.disturbances.base import DisturbanceContext
from skylock.simulation.disturbances.stack import DisturbanceStack
from skylock.simulation.gimbal import VirtualGimbal
from skylock.simulation.ground_truth import build_ground_truth
from skylock.simulation.targets import Target, make_sprite

if TYPE_CHECKING:
    pass


class OrbitalTarget(Target):
    """Dynamic optical beacon whose true position is computed from 3D orbital mechanics.

    Implements Target protocol expected by VirtualCamera and GroundTruth generator.
    """

    def __init__(
        self,
        observer_id: str,
        peer_id: str,
        gimbal: GimbalPlant,
        clock: SimClock | None = None,
        use_ephemeris_cue: bool = True,
        ephemeris_error_deg: float = 1.0,
        seed: int = 42,
        spec: TargetConfig | None = None,
    ) -> None:
        self.observer_id = observer_id.lower().replace("-", "")
        self.peer_id = peer_id.lower().replace("-", "")
        self.gimbal = gimbal
        self.clock = clock
        self.use_ephemeris_cue = use_ephemeris_cue
        self.ephemeris_error_deg = float(ephemeris_error_deg)
        self.seed = seed

        # Target specification and beacon shape
        self.spec = spec if spec is not None else TargetConfig(
            id=f"{self.peer_id}_beacon",
            brightness=240.0,
            size_px=10,
            shape="disc",
        )
        self.sprite = make_sprite(self.spec.shape, self.spec.size_px)
        self.brightness = self.spec.brightness
        self.visibility_windows = self.spec.visibility_windows

        # Sample initial ephemeris error offset
        self._rng = derive_rng(seed, f"orbital_cue_{self.observer_id}_{self.peer_id}")
        self._sample_ephemeris_error()

        # Telemetry cache
        self._last_t: float = 0.0
        self._last_dpan: float = 0.0
        self._last_dtilt: float = 0.0
        self._last_los_clear: bool = True
        self._last_in_front: bool = True
        self._last_true_pan: float = 0.0
        self._last_true_tilt: float = 0.0

    def _sample_ephemeris_error(self) -> None:
        """Sample constant or pass-level Gaussian ephemeris error offset."""
        if self.ephemeris_error_deg > 0.0:
            e_pan = float(self._rng.normal(0.0, self.ephemeris_error_deg))
            e_tilt = float(self._rng.normal(0.0, self.ephemeris_error_deg))
        else:
            e_pan, e_tilt = 0.0, 0.0
        self._ephem_err = (e_pan, e_tilt)

    def resample_cue_error(self) -> None:
        """Re-sample ephemeris error offset (e.g. upon entering reacquisition)."""
        self._sample_ephemeris_error()

    def _eval(self, t: float) -> tuple[float, float, bool]:
        t_sim = self.clock.now() if self.clock is not None else float(t)
        self._last_t = t_sim

        unit_vec, _, los_clear = peer_bearing(self.observer_id, t_sim, self.peer_id)
        true_pan, true_tilt = dir_to_pan_tilt(unit_vec)
        self._last_true_pan = true_pan
        self._last_true_tilt = true_tilt
        self._last_los_clear = los_clear

        pt = self.gimbal.pointing
        dpan, dtilt, in_front = camera_tangent_offset(unit_vec, pt.pan_deg, pt.tilt_deg)
        self._last_dpan = dpan
        self._last_dtilt = dtilt
        self._last_in_front = in_front

        return (dpan, dtilt, los_clear)

    def position(self, t: float) -> tuple[float, float]:
        """Compute synthetic target angular position so VirtualCamera planar differencing:

            dpan = angular_diff_deg(pos[0], pointing.pan_deg)
            dtilt = angular_diff_deg(pos[1], pointing.tilt_deg)

        yields the EXACT true tangent-plane angular offsets (dpan, dtilt).
        """
        dpan, dtilt, _ = self._eval(t)
        pt = self.gimbal.pointing
        return (pt.pan_deg + dpan, pt.tilt_deg + dtilt)

    def is_visible(self, t: float) -> bool:
        """Visible if optical line of sight is clear and peer is in front of camera sensor plane."""
        # Check explicit visibility windows if defined
        if not all(not (t0 <= t <= t1) for t0, t1 in self.visibility_windows):
            return False
        return self._last_los_clear and self._last_in_front

    @property
    def true_offset_deg(self) -> tuple[float, float]:
        """True tangent-plane angular offset from boresight (dpan_deg, dtilt_deg)."""
        return (self._last_dpan, self._last_dtilt)

    @property
    def true_bearing_deg(self) -> tuple[float, float]:
        """True peer bearing in observer frame (pan_deg, tilt_deg)."""
        return (self._last_true_pan, self._last_true_tilt)

    @property
    def los_clear(self) -> bool:
        """Whether optical line-of-sight is unobstructed by Earth."""
        return self._last_los_clear

    def get_cue(self, t: float | None = None) -> tuple[tuple[float, float] | None, bool]:
        """Compute current ephemeris cue (cue_pan, cue_tilt) and los_clear status.

        Returns:
            (cue_pan_tilt, los_clear) or (None, los_clear) if cue is disabled.
        """
        t_sim = self.clock.now() if (self.clock is not None and t is None) else (float(t) if t is not None else 0.0)
        unit_vec, _, los_clear = peer_bearing(self.observer_id, t_sim, self.peer_id)
        if not self.use_ephemeris_cue:
            return (None, los_clear)

        true_pan, true_tilt = dir_to_pan_tilt(unit_vec)
        cue_pan = wrap_deg(true_pan + self._ephem_err[0])
        cue_tilt = max(-90.0, min(90.0, true_tilt + self._ephem_err[1]))
        return ((cue_pan, cue_tilt), los_clear)

    def reset(self) -> None:
        self._rng = derive_rng(self.seed, f"orbital_cue_{self.observer_id}_{self.peer_id}")
        self._sample_ephemeris_error()
        self._last_t = 0.0
        self._last_dpan = 0.0
        self._last_dtilt = 0.0
        self._last_los_clear = True
        self._last_in_front = True


class OrbitalSource(FrameSource):
    """Live orbital frame source connecting 3D orbital geometry to sensor and gimbal."""

    def __init__(
        self,
        config: SkyLockConfig,
        observer_id: str = "s1",
        peer_id: str = "s2",
        gimbal: VirtualGimbal | None = None,
        clock: SimClock | None = None,
    ) -> None:
        self.config = config
        self.observer_id = observer_id.lower().replace("-", "")
        self.peer_id = peer_id.lower().replace("-", "")
        self.clock = clock if clock is not None else get_shared_clock()

        # Build live gimbal with live limits (pan +-180 wrap, tilt +-90, max slew 10 deg/s)
        if gimbal is not None:
            self.gimbal = gimbal
            slew_rate = min(10.0, max(0.1, float(config.gimbal.slew_rate_deg_s)))
            live_gimbal_cfg = GimbalConfig(
                slew_rate_deg_s=slew_rate,
                max_slew_rate_deg_s=slew_rate,
                accel_deg_s2=config.gimbal.accel_deg_s2,
                pan_limit_deg=(-180.0, 180.0),
                tilt_limit_deg=(-90.0, 90.0),
                initial=(0.0, 0.0),
                substeps=config.gimbal.substeps,
            )
            self.gimbal = VirtualGimbal(live_gimbal_cfg)

        # Orbital target adapter
        self.target = OrbitalTarget(
            observer_id=self.observer_id,
            peer_id=self.peer_id,
            gimbal=self.gimbal,
            clock=self.clock,
            use_ephemeris_cue=config.tracking.use_ephemeris_cue,
            ephemeris_error_deg=config.tracking.ephemeris_error_deg,
            seed=config.seed,
        )

        # Session start: initial pointing = cue (if cue enabled)
        if config.tracking.use_ephemeris_cue:
            t0 = self.clock.now()
            cue, _ = self.target.get_cue(t0)
            if cue is not None:
                self.gimbal.set_pointing(cue[0], cue[1])

        self.camera = VirtualCamera(config.camera)
        self.disturbances = DisturbanceStack(
            config=config.disturbances,
            seed=config.seed,
            background_level=config.camera.background_level,
        )
        self._frame_index = 0
        self._is_open = True

    def open(self) -> None:
        self._is_open = True

    def _render_frame(self) -> tuple[Frame, GroundTruthSample]:
        fps = self.config.camera.fps
        t_sim = self.clock.now() if self.clock is not None else (self._frame_index / fps)
        current_pointing = self.gimbal.pointing

        # 1. Disturbances
        dx, dy = self.disturbances.compute_geometric_offset(self._frame_index, t_sim)

        # 2. Render target via VirtualCamera
        image_float, render_infos = self.camera.render(
            pointing=current_pointing,
            t=t_sim,
            targets=[self.target],  # type: ignore[list-item]
            extra_offset_px=(dx, dy),
        )

        # 3. Photometric disturbances
        ctx = DisturbanceContext(frame_index=self._frame_index, timestamp_s=t_sim)
        image_disturbed = self.disturbances.apply_photometric(image_float, ctx)

        # 4. Quantize to 8-bit monochrome
        image_uint8 = self.disturbances.quantize(image_disturbed)

        # Build Frame (image only, strictly no ground truth)
        frame = Frame(
            image=image_uint8,
            index=self._frame_index,
            timestamp_s=t_sim,
            source_id=self.source_id,
            pointing=current_pointing,
        )

        # Ground truth
        gt = build_ground_truth(
            frame_index=self._frame_index,
            timestamp_s=t_sim,
            render_infos=render_infos,
            targets=[self.target],  # type: ignore[list-item]
            pointing=current_pointing,
            camera=self.config.camera,
            disturbance_offset_px=(dx, dy),
        )

        self._frame_index += 1
        return frame, gt

    def read(self) -> Frame | None:
        if not self._is_open:
            return None
        frame, _ = self._render_frame()
        return frame

    def read_with_truth(self) -> tuple[Frame, GroundTruthSample]:
        if not self._is_open:
            raise RuntimeError("OrbitalSource is closed")
        return self._render_frame()

    def get_cue(self, t: float | None = None) -> tuple[tuple[float, float] | None, bool]:
        """Provide current ephemeris cue and los_clear status for this session."""
        return self.target.get_cue(t)

    def reset(self) -> None:
        self._frame_index = 0
        self.target.reset()
        self.gimbal.reset()
        self.disturbances.reset()
        # Reset to cue on session start
        if self.config.tracking.use_ephemeris_cue:
            t0 = self.clock.now() if self.clock is not None else 0.0
            cue, _ = self.target.get_cue(t0)
            if cue is not None:
                self.gimbal.set_pointing(cue[0], cue[1])

    def close(self) -> None:
        self._is_open = False

    @property
    def width(self) -> int:
        return self.config.camera.width

    @property
    def height(self) -> int:
        return self.config.camera.height

    @property
    def fps(self) -> float:
        return self.config.camera.fps

    @property
    def source_id(self) -> str:
        return f"orbital_{self.observer_id}_to_{self.peer_id}"

    @property
    def kind(self) -> InputKind:
        return InputKind.ORBITAL


__all__ = ("OrbitalSource", "OrbitalTarget")
