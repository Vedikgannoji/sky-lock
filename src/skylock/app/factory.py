"""Component factory for constructing pipelines and sessions from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skylock.config.models import SkyLockConfig
from skylock.control.controller import PointingController
from skylock.core.enums import InputKind
from skylock.core.interfaces import FrameSource
from skylock.core.pipeline import TrackingPipeline
from skylock.simulation.gimbal import VirtualGimbal
from skylock.simulation.source import SimulationSource

if TYPE_CHECKING:
    from skylock.app.session import Session


def build_pipeline(config: SkyLockConfig) -> TrackingPipeline:
    """Build and initialize a TrackingPipeline instance.

    Args:
        config: Root SkyLock configuration.

    Returns:
        Configured TrackingPipeline.
    """
    return TrackingPipeline(config)


def create_session_components(
    config: SkyLockConfig,
    sat_id: str = "s1",
) -> tuple[FrameSource, TrackingPipeline, PointingController]:
    """Create concrete instances of source, pipeline, and controller for a Session.

    Args:
        config: Root SkyLock configuration.
        sat_id: Satellite identifier ('s1' or 's2').

    Returns:
        Tuple of (source, pipeline, controller).
    """
    sat_key = sat_id.lower().replace("-", "")

    if config.input.kind in (InputKind.SIMULATION, "simulation"):
        from dataclasses import replace

        # If s2, use distinct seed so disturbances and target instances don't collide
        eff_config = (
            replace(config, seed=config.seed + 100)
            if (sat_key == "s2" and config.seed is not None)
            else config
        )

        pipeline = build_pipeline(eff_config)
        controller = PointingController(
            eff_config.control,
            eff_config.camera,
            max_slew_rate_deg_s=eff_config.gimbal.slew_rate_deg_s,
            sat_id=sat_key,
        )
        gimbal = VirtualGimbal(eff_config.gimbal)
        source: FrameSource = SimulationSource(eff_config, gimbal=gimbal)
    elif config.input.kind in (InputKind.ORBITAL, "orbital"):
        from dataclasses import replace

        from skylock.core.sim_clock import get_shared_clock
        from skylock.simulation.orbital_source import OrbitalSource

        peer_key = "s2" if sat_key == "s1" else "s1"
        eff_config = (
            replace(config, seed=config.seed + 100)
            if (sat_key == "s2" and config.seed is not None)
            else config
        )
        # Live gimbal limits: pan +-180 wrap, tilt +-90, default max slew 10 deg/s
        slew = (
            10.0
            if eff_config.gimbal.slew_rate_deg_s == 5.0
            else min(10.0, eff_config.gimbal.slew_rate_deg_s)
        )
        live_gimbal = replace(
            eff_config.gimbal,
            pan_limit_deg=(-180.0, 180.0),
            tilt_limit_deg=(-90.0, 90.0),
            slew_rate_deg_s=slew,
            max_slew_rate_deg_s=10.0,
        )
        eff_config = replace(eff_config, gimbal=live_gimbal)

        pipeline = build_pipeline(eff_config)
        controller = PointingController(
            eff_config.control,
            eff_config.camera,
            max_slew_rate_deg_s=eff_config.gimbal.slew_rate_deg_s,
            sat_id=sat_key,
        )
        source = OrbitalSource(
            eff_config,
            observer_id=sat_key,
            peer_id=peer_key,
            clock=get_shared_clock(),
        )
    elif config.input.kind in (InputKind.MP4, "mp4"):
        from dataclasses import replace

        from skylock.input.video import Mp4Source

        mp4_source = Mp4Source(config.input)
        mp4_source.open()

        w = mp4_source.width
        h = mp4_source.height
        fov_h = config.input.mp4_assumed_fov_h_deg
        fov_v = fov_h * (float(h) / float(w))

        effective_camera = replace(
            config.camera,
            width=w,
            height=h,
            fov_h_deg=fov_h,
            fov_v_deg=fov_v,
            fps=mp4_source.fps,
            allow_below_spec_fps=True,
        )
        effective_config = replace(config, camera=effective_camera)
        pipeline = TrackingPipeline(effective_config)
        controller = PointingController(
            config.control,
            effective_camera,
            max_slew_rate_deg_s=config.gimbal.slew_rate_deg_s,
            sat_id=sat_key,
        )
        source = mp4_source
    else:
        raise ValueError(f"Unknown input kind: {config.input.kind}")

    return source, pipeline, controller


def build_session(config: SkyLockConfig, sat_id: str = "s1") -> Session:
    """Build a complete tracking and control Session from configuration.

    Args:
        config: Root SkyLock configuration.
        sat_id: Satellite identifier ('s1' or 's2').

    Returns:
        Fully initialized Session instance.
    """
    from skylock.app.session import Session

    source, pipeline, controller = create_session_components(config, sat_id=sat_id)
    return Session(
        config=config,
        source=source,
        pipeline=pipeline,
        controller=controller,
        sat_id=sat_id,
    )


__all__ = (
    "build_pipeline",
    "build_session",
    "create_session_components",
)
