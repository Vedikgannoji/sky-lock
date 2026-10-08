"""Configuration validation rules and error collection for SkyLock."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skylock.config.models import (
        CameraConfig,
        ControlConfig,
        DetectionConfig,
        DisturbanceConfig,
        GimbalConfig,
        InputConfig,
        KalmanConfig,
        RequirementsConfig,
        ScreenConfig,
        SearchConfig,
        SkyLockConfig,
        TargetConfig,
        TargetSetConfig,
        TrackingConfig,
    )


class ConfigError(ValueError):
    """Aggregates all configuration validation errors."""

    def __init__(self, violations: list[str]) -> None:
        self.violations: tuple[str, ...] = tuple(violations)
        super().__init__(
            f"Configuration validation failed with {len(violations)} violation(s):\n  - "
            + "\n  - ".join(violations)
        )


def validate_screen(screen: ScreenConfig) -> list[str]:
    """Validate virtual screen configuration."""
    violations: list[str] = []
    if screen.width_px < 640:
        violations.append(f"screen.width_px must be >= 640, got {screen.width_px}")
    if screen.height_px < 480:
        violations.append(f"screen.height_px must be >= 480, got {screen.height_px}")
    return violations


def validate_camera(camera: CameraConfig) -> list[str]:
    violations: list[str] = []
    if camera.width < 32:
        violations.append(f"camera.width must be >= 32, got {camera.width}")
    if camera.height < 32:
        violations.append(f"camera.height must be >= 32, got {camera.height}")
    if not (0.0 < camera.fov_h_deg < 180.0):
        violations.append(f"camera.fov_h_deg must be in (0, 180), got {camera.fov_h_deg}")
    if not (0.0 < camera.fov_v_deg < 180.0):
        violations.append(f"camera.fov_v_deg must be in (0, 180), got {camera.fov_v_deg}")

    if camera.width > 0 and camera.height > 0 and camera.fov_h_deg > 0:
        ifov_h = camera.fov_h_deg / camera.width
        ifov_v = camera.fov_v_deg / camera.height
        if ifov_h > 0:
            square_ratio_diff = abs(ifov_h - ifov_v) / ifov_h
            if square_ratio_diff > 0.01:
                violations.append(
                    f"Square-pixel check failed: IFOV mismatch |ifov_h ({ifov_h:.6f}) - "
                    f"ifov_v ({ifov_v:.6f})| / ifov_h is {square_ratio_diff * 100:.2f}% (> 1%)"
                )

    if not camera.allow_below_spec_fps and camera.fps < 30.0:
        violations.append(
            f"camera.fps must be >= 30.0 (PS_SPEC §2), got {camera.fps}. "
            "Set allow_below_spec_fps=True for test-only overrides."
        )

    if camera.bit_depth not in (8, 16):
        violations.append(f"camera.bit_depth must be 8 or 16, got {camera.bit_depth}")

    if not (0.0 <= camera.background_level <= 255.0):
        violations.append(
            f"camera.background_level must be in [0, 255], got {camera.background_level}"
        )

    return violations


def validate_gimbal(gimbal: GimbalConfig) -> list[str]:
    violations: list[str] = []
    if not (0.0 < gimbal.slew_rate_deg_s <= gimbal.max_slew_rate_deg_s <= 10.0):
        violations.append(
            f"Gimbal slew rate constraint violated: expected 0 < slew_rate "
            f"({gimbal.slew_rate_deg_s}) <= max_slew_rate ({gimbal.max_slew_rate_deg_s}) <= 10.0"
        )
    if gimbal.accel_deg_s2 <= 0.0:
        violations.append(f"gimbal.accel_deg_s2 must be > 0, got {gimbal.accel_deg_s2}")
    if gimbal.pan_limit_deg[0] >= gimbal.pan_limit_deg[1]:
        violations.append(
            f"gimbal.pan_limit_deg must be ordered (min < max), got {gimbal.pan_limit_deg}"
        )
    if gimbal.tilt_limit_deg[0] >= gimbal.tilt_limit_deg[1]:
        violations.append(
            f"gimbal.tilt_limit_deg must be ordered (min < max), got {gimbal.tilt_limit_deg}"
        )
    if gimbal.substeps < 1:
        violations.append(f"gimbal.substeps must be >= 1, got {gimbal.substeps}")

    return violations


def validate_target(target: TargetConfig) -> list[str]:
    violations: list[str] = []
    if target.strict_spec and not (5 <= target.size_px <= 20):
        violations.append(
            f"target.size_px must be in [5, 20] px (PS_SPEC §4), got {target.size_px}. "
            "Set strict_spec=False for custom sizes."
        )
    if target.size_px < 1:
        violations.append(f"target.size_px must be >= 1, got {target.size_px}")
    if not (0.0 < target.brightness <= 255.0):
        violations.append(f"target.brightness must be in (0, 255], got {target.brightness}")
    if target.shape not in ("square", "disc", "gaussian", "cross", "custom_mask"):
        violations.append(
            f"target.shape must be one of ('square', 'disc', 'gaussian', 'cross', 'custom_mask'), "
            f"got '{target.shape}'"
        )
    if target.initial not in ("random", "fixed"):
        violations.append(f"target.initial must be 'random' or 'fixed', got '{target.initial}'")

    for idx, (t0, t1) in enumerate(target.visibility_windows):
        if not (0.0 <= t0 <= t1):
            violations.append(
                f"target.visibility_windows[{idx}] must satisfy 0 <= t0 <= t1, got ({t0}, {t1})"
            )

    return violations


def validate_target_set(target_set: TargetSetConfig) -> list[str]:
    violations: list[str] = []
    if target_set.count < 1:
        violations.append(f"target_set.count must be >= 1, got {target_set.count}")
    if len(target_set.targets) != target_set.count:
        violations.append(
            f"target_set.targets length ({len(target_set.targets)}) != count ({target_set.count})"
        )
    for idx, t in enumerate(target_set.targets):
        for v in validate_target(t):
            violations.append(f"target_set.targets[{idx}]: {v}")
    return violations


def validate_detection(detection: DetectionConfig) -> list[str]:
    violations: list[str] = []
    if detection.min_area_px < 1:
        violations.append(f"detection.min_area_px must be >= 1, got {detection.min_area_px}")
    if detection.max_area_px < detection.min_area_px:
        violations.append(
            f"detection.max_area_px ({detection.max_area_px}) must be >= "
            f"min_area_px ({detection.min_area_px})"
        )
    if detection.max_blobs < 1:
        violations.append(f"detection.max_blobs must be >= 1, got {detection.max_blobs}")
    if detection.blur_sigma < 0.0:
        violations.append(f"detection.blur_sigma must be >= 0.0, got {detection.blur_sigma}")
    if detection.threshold_k_sigma <= 0.0:
        violations.append(
            f"detection.threshold_k_sigma must be > 0.0, got {detection.threshold_k_sigma}"
        )
    if detection.abs_min_threshold < 0.0:
        violations.append(
            f"detection.abs_min_threshold must be >= 0.0, got {detection.abs_min_threshold}"
        )
    if detection.roi_margin_px < 0:
        violations.append(f"detection.roi_margin_px must be >= 0, got {detection.roi_margin_px}")
    return violations


def validate_kalman(kalman: KalmanConfig) -> list[str]:
    violations: list[str] = []
    if kalman.q_accel_deg_s2 <= 0.0:
        violations.append(f"kalman.q_accel_deg_s2 must be > 0, got {kalman.q_accel_deg_s2}")
    if kalman.r_meas_px <= 0.0:
        violations.append(f"kalman.r_meas_px must be > 0, got {kalman.r_meas_px}")
    if kalman.gate_sigma <= 0.0:
        violations.append(f"kalman.gate_sigma must be > 0, got {kalman.gate_sigma}")
    return violations


def validate_search(search: SearchConfig) -> list[str]:
    violations: list[str] = []
    if search.field_of_regard_deg[0] <= 0.0 or search.field_of_regard_deg[1] <= 0.0:
        violations.append(
            f"search.field_of_regard_deg must be positive, got {search.field_of_regard_deg}"
        )
    if not (0.0 <= search.raster_overlap < 1.0):
        violations.append(
            f"search.raster_overlap must be in [0, 1), got {search.raster_overlap}"
        )
    if search.scan_rate_deg_s <= 0.0:
        violations.append(f"search.scan_rate_deg_s must be > 0, got {search.scan_rate_deg_s}")
    return violations


def validate_tracking(tracking: TrackingConfig) -> list[str]:
    violations: list[str] = []
    if tracking.confirm_hits < 1:
        violations.append(f"tracking.confirm_hits must be >= 1, got {tracking.confirm_hits}")
    if tracking.confirm_window < tracking.confirm_hits:
        violations.append(
            f"tracking.confirm_window ({tracking.confirm_window}) must be >= "
            f"confirm_hits ({tracking.confirm_hits})"
        )
    if tracking.acquire_timeout_s <= 0.0:
        violations.append(
            f"tracking.acquire_timeout_s must be > 0, got {tracking.acquire_timeout_s}"
        )
    if tracking.lost_after_misses < 1:
        violations.append(
            f"tracking.lost_after_misses must be >= 1, got {tracking.lost_after_misses}"
        )
    if tracking.coast_max_s < 0.0:
        violations.append(f"tracking.coast_max_s must be >= 0, got {tracking.coast_max_s}")
    if tracking.reacquire_timeout_s <= 0.0:
        violations.append(
            f"tracking.reacquire_timeout_s must be > 0, got {tracking.reacquire_timeout_s}"
        )
    if tracking.reacquire_radius_deg <= 0.0:
        violations.append(
            f"tracking.reacquire_radius_deg must be > 0, got {tracking.reacquire_radius_deg}"
        )
    if tracking.association_gate_px <= 0.0:
        violations.append(
            f"tracking.association_gate_px must be > 0, got {tracking.association_gate_px}"
        )
    if tracking.ephemeris_error_deg < 0.0:
        violations.append(
            f"tracking.ephemeris_error_deg must be >= 0, got {tracking.ephemeris_error_deg}"
        )

    violations.extend(validate_kalman(tracking.kalman))
    violations.extend(validate_search(tracking.search))
    return violations


def validate_control(control: ControlConfig) -> list[str]:
    violations: list[str] = []
    if control.kp < 0.0 or control.ki < 0.0 or control.kd < 0.0 or control.kff < 0.0:
        violations.append("control gains (kp, ki, kd, kff) must be >= 0.0")
    if not (0.0 <= control.d_filter_alpha <= 1.0):
        violations.append(
            f"control.d_filter_alpha must be in [0, 1], got {control.d_filter_alpha}"
        )
    if control.integral_clamp < 0.0:
        violations.append(f"control.integral_clamp must be >= 0, got {control.integral_clamp}")
    if control.deadband_px < 0.0:
        violations.append(f"control.deadband_px must be >= 0, got {control.deadband_px}")
    if control.latency_frames < 0:
        violations.append(f"control.latency_frames must be >= 0, got {control.latency_frames}")
    return violations


def validate_disturbances(disturbances: DisturbanceConfig) -> list[str]:
    violations: list[str] = []
    sp = disturbances.salt_pepper
    if not (0.0 <= sp.density <= 1.0):
        violations.append(f"disturbances.salt_pepper.density must be in [0, 1], got {sp.density}")

    gauss = disturbances.gaussian
    # PS_SPEC §6: noise standard deviation max 20 (interpreted as 20 grey levels)
    if not (0.0 <= gauss.sigma_levels <= 20.0):
        violations.append(
            f"disturbances.gaussian.sigma_levels must be in [0, 20.0] (PS_SPEC §6), "
            f"got {gauss.sigma_levels}"
        )

    jit = disturbances.camera_jitter
    if abs(jit.max_px_frame) > 20.0:
        violations.append(
            f"disturbances.camera_jitter.max_px_frame must be <= 20 px/frame (PS_SPEC §6), "
            f"got {jit.max_px_frame}"
        )
    if not (0.0 <= jit.correlation <= 1.0):
        violations.append(
            f"disturbances.camera_jitter.correlation must be in [0, 1], got {jit.correlation}"
        )

    plat = disturbances.platform
    if isinstance(plat.velocity_px_frame, (int, float)):
        v_mag = abs(plat.velocity_px_frame)
    elif isinstance(plat.velocity_px_frame, tuple) and len(plat.velocity_px_frame) == 2:
        v_mag = math.hypot(plat.velocity_px_frame[0], plat.velocity_px_frame[1])
    else:
        v_mag = 21.0
        violations.append(
            f"disturbances.platform.velocity_px_frame must be float or (vx, vy), "
            f"got {plat.velocity_px_frame}"
        )
    if v_mag > 20.0 and not any("velocity_px_frame must be float" in v for v in violations):
        violations.append(
            f"disturbances.platform.velocity_px_frame must be <= 20 px/frame (PS_SPEC §6), "
            f"got {plat.velocity_px_frame}"
        )
    if abs(plat.max_px_frame) > 20.0:
        violations.append(
            f"disturbances.platform.max_px_frame must be <= 20 px/frame (PS_SPEC §6), "
            f"got {plat.max_px_frame}"
        )

    atm = disturbances.atmosphere
    valid_modes = ("clear", "haze", "fog", "rain", "low_light")
    if atm.mode not in valid_modes:
        violations.append(
            f"disturbances.atmosphere.mode must be one of {valid_modes}, got '{atm.mode}'"
        )
    if not (0.0 <= atm.strength <= 1.0):
        violations.append(
            f"disturbances.atmosphere.strength must be in [0, 1], got {atm.strength}"
        )

    if disturbances.blur.sigma_px < 0.0:
        violations.append(
            f"disturbances.blur.sigma_px must be >= 0.0, got {disturbances.blur.sigma_px}"
        )

    return violations


def validate_requirements(req: RequirementsConfig) -> list[str]:
    violations: list[str] = []
    if req.acquisition_max_s <= 0.0:
        violations.append(
            f"requirements.acquisition_max_s must be > 0, got {req.acquisition_max_s}"
        )
    if req.tracking_error_px_max <= 0.0:
        violations.append(
            f"requirements.tracking_error_px_max must be > 0, got {req.tracking_error_px_max}"
        )
    if req.target_loss_rate_max <= 0.0:
        violations.append(
            f"requirements.target_loss_rate_max must be > 0, got {req.target_loss_rate_max}"
        )
    if req.reacquisition_max_s <= 0.0:
        violations.append(
            f"requirements.reacquisition_max_s must be > 0, got {req.reacquisition_max_s}"
        )
    if req.processing_fps_min <= 0.0:
        violations.append(
            f"requirements.processing_fps_min must be > 0, got {req.processing_fps_min}"
        )
    if req.lock_radius_px <= 0.0:
        violations.append(f"requirements.lock_radius_px must be > 0, got {req.lock_radius_px}")
    return violations


def validate_input(inp: InputConfig) -> list[str]:
    violations: list[str] = []
    if inp.kind not in ("simulation", "mp4", "orbital"):
        violations.append(f"input.kind must be 'simulation', 'mp4', or 'orbital', got '{inp.kind}'")
    if inp.mp4_assumed_fov_h_deg <= 0.0 or inp.mp4_assumed_fov_h_deg >= 180.0:
        violations.append(
            f"input.mp4_assumed_fov_h_deg must be in (0, 180), got {inp.mp4_assumed_fov_h_deg}"
        )
    if inp.fps_override is not None and inp.fps_override <= 0.0:
        violations.append(f"input.fps_override must be > 0.0, got {inp.fps_override}")
    return violations


def validate_root(cfg: SkyLockConfig) -> list[str]:
    """Validate all individual sections and cross-field constraints."""
    violations: list[str] = []

    # Section validations
    violations.extend(validate_screen(cfg.screen))
    violations.extend(validate_camera(cfg.camera))
    violations.extend(validate_gimbal(cfg.gimbal))
    violations.extend(validate_target_set(cfg.target))
    violations.extend(validate_detection(cfg.detection))
    violations.extend(validate_tracking(cfg.tracking))
    violations.extend(validate_control(cfg.control))
    violations.extend(validate_disturbances(cfg.disturbances))
    violations.extend(validate_input(cfg.input))
    violations.extend(validate_requirements(cfg.requirements))

    # Cross-field rules
    min_dim = min(cfg.camera.width, cfg.camera.height)
    for idx, target in enumerate(cfg.target.targets):
        if target.size_px >= min_dim / 4:
            violations.append(
                f"target[{idx}].size_px ({target.size_px}) must be < "
                f"min(width,height)/4 ({min_dim / 4})"
            )

    if cfg.target.targets:
        primary_size = cfg.target.targets[0].size_px
        if cfg.tracking.association_gate_px < primary_size:
            violations.append(
                f"tracking.association_gate_px ({cfg.tracking.association_gate_px}) must be >= "
                f"2 * size_px/2 ({primary_size}) of target"
            )

    if cfg.tracking.search.scan_rate_deg_s > cfg.gimbal.slew_rate_deg_s:
        violations.append(
            f"tracking.search.scan_rate_deg_s ({cfg.tracking.search.scan_rate_deg_s}) "
            f"must be <= gimbal.slew_rate_deg_s ({cfg.gimbal.slew_rate_deg_s})"
        )

    if cfg.tracking.reacquire_timeout_s > cfg.requirements.reacquisition_max_s:
        violations.append(
            f"tracking.reacquire_timeout_s ({cfg.tracking.reacquire_timeout_s}) "
            f"must be <= requirements.reacquisition_max_s ({cfg.requirements.reacquisition_max_s})"
        )

    if cfg.tracking.acquire_timeout_s > cfg.requirements.acquisition_max_s:
        violations.append(
            f"tracking.acquire_timeout_s ({cfg.tracking.acquire_timeout_s}) "
            f"must be <= requirements.acquisition_max_s ({cfg.requirements.acquisition_max_s})"
        )

    if cfg.control.deadband_px >= cfg.requirements.lock_radius_px:
        violations.append(
            f"control.deadband_px ({cfg.control.deadband_px}) "
            f"must be < requirements.lock_radius_px ({cfg.requirements.lock_radius_px})"
        )

    # Screen extent vs gimbal limits (simulation input only; orbital input operates on 3D sphere)
    if cfg.input.kind not in ("orbital", "Orbital"):
        px_per_deg = cfg.camera.px_per_deg
        left, right, bottom, top = cfg.screen.world_extent_deg(px_per_deg)

        if cfg.gimbal.pan_limit_deg[0] < left or cfg.gimbal.pan_limit_deg[1] > right:
            violations.append(
                f"gimbal.pan_limit_deg {cfg.gimbal.pan_limit_deg} extends beyond screen bounds "
                f"({left:.2f}, {right:.2f}) deg"
            )

        if cfg.gimbal.tilt_limit_deg[0] < bottom or cfg.gimbal.tilt_limit_deg[1] > top:
            violations.append(
                f"gimbal.tilt_limit_deg {cfg.gimbal.tilt_limit_deg} extends beyond screen bounds "
                f"({bottom:.2f}, {top:.2f}) deg"
            )

    return violations
