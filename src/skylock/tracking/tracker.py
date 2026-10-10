"""Target tracking orchestrator.

Coordinates:
- Pixel-to-sky coordinate transformations
- Multi-candidate association
- 4-state constant-velocity Kalman filter
- 5-state operational tracking state machine
- Dynamic region of interest (ROI) gating
- ControlIntent generation
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from skylock.config.models import CameraConfig, DetectionConfig, SkyLockConfig, TrackingConfig
from skylock.core.enums import ControlIntentMode, TrackState
from skylock.core.geometry import (
    angle_offset_to_pixel,
    angular_diff_deg,
    pixel_to_angle_offset,
    wrap_deg,
)
from skylock.core.types import Candidate, ControlIntent, Pointing, TargetEstimate
from skylock.tracking.candidate_tracker import CandidateTracker
from skylock.tracking.kalman import KalmanFilter
from skylock.tracking.search_patterns import LocalSpiral, RasterScan
from skylock.tracking.state_machine import StateMachine

if TYPE_CHECKING:
    from skylock.core.types import Detection, Frame


class Tracker:
    """Target tracker integrating CV candidate association, Kalman estimation, and FSM."""

    def __init__(
        self,
        config: SkyLockConfig | None = None,
        *,
        camera: CameraConfig | None = None,
        tracking: TrackingConfig | None = None,
        detection: DetectionConfig | None = None,
    ) -> None:
        """Initialize Tracker.

        Args:
            config: Complete Skylock configuration or None.
            camera: Camera geometry parameters.
            tracking: Tracking state machine and Kalman parameters.
            detection: Blob detection parameters (for ROI calculation).
        """
        if config is not None:
            self.camera = config.camera
            self.tracking_cfg = config.tracking
            self.detection_cfg = config.detection
            self.screen_cfg = config.screen
        else:
            self.camera = camera if camera is not None else CameraConfig()
            self.tracking_cfg = tracking if tracking is not None else TrackingConfig()
            self.detection_cfg = detection if detection is not None else DetectionConfig()
            self.screen_cfg = None

        self.candidate_tracker = CandidateTracker(self.tracking_cfg)
        self.kalman = KalmanFilter(self.tracking_cfg.kalman)
        self.state_machine = StateMachine(self.tracking_cfg, self.tracking_cfg.search)

        # Compute field of regard from screen bounds if available, else use config default
        if self.screen_cfg is not None:
            px_per_deg = self.camera.px_per_deg
            left, right, bottom, top = self.screen_cfg.world_extent_deg(px_per_deg)
            field_of_regard = (right - left, top - bottom)
        else:
            field_of_regard = self.tracking_cfg.search.field_of_regard_deg

        self.raster_scan = RasterScan(
            field_of_regard=field_of_regard,
            fov=(self.camera.fov_h_deg, self.camera.fov_v_deg),
            overlap=self.tracking_cfg.search.raster_overlap,
            scan_rate=self.tracking_cfg.search.scan_rate_deg_s,
        )
        self.spiral_scan: LocalSpiral | None = None

        self._last_estimate: TargetEstimate | None = None
        self._last_pointing: Pointing = Pointing(pan_deg=0.0, tilt_deg=0.0)

        # Ephemeris cueing and LOS occlusion state (live orbital sessions)
        self.cue: tuple[float, float] | None = None
        self.los_clear: bool = True
        self.blocked: bool = False
        self.cue_active: bool = False
        self._cue_aligned: bool = False
        self._cue_scan_configured: bool = False

    def set_cue(self, cue: tuple[float, float] | None, los_clear: bool = True) -> None:
        """Set ephemeris cue pointing setpoint and line-of-sight status."""
        self.cue = cue
        self.los_clear = bool(los_clear)

    @property
    def state(self) -> TrackState:
        """Current operational tracking state."""
        return self.state_machine.state

    @property
    def estimate(self) -> TargetEstimate | None:
        """Most recent target kinematic estimate."""
        return self._last_estimate

    def reset(self, t: float = 0.0) -> None:
        """Reset internal candidate tracks, Kalman filter, and state machine."""
        self.candidate_tracker.reset()
        self.kalman.reset()
        self.state_machine.reset(t)
        self.spiral_scan = None
        self._last_estimate = None
        self._last_pointing = Pointing(pan_deg=0.0, tilt_deg=0.0)
        self.cue = None
        self.los_clear = True
        self.blocked = False
        self.cue_active = False
        self._cue_aligned = False
        self._cue_scan_configured = False

    def get_roi(
        self,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[int, int, int, int] | None:
        """Compute search region of interest (ROI) in pixel coordinates.

        Uses roi_margin_px around Kalman predicted pixel in TRACK and LOST.
        Returns None (full frame) otherwise.

        Args:
            width: Image width (defaults to camera.width).
            height: Image height (defaults to camera.height).

        Returns:
            (x, y, w, h) bounding box tuple or None.
        """
        w_img = width if width is not None else self.camera.width
        h_img = height if height is not None else self.camera.height

        if self.state not in (TrackState.TRACK, TrackState.LOST):
            return None

        if not self.kalman.is_initialized:
            return None

        # Project predicted sky angle to camera pixel space
        pred_pan, pred_tilt = self.kalman.state[0], self.kalman.state[1]
        dpan = angular_diff_deg(pred_pan, self._last_pointing.pan_deg)
        dtilt = pred_tilt - self._last_pointing.tilt_deg

        try:
            px, py = angle_offset_to_pixel(dpan, dtilt, self.camera)
        except ZeroDivisionError:
            return None

        margin = self.detection_cfg.roi_margin_px
        x0 = max(0, math.floor(px - margin))
        y0 = max(0, math.floor(py - margin))
        x1 = min(w_img, math.ceil(px + margin))
        y1 = min(h_img, math.ceil(py + margin))

        if x1 > x0 and y1 > y0:
            return (x0, y0, x1 - x0, y1 - y0)
        return None

    def step(
        self,
        frame: Frame,
        detections: Sequence[Detection] = (),
    ) -> tuple[TrackState, TargetEstimate | None, Candidate | None, ControlIntent]:
        """Process one frame of detections and generate state estimate and control intent.

        Args:
            frame: Input video sensor frame.
            detections: Blob detections for this frame.

        Returns:
            Tuple of (current_state, target_estimate, selected_candidate, control_intent).
        """
        t = frame.timestamp_s
        pointing = (
            frame.pointing if frame.pointing is not None else Pointing(pan_deg=0.0, tilt_deg=0.0)
        )
        self._last_pointing = pointing

        # 0. Check LOS occlusion (live orbital sessions)
        if not self.los_clear:
            self.blocked = True
            self.cue_active = False
            self._cue_aligned = False
            self.state_machine.state = TrackState.LOST
            intent = ControlIntent(
                mode=ControlIntentMode.HOLD,
                setpoint_pan_deg=pointing.pan_deg,
                setpoint_tilt_deg=pointing.tilt_deg,
                image_error_px=None,
            )
            return (TrackState.LOST, None, None, intent)

        was_blocked = self.blocked
        self.blocked = False
        if was_blocked:
            self.kalman.reset()
            self.candidate_tracker.reset()
            self.state_machine.reset(t)
            self._cue_aligned = False
            self._cue_scan_configured = False

        prev_state = self.state_machine.state

        # 1. Kalman prediction in TRACK and LOST
        pred_px_py: tuple[float, float] | None = None
        if self.kalman.is_initialized and prev_state in (TrackState.TRACK, TrackState.LOST):
            self.kalman.predict(t)
            pred_pan, pred_tilt = self.kalman.state[0], self.kalman.state[1]
            dpan = angular_diff_deg(pred_pan, pointing.pan_deg)
            dtilt = pred_tilt - pointing.tilt_deg
            pred_px_py = angle_offset_to_pixel(dpan, dtilt, self.camera)

        # 2. Update CandidateTracker with detections
        candidates = self.candidate_tracker.update(
            list(detections),
            frame.index,
            predicted_px=pred_px_py,
        )

        # 3. Measurement selection
        selected_candidate: Candidate | None = None
        has_gated_measurement = False
        meas_sky: tuple[float, float] | None = None

        if prev_state in (TrackState.SEARCH, TrackState.ACQUIRE):
            active_cands = [c for c in candidates if c.misses == 0]
            confirmed_candidates = [c for c in active_cands if c.confirmed]
            if confirmed_candidates:
                selected_candidate = max(confirmed_candidates, key=lambda c: c.score)
            elif active_cands:
                selected_candidate = max(active_cands, key=lambda c: c.score)

            if selected_candidate is not None:
                dpan, dtilt = pixel_to_angle_offset(
                    selected_candidate.cx, selected_candidate.cy, self.camera
                )
                meas_sky = (wrap_deg(pointing.pan_deg + dpan), pointing.tilt_deg + dtilt)

        elif prev_state in (TrackState.TRACK, TrackState.LOST, TrackState.REACQUIRE):
            # Nearest to Kalman prediction inside gate
            if self.kalman.is_initialized:
                best_cand: Candidate | None = None
                best_dist = float("inf")
                best_sky: tuple[float, float] | None = None

                # Search among active candidates observed this frame
                for cand in candidates:
                    if cand.misses > 0:
                        continue
                    dpan, dtilt = pixel_to_angle_offset(cand.cx, cand.cy, self.camera)
                    z_sky = (wrap_deg(pointing.pan_deg + dpan), pointing.tilt_deg + dtilt)
                    gate_dist = self.kalman.mahalanobis(np.array(z_sky))

                    if gate_dist <= self.tracking_cfg.kalman.gate_sigma and gate_dist < best_dist:
                        best_dist = gate_dist
                        best_cand = cand
                        best_sky = z_sky

                if best_cand is not None and best_sky is not None:
                    selected_candidate = best_cand
                    has_gated_measurement = True
                    meas_sky = best_sky

                    # Convert pixel measurement noise to sky degrees squared
                    deg_per_px = self.camera.fov_h_deg / self.camera.width
                    r_deg2 = (self.tracking_cfg.kalman.r_meas_px * deg_per_px) ** 2
                    R = np.diag([r_deg2, r_deg2])
                    self.kalman.update(meas_sky, t, R=R)
            else:
                # If Kalman lost/reset in REACQUIRE, select best candidate observed this frame
                active_cands = [c for c in candidates if c.misses == 0]
                if active_cands:
                    selected_candidate = max(active_cands, key=lambda c: c.score)
                    dpan, dtilt = pixel_to_angle_offset(
                        selected_candidate.cx, selected_candidate.cy, self.camera
                    )
                    meas_sky = (wrap_deg(pointing.pan_deg + dpan), pointing.tilt_deg + dtilt)

        # 4. State Machine evaluation
        curr_state = self.state_machine.step(
            t=t,
            candidates=candidates,
            has_gated_measurement=has_gated_measurement,
            detections=detections,
        )

        # 5. Handle transitions
        if (
            prev_state != TrackState.TRACK
            and curr_state == TrackState.TRACK
            and not self.kalman.is_initialized
            and meas_sky is not None
        ):
            self.kalman.init(meas_sky, t)

        if curr_state == TrackState.REACQUIRE and prev_state != TrackState.REACQUIRE:
            # Initialize spiral scan centered at last estimated LOS
            if self.kalman.is_initialized:
                center_pan = self.kalman.state[0]
                center_tilt = self.kalman.state[1]
            elif meas_sky is not None:
                center_pan, center_tilt = meas_sky
            else:
                center_pan, center_tilt = pointing.pan_deg, pointing.tilt_deg

            overlap = self.tracking_cfg.search.raster_overlap
            spacing = self.camera.fov_v_deg * (1.0 - overlap)
            self.spiral_scan = LocalSpiral(
                center=(center_pan, center_tilt),
                max_radius=self.tracking_cfg.reacquire_radius_deg,
                spacing=spacing,
                rate=self.tracking_cfg.search.scan_rate_deg_s,
            )

        if curr_state == TrackState.SEARCH and prev_state != TrackState.SEARCH:
            self.kalman.reset()
            self.candidate_tracker.reset()
            self.spiral_scan = None
            # Recenter raster scan at current pointing to start search from where we are
            self.raster_scan.recenter((pointing.pan_deg, pointing.tilt_deg))

        # 6. Build TargetEstimate
        estimate: TargetEstimate | None = None
        if curr_state in (TrackState.TRACK, TrackState.LOST) and self.kalman.is_initialized:
            k_state = self.kalman.state
            pan_est, tilt_est, vpan_est, vtilt_est = (
                k_state[0],
                k_state[1],
                k_state[2],
                k_state[3],
            )
            dpan = angular_diff_deg(pan_est, pointing.pan_deg)
            dtilt = tilt_est - pointing.tilt_deg
            px_est, py_est = angle_offset_to_pixel(dpan, dtilt, self.camera)

            estimate = TargetEstimate(
                pan_deg=pan_est,
                tilt_deg=tilt_est,
                pan_rate=vpan_est,
                tilt_rate=vtilt_est,
                px=px_est,
                py=py_est,
                sigma_deg=self.kalman.position_sigma_deg,
                from_measurement=has_gated_measurement,
            )
        elif (
            curr_state == TrackState.ACQUIRE
            and selected_candidate is not None
            and meas_sky is not None
        ):
            estimate = TargetEstimate(
                pan_deg=meas_sky[0],
                tilt_deg=meas_sky[1],
                pan_rate=0.0,
                tilt_rate=0.0,
                px=selected_candidate.cx,
                py=selected_candidate.cy,
                sigma_deg=0.1,
                from_measurement=True,
            )

        self._last_estimate = estimate

        # 7. Generate ControlIntent
        boresight_x = (self.camera.width - 1.0) / 2.0
        boresight_y = (self.camera.height - 1.0) / 2.0

        if curr_state in (TrackState.ACQUIRE, TrackState.TRACK):
            # In ACQUIRE / TRACK the ephemeris cue is strictly ignored
            self.cue_active = False
            self._cue_aligned = False
            self._cue_scan_configured = False

            if estimate is not None:
                err_px = (estimate.px - boresight_x, estimate.py - boresight_y)
                intent = ControlIntent(
                    mode=ControlIntentMode.TRACK,
                    setpoint_pan_deg=estimate.pan_deg,
                    setpoint_tilt_deg=estimate.tilt_deg,
                    image_error_px=err_px,
                )
            else:
                intent = ControlIntent(
                    mode=ControlIntentMode.TRACK,
                    setpoint_pan_deg=pointing.pan_deg,
                    setpoint_tilt_deg=pointing.tilt_deg,
                    image_error_px=None,
                )

        elif self.cue is not None and curr_state in (
            TrackState.SEARCH,
            TrackState.LOST,
            TrackState.REACQUIRE,
        ):
            # CUE ACTIVE in SEARCH, LOST, or REACQUIRE
            self.cue_active = True
            cue_pan, cue_tilt = self.cue
            dpan_cue = abs(angular_diff_deg(cue_pan, pointing.pan_deg))
            dtilt_cue = abs(cue_tilt - pointing.tilt_deg)

            # Slew to cue first if not aligned
            if not self._cue_aligned and (dpan_cue > 0.5 or dtilt_cue > 0.5):
                intent = ControlIntent(
                    mode=ControlIntentMode.GOTO,
                    setpoint_pan_deg=cue_pan,
                    setpoint_tilt_deg=cue_tilt,
                    image_error_px=None,
                )
            else:
                self._cue_aligned = True
                # Run existing scan recentered on cue with field of regard = max(3 * sigma, FOV/2)
                sigma = float(getattr(self.tracking_cfg, "ephemeris_error_deg", 1.0))
                for_h = max(3.0 * sigma, self.camera.fov_h_deg / 2.0)
                for_v = max(3.0 * sigma, self.camera.fov_v_deg / 2.0)

                if (
                    not self._cue_scan_configured
                    or self.raster_scan.field_of_regard != (for_h, for_v)
                ):
                    self.raster_scan = RasterScan(
                        field_of_regard=(for_h, for_v),
                        fov=(self.camera.fov_h_deg, self.camera.fov_v_deg),
                        overlap=self.tracking_cfg.search.raster_overlap,
                        scan_rate=self.tracking_cfg.search.scan_rate_deg_s,
                        center=(cue_pan, cue_tilt),
                    )
                    self._cue_scan_configured = True
                else:
                    self.raster_scan.recenter((cue_pan, cue_tilt))

                elapsed_search = t - self.state_machine.state_start_time
                pan_sp, tilt_sp = self.raster_scan.setpoint(elapsed_search)
                intent = ControlIntent(
                    mode=ControlIntentMode.GOTO,
                    setpoint_pan_deg=pan_sp,
                    setpoint_tilt_deg=tilt_sp,
                    image_error_px=None,
                )

        elif curr_state == TrackState.SEARCH:
            self.cue_active = False
            elapsed_search = t - self.state_machine.state_start_time
            pan_sp, tilt_sp = self.raster_scan.setpoint(elapsed_search)
            intent = ControlIntent(
                mode=ControlIntentMode.GOTO,
                setpoint_pan_deg=pan_sp,
                setpoint_tilt_deg=tilt_sp,
                image_error_px=None,
            )

        elif curr_state == TrackState.REACQUIRE:
            self.cue_active = False
            elapsed_reacquire = t - self.state_machine.state_start_time
            if self.spiral_scan is not None:
                pan_sp, tilt_sp = self.spiral_scan.setpoint(elapsed_reacquire)
            else:
                pan_sp, tilt_sp = pointing.pan_deg, pointing.tilt_deg
            intent = ControlIntent(
                mode=ControlIntentMode.GOTO,
                setpoint_pan_deg=pan_sp,
                setpoint_tilt_deg=tilt_sp,
                image_error_px=None,
            )

        elif curr_state == TrackState.LOST:
            self.cue_active = False
            if estimate is not None:
                err_px = (estimate.px - boresight_x, estimate.py - boresight_y)
                intent = ControlIntent(
                    mode=ControlIntentMode.TRACK,
                    setpoint_pan_deg=estimate.pan_deg,
                    setpoint_tilt_deg=estimate.tilt_deg,
                    image_error_px=err_px,
                )
            else:
                intent = ControlIntent(
                    mode=ControlIntentMode.TRACK,
                    setpoint_pan_deg=pointing.pan_deg,
                    setpoint_tilt_deg=pointing.tilt_deg,
                    image_error_px=None,
                )
        else:
            self.cue_active = False
            intent = ControlIntent(
                mode=ControlIntentMode.HOLD,
                setpoint_pan_deg=pointing.pan_deg,
                setpoint_tilt_deg=pointing.tilt_deg,
                image_error_px=None,
            )

        return (curr_state, estimate, selected_candidate, intent)


__all__ = ("Tracker",)
