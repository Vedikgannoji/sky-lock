"""Background worker running the Session step loop in a dedicated QThread."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from skylock.app.factory import build_session
from skylock.app.session import Session
from skylock.config.models import SkyLockConfig
from skylock.core.enums import InputKind, TrackState
from skylock.core.types import StepResult
from skylock.ui.live_metrics import LiveMetrics


@dataclass(frozen=True, slots=True)
class FrameView:
    """Immutable UI presentation DTO emitted after each session step."""

    image: np.ndarray
    frame_index: int
    timestamp_s: float
    track_state: TrackState
    detections: tuple[tuple[float, float, float, float], ...]  # (cx, cy, w, h)
    estimate: tuple[float, float] | None  # (px, py)
    gate_px: float
    boresight_px: tuple[float, float]
    pointing_pan_deg: float
    pointing_tilt_deg: float
    ground_truth_px: tuple[float, float] | None
    is_simulation: bool
    fps_pipeline: float | None
    fps_wall: float | None
    latency_ms: float | None
    acquisition_time_s: float | None
    tracking_error_px: float | None
    is_locked: bool
    control_mode: str
    command_pan_rate: float
    command_tilt_rate: float
    dropped_ui_frames: int

    # G4 additions
    n_detections: int = 0
    best_detection_px: tuple[float, float] | None = None
    state_history_tail: tuple[tuple[float, str], ...] = ()
    live: dict = field(default_factory=dict)
    total_frames: int | None = None
    sat_id: str = "s1"
    cue_active: bool = False
    blocked: bool = False


class SessionWorker(QObject):
    """QObject executing dual tracking Sessions loop inside a QThread."""

    frame_ready = Signal(object)  # Emits FrameView for active satellite
    frame_ready_s1 = Signal(object)  # Emits FrameView for S-1
    frame_ready_s2 = Signal(object)  # Emits FrameView for S-2
    dual_frame_ready = Signal(str, object)  # Emits (sat_id, FrameView)
    session_error = Signal(str)
    running_changed = Signal(bool)
    session_rebuilt = Signal(str)  # Emitted after successful config rebuild
    session_finished = Signal(str)  # Emitted when stream ends

    def __init__(
        self,
        config: SkyLockConfig,
        parent: QObject | None = None,
        speed: float = 1.0,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._session_s1: Session | None = None
        self._session_s2: Session | None = None
        self._session: Session | None = None  # Kept for direct access / backwards compatibility
        self._active_sat: str = "s1"
        self._timer: QTimer | None = None
        self._is_running = False
        self._shutdown_requested = False
        self.speed = max(0.0, speed)  # Speed multiplier (>0), 0 => interval 0

        # Timing: single-shot timer with deadline tracking
        self._next_deadline: float = 0.0
        self._period_s: float = 1.0 / 30.0  # Will be set from config

        # S-1 Live metric accumulators
        self._live_metrics_s1 = LiveMetrics()
        self._state_history_s1: deque[tuple[float, str]] = deque(maxlen=120)
        self._first_track_time_s1: float | None = None
        self._first_obs_time_s1: float | None = None

        # S-2 Live metric accumulators
        self._live_metrics_s2 = LiveMetrics()
        self._state_history_s2: deque[tuple[float, str]] = deque(maxlen=120)
        self._first_track_time_s2: float | None = None
        self._first_obs_time_s2: float | None = None

        # Shared / legacy aliases
        self._live_metrics = self._live_metrics_s1
        self._state_history = self._state_history_s1
        self._first_track_time_s: float | None = None
        self._first_obs_time_s: float | None = None

        self._frame_counter = 0
        self._wall_frame_times: deque[float] = deque(maxlen=60)  # 2s window at 30fps
        self._latency_samples: deque[float] = deque(maxlen=30)

        # Control state per session
        self._manual_pan_rate_s1 = 0.0
        self._manual_tilt_rate_s1 = 0.0
        self._mode_str_s1 = "AUTO"

        self._manual_pan_rate_s2 = 0.0
        self._manual_tilt_rate_s2 = 0.0
        self._mode_str_s2 = "AUTO"

        # Legacy aliases
        self._manual_pan_rate = 0.0
        self._manual_tilt_rate = 0.0
        self._mode_str = "AUTO"

        # Latest views & results
        self._latest_frame_view_s1: FrameView | None = None
        self._latest_frame_view_s2: FrameView | None = None
        self._latest_step_result_s1: StepResult | None = None
        self._latest_step_result_s2: StepResult | None = None

        # Back-pressure
        self._pending_frames = 0
        self._dropped_ui_frames = 0

    @property
    def session(self) -> Session | None:
        """Alias for S-1 session (backward compatibility)."""
        return self._session_s1

    @property
    def session_s1(self) -> Session | None:
        """S-1 session instance (observing S-2)."""
        return self._session_s1

    @property
    def session_s2(self) -> Session | None:
        """S-2 session instance (observing S-1)."""
        return self._session_s2

    @property
    def active_satellite(self) -> str:
        """ID of the currently active/displayed satellite ('s1' or 's2')."""
        return self._active_sat

    def set_active_satellite(self, sat_id: str) -> None:
        """Set the active satellite ('s1' or 's2')."""
        sat_id_norm = sat_id.lower()
        if sat_id_norm in ("s1", "s2"):
            self._active_sat = sat_id_norm
            self._mode_str = (
                self._mode_str_s1 if sat_id_norm == "s1" else self._mode_str_s2
            )
            self._live_metrics = (
                self._live_metrics_s1 if sat_id_norm == "s1" else self._live_metrics_s2
            )
            self._state_history = (
                self._state_history_s1 if sat_id_norm == "s1" else self._state_history_s2
            )

    def get_session(self, sat_id: str) -> Session | None:
        """Return the Session instance for 's1' or 's2'."""
        sat_id_norm = sat_id.lower()
        if sat_id_norm == "s1":
            return self._session_s1
        elif sat_id_norm == "s2":
            return self._session_s2
        return None

    def get_tracking_state(self, sat_id: str = "s1") -> TrackState:
        """Get the current tracking state for the given satellite session."""
        sat_id_norm = sat_id.lower()
        sess = self._session_s1 if sat_id_norm == "s1" else self._session_s2
        if sess is not None and hasattr(sess, "tracker") and hasattr(sess.tracker, "state"):
            return sess.tracker.state
        latest_fv = (
            self._latest_frame_view_s1 if sat_id_norm == "s1" else self._latest_frame_view_s2
        )
        if latest_fv is not None:
            return latest_fv.track_state
        return TrackState.SEARCH

    def is_locked(self, sat_id: str = "s1") -> bool:
        """Return True if the specified satellite session is actively tracking."""
        return self.get_tracking_state(sat_id) == TrackState.TRACK

    def get_latest_frame_view(self, sat_id: str = "s1") -> FrameView | None:
        """Return the latest FrameView for the specified satellite."""
        sat_id_norm = sat_id.lower()
        return self._latest_frame_view_s1 if sat_id_norm == "s1" else self._latest_frame_view_s2

    def get_latest_step_result(self, sat_id: str = "s1") -> StepResult | None:
        """Return the latest StepResult for the specified satellite."""
        sat_id_norm = sat_id.lower()
        return self._latest_step_result_s1 if sat_id_norm == "s1" else self._latest_step_result_s2

    @Slot()
    def initialize(self) -> None:
        """Initialize session and timer within the target thread."""
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._step)
        self._build_new_session(self._config)

    @Slot()
    def shutdown(self) -> None:
        """Safely shut down the worker from within its thread."""
        self._shutdown_requested = True
        if self._timer is not None:
            self._timer.stop()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Whether the worker simulation loop is currently active."""
        return self._is_running

    def _build_new_session(self, cfg: SkyLockConfig) -> bool:
        """Rebuild tracking sessions from a new configuration."""
        try:
            self._session_s1 = build_session(cfg, sat_id="s1")
            self._session_s2 = build_session(cfg, sat_id="s2")
            self._session = self._session_s1
            self._config = cfg

            # Compute period for real-time pacing (no 0.9 factor)
            target_fps = max(1.0, cfg.camera.fps)
            self._period_s = 1.0 / target_fps

            # Reset timing and metrics
            self._next_deadline = 0.0
            self._live_metrics_s1.reset()
            self._live_metrics_s2.reset()
            self._state_history_s1.clear()
            self._state_history_s2.clear()
            self._first_track_time_s1 = None
            self._first_obs_time_s1 = None
            self._first_track_time_s2 = None
            self._first_obs_time_s2 = None
            self._first_track_time_s = None
            self._first_obs_time_s = None
            self._latest_frame_view_s1 = None
            self._latest_frame_view_s2 = None
            self._latest_step_result_s1 = None
            self._latest_step_result_s2 = None
            self._frame_counter = 0
            self._wall_frame_times.clear()
            self._latency_samples.clear()
            self._dropped_ui_frames = 0
            self._pending_frames = 0

            # Restore control mode and manual rates
            self._apply_control_state()

            return True
        except Exception as e:
            self.session_error.emit(f"Failed to build session: {e}")
            return False

    def _apply_control_state(self) -> None:
        """Apply stored control mode and manual rates to both sessions."""
        if self._session_s1 is not None:
            self._session_s1.set_mode(self._mode_str_s1)
            self._session_s1.set_manual_rates(
                self._manual_pan_rate_s1, self._manual_tilt_rate_s1
            )
        if self._session_s2 is not None:
            self._session_s2.set_mode(self._mode_str_s2)
            self._session_s2.set_manual_rates(
                self._manual_pan_rate_s2, self._manual_tilt_rate_s2
            )

    @Slot()
    def start_running(self) -> None:
        """Start the single-shot step timer with deadline-based pacing."""
        if self._is_running or self._shutdown_requested:
            return
        if self._session_s1 is None and not self._build_new_session(self._config):
            return
        if self._timer is None:
            self.initialize()
        assert self._timer is not None

        self._is_running = True
        self._next_deadline = time.perf_counter()
        self._frame_counter = 0
        self._wall_frame_times.clear()
        self._timer.start(0)  # Trigger first step immediately
        self.running_changed.emit(True)

    @Slot()
    def stop_running(self) -> None:
        """Stop the single-shot step timer."""
        if not self._is_running:
            return
        if self._timer is not None:
            self._timer.stop()
        self._is_running = False
        self.running_changed.emit(False)

    @Slot()
    def reset_session(self) -> None:
        """Reset both sessions to initial conditions."""
        was_running = self._is_running
        self.stop_running()
        if self._session_s1 is not None:
            self._session_s1.reset()
        if self._session_s2 is not None:
            self._session_s2.reset()
        self._live_metrics_s1.reset()
        self._live_metrics_s2.reset()
        self._state_history_s1.clear()
        self._state_history_s2.clear()
        self._first_track_time_s1 = None
        self._first_obs_time_s1 = None
        self._first_track_time_s2 = None
        self._first_obs_time_s2 = None
        self._first_track_time_s = None
        self._first_obs_time_s = None
        self._latest_frame_view_s1 = None
        self._latest_frame_view_s2 = None
        self._latest_step_result_s1 = None
        self._latest_step_result_s2 = None
        self._frame_counter = 0
        self._wall_frame_times.clear()
        self._latency_samples.clear()
        self._dropped_ui_frames = 0
        self._pending_frames = 0
        self._next_deadline = 0.0
        if was_running:
            self.start_running()

    def set_mode(self, sat_id_or_mode: str, mode: str | None = None) -> None:
        """Change control mode for a session without rebuilding.

        Supports:
          set_mode("s1", "EARTH")
          set_mode("s2", "AUTO")
          set_mode("MANUAL")  # applies to active satellite
        """
        if mode is None:
            sat_id = self._active_sat
            mode_str = sat_id_or_mode
        else:
            sat_id = sat_id_or_mode.lower()
            mode_str = mode

        mode_str_clean = mode_str.upper()
        if mode_str_clean == "EARTH_BORESIGHT":
            mode_str_clean = "EARTH"

        if sat_id == "s1":
            self._mode_str_s1 = mode_str_clean
            if self._active_sat == "s1":
                self._mode_str = mode_str_clean
            if self._session_s1 is not None:
                self._session_s1.set_mode(mode_str_clean)
        elif sat_id == "s2":
            self._mode_str_s2 = mode_str_clean
            if self._active_sat == "s2":
                self._mode_str = mode_str_clean
            if self._session_s2 is not None:
                self._session_s2.set_mode(mode_str_clean)

    @Slot(str)
    def set_control_mode(self, mode: str) -> None:
        """Change control mode for active satellite without rebuilding session."""
        self.set_mode(self._active_sat, mode)

    def set_manual_rates(self, *args: Any) -> None:
        """Set manual slew rates (deg/s) applied when in MANUAL mode.

        Supports:
          set_manual_rates(pan_rate: float, tilt_rate: float) -> active satellite
          set_manual_rates(sat_id: str, pan_rate: float, tilt_rate: float) -> target satellite
        """
        if len(args) == 2:
            sat_id = self._active_sat
            pan_rate = float(args[0])
            tilt_rate = float(args[1])
        elif len(args) == 3:
            sat_id = str(args[0]).lower()
            pan_rate = float(args[1])
            tilt_rate = float(args[2])
        else:
            raise TypeError(f"set_manual_rates expects 2 or 3 arguments, got {len(args)}")

        if sat_id == "s1":
            self._manual_pan_rate_s1 = pan_rate
            self._manual_tilt_rate_s1 = tilt_rate
            if self._active_sat == "s1":
                self._manual_pan_rate = pan_rate
                self._manual_tilt_rate = tilt_rate
            if self._session_s1 is not None:
                self._session_s1.set_manual_rates(pan_rate, tilt_rate)
        elif sat_id == "s2":
            self._manual_pan_rate_s2 = pan_rate
            self._manual_tilt_rate_s2 = tilt_rate
            if self._active_sat == "s2":
                self._manual_pan_rate = pan_rate
                self._manual_tilt_rate = tilt_rate
            if self._session_s2 is not None:
                self._session_s2.set_manual_rates(pan_rate, tilt_rate)

    def set_gimbal_pointing(self, sat_id: str, pan_deg: float, tilt_deg: float) -> None:
        """Command absolute gimbal pointing angles on the specified session."""
        session = self.get_session(sat_id)
        if session is not None and hasattr(session, "set_gimbal_pointing"):
            session.set_gimbal_pointing(pan_deg, tilt_deg)

    @Slot()
    def ack_frame(self) -> None:
        """Acknowledge that the UI has finished processing a frame."""
        if self._pending_frames > 0:
            self._pending_frames -= 1

    def update_disturbances(self, dist_cfg: Any) -> None:
        """Update disturbance parameters on running sessions without rebuild."""
        from dataclasses import replace

        self._config = replace(self._config, disturbances=dist_cfg)
        for sess in (self._session_s1, self._session_s2):
            if sess is not None:
                sess.config = replace(sess.config, disturbances=dist_cfg)
                if hasattr(sess.source, "disturbances") and hasattr(sess.source.disturbances, "update_config"):
                    sess.source.disturbances.update_config(dist_cfg)

    def set_camera_fov(self, fov_deg: float) -> None:
        """Update horizontal FOV on running virtual camera and controller."""
        from dataclasses import replace

        fov_h = float(fov_deg)
        fov_v = 0.75 * fov_h
        new_cam = replace(self._config.camera, fov_h_deg=fov_h, fov_v_deg=fov_v)
        self._config = replace(self._config, camera=new_cam)
        for sess in (self._session_s1, self._session_s2):
            if sess is not None:
                sess.config = replace(sess.config, camera=new_cam)
                if hasattr(sess.source, "camera"):
                    sess.source.camera.config = new_cam
                if hasattr(sess.controller, "camera_cfg"):
                    sess.controller.camera_cfg = new_cam

    def set_max_slew(self, slew_deg_s: float) -> None:
        """Update max slew rate on running controllers and gimbals."""
        from dataclasses import replace

        slew = min(10.0, max(0.1, float(slew_deg_s)))
        new_gimbal = replace(self._config.gimbal, slew_rate_deg_s=slew, max_slew_rate_deg_s=slew)
        self._config = replace(self._config, gimbal=new_gimbal)
        for sess in (self._session_s1, self._session_s2):
            if sess is not None:
                sess.config = replace(sess.config, gimbal=new_gimbal)
                if hasattr(sess.controller, "max_slew_rate_deg_s"):
                    sess.controller.max_slew_rate_deg_s = slew
                if hasattr(sess.source, "gimbal") and hasattr(sess.source.gimbal, "config"):
                    sess.source.gimbal.config = replace(
                        sess.source.gimbal.config, slew_rate_deg_s=slew, max_slew_rate_deg_s=slew
                    )

    @Slot(object)
    def apply_config(self, new_config: SkyLockConfig) -> None:
        """Apply a new SkyLockConfig.

        If only disturbances changed on running sessions, apply them live to avoid
        resetting the simulation time, orbits, and stateful RNG accumulators.
        Otherwise, rebuild both sessions cleanly.
        """
        only_disturbances = (
            self._session_s1 is not None
            and self._session_s2 is not None
            and self._config.input == new_config.input
            and self._config.camera == new_config.camera
            and self._config.target == new_config.target
            and self._config.control == new_config.control
            and self._config.gimbal == new_config.gimbal
            and self._config.seed == new_config.seed
            and self._config.disturbances != new_config.disturbances
        )
        if only_disturbances:
            self.update_disturbances(new_config.disturbances)
            return

        was_running = self._is_running
        self.stop_running()
        success = self._build_new_session(new_config)
        if success:
            self.session_rebuilt.emit("Configuration applied")
            if was_running:
                self.start_running()

    def _step(self) -> None:
        """Execute a single session step for both sessions with deadline-based pacing."""
        if (
            self._session_s1 is None and self._session_s2 is None
        ) or self._shutdown_requested:
            return

        res_s1: StepResult | None = None
        res_s2: StepResult | None = None

        try:
            if self._session_s1 is not None:
                res_s1 = self._session_s1.step()
            if self._session_s2 is not None:
                res_s2 = self._session_s2.step()
        except Exception as e:
            self.stop_running()
            self.session_error.emit(f"Step failed: {e}")
            return

        if res_s1 is None and res_s2 is None:
            # End of stream reached on both sessions
            self.stop_running()
            is_mp4 = self._config.input.kind in (InputKind.MP4, "mp4")
            msg = (
                f"End of stream after {self._frame_counter} frames"
                if is_mp4
                else "Run finished"
            )
            self.session_finished.emit(msg)
            return

        self._frame_counter += 1
        now = time.perf_counter()

        # Update wall FPS window
        self._wall_frame_times.append(now)

        fv_s1: FrameView | None = None
        fv_s2: FrameView | None = None

        if res_s1 is not None:
            self._latest_step_result_s1 = res_s1
            if res_s1.output.latency_ms > 0:
                self._latency_samples.append(res_s1.output.latency_ms)
            fv_s1 = self._make_frame_view(res_s1, "s1")
            self._latest_frame_view_s1 = fv_s1
            self.frame_ready_s1.emit(fv_s1)
            self.dual_frame_ready.emit("s1", fv_s1)

        if res_s2 is not None:
            self._latest_step_result_s2 = res_s2
            fv_s2 = self._make_frame_view(res_s2, "s2")
            self._latest_frame_view_s2 = fv_s2
            self.frame_ready_s2.emit(fv_s2)
            self.dual_frame_ready.emit("s2", fv_s2)

        # Back-pressure: skip emitting active frame_ready if UI is behind
        skip_emit = self._pending_frames >= 2
        if skip_emit:
            self._dropped_ui_frames += 1
        else:
            active_fv = fv_s1 if self._active_sat == "s1" else fv_s2
            if active_fv is not None:
                self._pending_frames += 1
                self.frame_ready.emit(active_fv)

        # Schedule next step with deadline-based pacing
        if self._is_running and not self._shutdown_requested:
            self._next_deadline += self._period_s / max(0.001, self.speed)
            delay_s = max(0.0, self._next_deadline - time.perf_counter())
            delay_ms = int(delay_s * 1000) if self.speed > 0 else 0
            if self._timer is not None:
                self._timer.start(delay_ms)

    def _make_frame_view(self, res: StepResult, sat_id: str = "s1") -> FrameView:
        """Convert a StepResult into an immutable UI FrameView."""
        out = res.output
        frame = res.frame
        is_sim = self._config.input.kind in (InputKind.SIMULATION, "simulation")

        is_s1 = sat_id == "s1"
        first_obs = self._first_obs_time_s1 if is_s1 else self._first_obs_time_s2
        first_track = self._first_track_time_s1 if is_s1 else self._first_track_time_s2
        live_metrics = self._live_metrics_s1 if is_s1 else self._live_metrics_s2
        state_history = self._state_history_s1 if is_s1 else self._state_history_s2
        mode_str = self._mode_str_s1 if is_s1 else self._mode_str_s2

        # Track first observable and acquisition times
        has_visible_truth = (
            is_sim and res.truth is not None and res.truth.primary_visible
        )
        if has_visible_truth and first_obs is None:
            first_obs = frame.timestamp_s
            if is_s1:
                self._first_obs_time_s1 = first_obs
                self._first_obs_time_s = first_obs
            else:
                self._first_obs_time_s2 = first_obs

        if out.state == TrackState.TRACK and first_track is None:
            first_track = frame.timestamp_s
            if is_s1:
                self._first_track_time_s1 = first_track
                self._first_track_time_s = first_track
            else:
                self._first_track_time_s2 = first_track

        acq_time_s: float | None = None
        if first_track is not None:
            ref_t = first_obs if first_obs is not None else 0.0
            acq_time_s = max(0.0, first_track - ref_t)

        # Tracking error px
        tracking_err: float | None = None
        gt_px: tuple[float, float] | None = None
        has_gt_position = (
            is_sim
            and res.truth is not None
            and res.truth.primary_visible
            and res.truth.primary_px is not None
        )
        if has_gt_position:
            gt_px = (
                float(res.truth.primary_px[0]),  # type: ignore[index]
                float(res.truth.primary_px[1]),  # type: ignore[index]
            )
            if out.state == TrackState.TRACK and out.estimate is not None:
                dx = out.estimate.px - gt_px[0]
                dy = out.estimate.py - gt_px[1]
                tracking_err = math.hypot(dx, dy)

        # Smoothed pipeline FPS from latency samples
        pipe_fps: float | None = None
        if len(self._latency_samples) > 0:
            mean_latency = sum(self._latency_samples) / len(self._latency_samples)
            if mean_latency > 0:
                pipe_fps = 1000.0 / mean_latency

        # Wall FPS from 2.0s window (minimum 5 frames)
        wall_fps: float | None = None
        if len(self._wall_frame_times) >= 5:
            window_duration = self._wall_frame_times[-1] - self._wall_frame_times[0]
            if window_duration > 0:
                wall_fps = (len(self._wall_frame_times) - 1) / window_duration

        # Boresight
        bx = self._config.camera.width / 2.0
        by = self._config.camera.height / 2.0

        # Detections
        det_tuples = tuple(
            (float(d.cx), float(d.cy), float(d.bbox[2]), float(d.bbox[3]))
            for d in out.detections
        )
        n_detections = len(out.detections)
        best_detection_px: tuple[float, float] | None = None
        if out.detections:
            # Rank detections by SNR / peak (see Detection fields: peak, snr, area_px)
            best_det = max(out.detections, key=lambda d: (d.snr, d.peak))
            best_detection_px = (float(best_det.cx), float(best_det.cy))

        # Estimate coords
        est_px = (
            (float(out.estimate.px), float(out.estimate.py)) if out.estimate else None
        )

        # Query tracker for cue and blocked flags
        sess = self._session_s1 if is_s1 else self._session_s2
        cue_active = False
        blocked = False
        if sess is not None and hasattr(sess, "pipeline") and hasattr(sess.pipeline, "tracker"):
            cue_active = bool(getattr(sess.pipeline.tracker, "cue_active", False))
            blocked = bool(getattr(sess.pipeline.tracker, "blocked", False))

        # Update LiveMetrics and state history
        live_metrics.update(
            state=out.state,
            timestamp_s=frame.timestamp_s,
            n_detections=n_detections,
            tracking_error_px=tracking_err,
            blocked=blocked,
        )
        state_history.append((frame.timestamp_s, out.state.name))

        # Pointing telemetry
        p_pan = frame.pointing.pan_deg if frame.pointing else 0.0
        p_tilt = frame.pointing.tilt_deg if frame.pointing else 0.0

        # Total frames from source if available
        total_frames: int | None = None
        if sess is not None and hasattr(sess, "source"):
            source = sess.source
            if hasattr(source, "frames_expected"):
                total_frames = source.frames_expected

        return FrameView(
            image=np.ascontiguousarray(frame.image.copy()),
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            track_state=out.state,
            detections=det_tuples,
            estimate=est_px,
            gate_px=float(self._config.tracking.association_gate_px),
            boresight_px=(bx, by),
            pointing_pan_deg=p_pan,
            pointing_tilt_deg=p_tilt,
            ground_truth_px=gt_px,
            is_simulation=is_sim,
            fps_pipeline=pipe_fps,
            fps_wall=wall_fps,
            latency_ms=out.latency_ms,
            acquisition_time_s=acq_time_s,
            tracking_error_px=tracking_err,
            is_locked=(out.state == TrackState.TRACK),
            control_mode=mode_str,
            command_pan_rate=float(res.command.pan_rate_deg_s),
            command_tilt_rate=float(res.command.tilt_rate_deg_s),
            dropped_ui_frames=self._dropped_ui_frames,
            n_detections=n_detections,
            best_detection_px=best_detection_px,
            state_history_tail=tuple(state_history),
            live=live_metrics.snapshot(),
            total_frames=total_frames,
            sat_id=sat_id,
            cue_active=cue_active,
            blocked=blocked,
        )


__all__ = ("FrameView", "SessionWorker")
