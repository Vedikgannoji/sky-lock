"""Lightweight real-time metrics tracking for the live GUI.

Deterministic, pure-python tracker that computes running state counts,
durations, lock retention, loss events, and reacquisition timings.
NO Qt imports allowed.
"""

from __future__ import annotations

import math
from typing import Any

from skylock.core.enums import TrackState


class LiveMetrics:
    """Accumulates per-run live tracking statistics across frames."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to initial conditions."""
        self._frames: int = 0
        self._state_frames: dict[str, int] = {
            TrackState.SEARCH.name: 0,
            TrackState.ACQUIRE.name: 0,
            TrackState.TRACK.name: 0,
            TrackState.LOST.name: 0,
            TrackState.REACQUIRE.name: 0,
        }
        self._first_track_seen: bool = False
        self._post_first_track_frames: int = 0
        self._post_first_track_track_frames: int = 0

        self._loss_events: int = 0
        self._in_loss: bool = False
        self._loss_start_s: float | None = None
        self._current_loss_s: float | None = None
        self._last_reacq_s: float | None = None

        self._detections_total: int = 0

        # Tracking error aggregates (over frames with non-None tracking_error_px)
        self._err_count: int = 0
        self._err_sum: float = 0.0
        self._err_sq_sum: float = 0.0
        self._err_max: float | None = None

        self._prev_state: TrackState | None = None

    def update(
        self,
        state: TrackState,
        timestamp_s: float,
        n_detections: int,
        tracking_error_px: float | None,
        blocked: bool = False,
    ) -> None:
        """Process a single frame update.

        Args:
            state: Tracking state machine state for this frame.
            timestamp_s: Timestamp of current frame in seconds.
            n_detections: Number of detections in this frame.
            tracking_error_px: Tracking error against ground truth in pixels, or None.
            blocked: Whether line of sight is currently blocked (not a tracking failure).
        """
        self._frames += 1
        self._detections_total += max(0, n_detections)

        # State frame counts
        state_name = state.name if isinstance(state, TrackState) else str(state)
        if state_name in self._state_frames:
            self._state_frames[state_name] += 1
        else:
            self._state_frames[state_name] = 1

        # First track check
        if state == TrackState.TRACK:
            self._first_track_seen = True

        if self._first_track_seen:
            self._post_first_track_frames += 1
            if state == TrackState.TRACK:
                self._post_first_track_track_frames += 1

        # Loss and reacquisition tracking
        # Reacquisition duration = timestamp of first TRACK after a LOST (or REACQUIRE) period
        # minus timestamp of the TRACK->LOST transition.
        if self._prev_state is not None:
            # Transition into loss: TRACK -> LOST or REACQUIRE (not counted if blocked by Earth LOS)
            if (
                not blocked
                and self._prev_state == TrackState.TRACK
                and state in (TrackState.LOST, TrackState.REACQUIRE)
            ):
                if not self._in_loss:
                    self._in_loss = True
                    self._loss_events += 1
                    self._loss_start_s = timestamp_s

            # Transition out of loss back to TRACK
            elif self._in_loss and state == TrackState.TRACK:
                if self._loss_start_s is not None:
                    self._last_reacq_s = max(0.0, timestamp_s - self._loss_start_s)
                self._in_loss = False
                self._loss_start_s = None
                self._current_loss_s = None

        # If currently in loss, track current loss elapsed time
        if self._in_loss and self._loss_start_s is not None:
            self._current_loss_s = max(0.0, timestamp_s - self._loss_start_s)
        else:
            self._current_loss_s = None

        # Tracking error statistics
        if tracking_error_px is not None and not math.isnan(tracking_error_px):
            self._err_count += 1
            self._err_sum += tracking_error_px
            self._err_sq_sum += tracking_error_px * tracking_error_px
            if self._err_max is None or tracking_error_px > self._err_max:
                self._err_max = tracking_error_px

        self._prev_state = state

    def snapshot(self) -> dict[str, Any]:
        """Return a read-only snapshot dictionary of accumulated live metrics."""
        lock_retention: float | None = None
        if self._first_track_seen and self._post_first_track_frames > 0:
            lock_retention = self._post_first_track_track_frames / self._post_first_track_frames

        err_mean: float | None = None
        err_rms: float | None = None
        err_max: float | None = self._err_max
        if self._err_count > 0:
            err_mean = self._err_sum / self._err_count
            err_rms = math.sqrt(max(0.0, self._err_sq_sum / self._err_count))

        return {
            "frames": self._frames,
            "state_frames": dict(self._state_frames),
            "lock_retention": lock_retention,
            "loss_events": self._loss_events,
            "last_reacq_s": self._last_reacq_s,
            "current_loss_s": self._current_loss_s,
            "err_mean": err_mean,
            "err_rms": err_rms,
            "err_max": err_max,
            "detections_total": self._detections_total,
        }


__all__ = ("LiveMetrics",)
