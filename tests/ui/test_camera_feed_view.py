"""Unit and integration tests for CameraFeedView panel."""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from skylock.config.models import SkyLockConfig
from skylock.core.enums import ControlMode, TrackState
from skylock.ui.main_window import ManualSteeringFilter
from skylock.ui.panels.camera_feed_view import CameraFeedView
from skylock.ui.worker import FrameView, SessionWorker


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QCoreApplication / QApplication instance exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


def _make_dummy_frame_view(
    sat_id: str = "s1",
    t_s: float = 0.0,
    is_locked: bool = False,
    state: TrackState = TrackState.SEARCH,
) -> FrameView:
    return FrameView(
        image=np.zeros((100, 100), dtype=np.uint8),
        frame_index=int(t_s * 30),
        timestamp_s=t_s,
        track_state=state,
        detections=(),
        estimate=None,
        gate_px=20.0,
        boresight_px=(50.0, 50.0),
        pointing_pan_deg=0.0,
        pointing_tilt_deg=0.0,
        ground_truth_px=None,
        is_simulation=True,
        fps_pipeline=30.0,
        fps_wall=30.0,
        latency_ms=10.0,
        acquisition_time_s=None,
        tracking_error_px=None,
        is_locked=is_locked,
        control_mode="AUTO",
        command_pan_rate=0.0,
        command_tilt_rate=0.0,
        dropped_ui_frames=0,
        sat_id=sat_id,
    )


def test_camera_feed_view_defaults(qapp) -> None:
    """Test initial UI state of CameraFeedView on fresh launch."""
    feed = CameraFeedView()
    assert feed.selected_satellite == "s1"
    assert feed.combo_observer.currentIndex() == 0
    assert feed.combo_observer.currentText() == "View S-2 from S-1"
    assert feed.btn_focus_target.text() == "Focus S-2"
    # Initial mode starts in AUTO
    assert feed.btn_focus_target.isChecked() is True
    assert feed.btn_focus_earth.isChecked() is False
    assert feed.btn_manual.isChecked() is False

    # Check status log initialized
    lines = feed.get_log_lines()
    assert len(lines) >= 1
    assert "SYSTEM READY" in lines[0]


def test_observer_switch_without_disturbing_sessions(qapp) -> None:
    """Verify switching observer swaps displayed feed without resetting sessions."""
    cfg = SkyLockConfig()
    worker = SessionWorker(cfg)
    worker._build_new_session(cfg)

    feed = CameraFeedView()
    feed.set_worker(worker)

    # Initial view is S-1
    assert feed.selected_satellite == "s1"
    assert feed.btn_focus_target.text() == "Focus S-2"

    # Step both sessions in worker
    worker._step()
    s1_t_before = worker.session_s1.controller.current_time_s
    s2_t_before = worker.session_s2.controller.current_time_s

    # Switch observer to S-2
    feed.combo_observer.setCurrentIndex(1)
    assert feed.selected_satellite == "s2"
    assert feed.btn_focus_target.text() == "Focus S-1"
    assert worker.active_satellite == "s2"

    # Step again
    worker._step()

    # S-1 must keep advancing and not be paused or disturbed
    assert worker.session_s1.controller.current_time_s >= s1_t_before
    assert worker.session_s2.controller.current_time_s >= s2_t_before


def test_focus_mode_controls_per_session(qapp) -> None:
    """Verify Earth, AUTO, and MANUAL focus modes apply per-session independently."""
    cfg = SkyLockConfig()
    worker = SessionWorker(cfg)
    worker._build_new_session(cfg)

    feed = CameraFeedView()
    steering_filter = ManualSteeringFilter()
    feed.set_worker(worker, steering_filter)

    # 1. On S-1: Switch to Focus Earth
    feed.btn_focus_earth.click()
    assert worker.session_s1.controller.mode == ControlMode.EARTH
    assert steering_filter.is_manual_mode is False

    # 2. Switch observer to S-2 (starts in AUTO)
    feed.combo_observer.setCurrentIndex(1)
    assert feed.selected_satellite == "s2"
    assert feed.btn_focus_target.text() == "Focus S-1"
    assert feed.btn_focus_target.isChecked() is True
    assert worker.session_s2.controller.mode == ControlMode.AUTO

    # 3. On S-2: Switch to Manual
    feed.btn_manual.click()
    assert worker.session_s2.controller.mode == ControlMode.MANUAL
    assert steering_filter.is_manual_mode is True

    # 4. Switch back to S-1
    feed.combo_observer.setCurrentIndex(0)
    assert feed.selected_satellite == "s1"
    assert feed.btn_focus_target.text() == "Focus S-2"
    # S-1's mode must be preserved as EARTH!
    assert feed.btn_focus_earth.isChecked() is True
    assert worker.session_s1.controller.mode == ControlMode.EARTH
    assert steering_filter.is_manual_mode is False


def test_status_feed_events_sequence(qapp) -> None:
    """Verify sequence of LOCKED, LOST, HANDSHAKE, and LINK BLOCKED events."""
    feed = CameraFeedView()

    # 1. Clear LOS initially at t=0
    fv_s1_0 = _make_dummy_frame_view("s1", t_s=0.0, is_locked=False)
    feed.on_dual_frame_ready("s1", fv_s1_0)
    logs = feed.get_log_lines()
    assert any("LINK ACTIVE" in line for line in logs)

    # 2. S-1 locks on S-2 at t=12.4s
    fv_s1_lock = _make_dummy_frame_view("s1", t_s=12.4, is_locked=True, state=TrackState.TRACK)
    feed.on_dual_frame_ready("s1", fv_s1_lock)
    logs = feed.get_log_lines()
    assert any("12.4s — S-1 LOCKED on S-2" in line for line in logs)

    # 3. S-2 locks on S-1 at t=13.0s -> triggers HANDSHAKE
    fv_s2_lock = _make_dummy_frame_view("s2", t_s=13.0, is_locked=True, state=TrackState.TRACK)
    feed.on_dual_frame_ready("s2", fv_s2_lock)
    logs = feed.get_log_lines()
    assert any("13.0s — S-2 LOCKED on S-1" in line for line in logs)
    assert any("HANDSHAKE ESTABLISHED" in line for line in logs)
    assert "ACTIVE" in feed.lbl_handshake_indicator.text()

    # 4. Advance to Earth occlusion (scaled by ORBIT_SPEED_SCALE)
    from skylock.core.orbital_world import ORBIT_SPEED_SCALE

    t_occ = 35.0 / ORBIT_SPEED_SCALE
    fv_s1_occ = _make_dummy_frame_view("s1", t_s=t_occ, is_locked=True, state=TrackState.TRACK)
    feed.on_dual_frame_ready("s1", fv_s1_occ)
    logs = feed.get_log_lines()
    assert any("LINK BLOCKED (Earth occlusion)" in line for line in logs)
    assert "BLOCKED" in feed.lbl_link_indicator.text()

    # 5. S-1 loses lock during occlusion -> HANDSHAKE LOST
    t_lost = 36.0 / ORBIT_SPEED_SCALE
    fv_s1_lost = _make_dummy_frame_view("s1", t_s=t_lost, is_locked=False, state=TrackState.LOST)
    feed.on_dual_frame_ready("s1", fv_s1_lost)
    logs = feed.get_log_lines()
    assert any(f"{t_lost:.1f}s — S-1 LOST S-2" in line for line in logs)
    assert any("HANDSHAKE LOST" in line for line in logs)
    assert "OFF" in feed.lbl_handshake_indicator.text()
