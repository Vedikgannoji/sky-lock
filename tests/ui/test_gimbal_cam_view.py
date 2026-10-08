"""Tests for GimbalCamView (3D POV camera view) and dual 3D view consistency."""

from __future__ import annotations

import math
import os
import sys
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from skylock.core.enums import TrackState
from skylock.core.los import get_satellite_position
from skylock.ui.panels.camera_feed_view import CameraFeedView
from skylock.ui.web3d.gimbal_cam_view import GimbalCamView
from skylock.ui.web3d.server import Embedded3DServer
from skylock.ui.web3d.view_3d import SpaceView3D
from skylock.ui.worker import FrameView


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication instance exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


def test_gimbal_cam_view_properties(qapp):
    """Test GimbalCamView properties, pose, fov, mount, and pause states."""
    view = GimbalCamView()
    assert view.mount == "s1"
    assert view.is_paused is False
    assert view.last_pan == 0.0
    assert view.last_tilt == 0.0
    assert view.last_fov == 3.0

    # Update pose
    view.set_pose(15.5, -7.2)
    assert view.last_pan == 15.5
    assert view.last_tilt == -7.2

    # Update FOV
    view.set_fov(5.0)
    assert view.last_fov == 5.0

    # Update mount
    view.set_mount("s2")
    assert view.mount == "s2"

    # Update pause
    view.set_paused(True)
    assert view.is_paused is True

    # Check state dictionary
    state = view.get_state()
    assert state["mount"] == "s2"
    assert state["pan"] == 15.5
    assert state["tilt"] == -7.2
    assert state["fov"] == 5.0
    assert state["is_paused"] is True

    # Reset
    view.reset_pose()
    assert view.last_pan == 0.0
    assert view.last_tilt == 0.0
    assert view.last_fov == 3.0


def test_shared_embedded_server_multi_bundle(qapp):
    """Test that Embedded3DServer serves both bundles and maps /gimbal correctly."""
    server = Embedded3DServer.get_shared_server()
    assert server is not None
    server.start()
    assert server.port is not None
    assert server.get_url("/index.html").startswith("http://127.0.0.1:")
    assert server.get_url("/gimbal/index.html").endswith("/gimbal/index.html")


def test_dual_view_positional_consistency():
    """Verify that both 3D views compute identical satellite coordinates matching Python ground truth.

    Both Three.js scenes (space_view and gimbal_view) compute orbital position as:
        currentAngle = initialAngle + speed * t;
        x = R * cos(currentAngle);
        y = R * sin(currentAngle) * sin(inc);
        z = R * sin(currentAngle) * cos(inc);
    """
    timestamps = [0.0, 1.0, 5.0, 12.5, 30.0, 60.0, 120.0]

    from skylock.core.orbital_world import ORBIT_SPEED_SCALE

    orbit_configs = {
        "s1": {"r": 20.0, "speed": 0.3 * ORBIT_SPEED_SCALE, "inc": math.radians(25.0), "phase": math.radians(0.0)},
        "s2": {"r": 26.0, "speed": 0.2 * ORBIT_SPEED_SCALE, "inc": math.radians(65.0), "phase": math.radians(45.0)},
    }

    for sat_id, cfg in orbit_configs.items():
        r = cfg["r"]
        speed = cfg["speed"]
        inc = cfg["inc"]
        phase = cfg["phase"]

        for t in timestamps:
            # 1. Python core authoritative calculation
            py_pos = get_satellite_position(sat_id, t)

            # 2. JavaScript OrbitState calculation formula
            angle = phase + speed * t
            js_x = r * math.cos(angle)
            js_y = r * math.sin(angle) * math.sin(inc)
            js_z = r * math.sin(angle) * math.cos(inc)

            assert math.isclose(py_pos[0], js_x, rel_tol=1e-5, abs_tol=1e-5), (
                f"Mismatch at t={t} for {sat_id} X: {py_pos[0]} vs {js_x}"
            )
            assert math.isclose(py_pos[1], js_y, rel_tol=1e-5, abs_tol=1e-5), (
                f"Mismatch at t={t} for {sat_id} Y: {py_pos[1]} vs {js_y}"
            )
            assert math.isclose(py_pos[2], js_z, rel_tol=1e-5, abs_tol=1e-5), (
                f"Mismatch at t={t} for {sat_id} Z: {py_pos[2]} vs {js_z}"
            )


def test_camera_feed_view_mounts_gimbal_3d_view(qapp):
    """Verify CameraFeedView mounts GimbalCamView and synchronizes observer and frames."""
    panel = CameraFeedView()
    assert hasattr(panel, "gimbal_3d_view")
    assert isinstance(panel.gimbal_cam, GimbalCamView)
    assert panel.gimbal_cam.mount == "s1"

    # Switch observer to S-2
    panel.combo_observer.setCurrentIndex(1)
    assert panel.selected_satellite == "s2"
    assert panel.gimbal_cam.mount == "s2"

    # Send frame for s2
    fv = FrameView(
        image=np.zeros((64, 64), dtype=np.uint8),
        frame_index=1,
        timestamp_s=2.5,
        track_state=TrackState.TRACK,
        detections=(),
        estimate=None,
        gate_px=10.0,
        boresight_px=(32.0, 32.0),
        pointing_pan_deg=12.0,
        pointing_tilt_deg=-4.0,
        ground_truth_px=None,
        is_simulation=True,
        fps_pipeline=30.0,
        fps_wall=30.0,
        latency_ms=10.0,
        acquisition_time_s=None,
        tracking_error_px=None,
        is_locked=True,
        control_mode="AUTO",
        command_pan_rate=0.0,
        command_tilt_rate=0.0,
        dropped_ui_frames=0,
        sat_id="s2",
    )
    panel.on_dual_frame_ready("s2", fv)
    assert panel.gimbal_cam.last_pan == 12.0
    assert panel.gimbal_cam.last_tilt == -4.0


def test_pause_synchronization(qapp):
    """Test pause state synchronization across SpaceView3D and GimbalCamView."""
    space_view = SpaceView3D()
    feed_view = CameraFeedView()

    # Wire up pause signals as in MainWindow
    space_view.pause_toggled.connect(feed_view.set_paused)
    feed_view.pause_toggled.connect(space_view.set_paused)

    assert space_view._is_paused is False
    assert feed_view._is_paused is False
    assert feed_view.gimbal_cam.is_paused is False

    # Pause from space_view
    space_view.set_paused(True)
    assert space_view._is_paused is True
    assert feed_view._is_paused is True
    assert feed_view.gimbal_cam.is_paused is True

    # Resume from feed_view
    feed_view.set_paused(False)
    assert space_view._is_paused is False
    assert feed_view._is_paused is False
    assert feed_view.gimbal_cam.is_paused is False
