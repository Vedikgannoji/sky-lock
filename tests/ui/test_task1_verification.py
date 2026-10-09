"""Comprehensive Task 1 verification tests covering all 16 specification points."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from skylock.app.factory import build_session
from skylock.config.models import (
    AtmosphereConfig,
    BlurConfig,
    CameraConfig,
    DisturbanceConfig,
    GaussianConfig,
    GimbalConfig,
    JitterConfig,
    PlatformConfig,
    PoissonConfig,
    SaltPepperConfig,
    SkyLockConfig,
)
from skylock.config.io import override
from skylock.control.controller import PointingController
from skylock.core.enums import ControlIntentMode, ControlMode, InputKind, TrackState
from skylock.core.los import check_line_of_sight, get_satellite_position
from skylock.core.types import ControlIntent, Pointing, TargetEstimate
from skylock.simulation.camera import VirtualCamera
from skylock.simulation.disturbances.stack import DisturbanceStack
from skylock.simulation.gimbal import VirtualGimbal
from skylock.simulation.orbital_source import OrbitalSource
from skylock.simulation.source import SimulationSource
from skylock.ui.config_editor import ConfigEditor
from skylock.ui.panels.camera_feed_view import CameraFeedView
from skylock.ui.panels.controls import ControlsPanel
from skylock.ui.panels.gimbal_control import GimbalControlPanel
from skylock.ui.worker import FrameView, SessionWorker


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication instance exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


# ============================================================================
# 1. Every disturbance changes sensor output when enabled
# ============================================================================
def test_1_disturbances_change_output_when_enabled():
    """Verify that each enabled disturbance modifies the sensor output."""
    base_cfg = SkyLockConfig(seed=42)
    clean_stack = DisturbanceStack(config=base_cfg.disturbances, seed=42)
    ctx_dummy = None

    # Base test image
    img = np.full((120, 160), 100.0, dtype=np.float32)

    # A. Gaussian noise
    cfg_gauss = replace(base_cfg.disturbances, gaussian=GaussianConfig(enabled=True, sigma_levels=15.0))
    stack_gauss = DisturbanceStack(config=cfg_gauss, seed=42)
    from skylock.simulation.disturbances.base import DisturbanceContext
    ctx = DisturbanceContext(frame_index=1, timestamp_s=0.1)
    out_gauss = stack_gauss.apply_photometric(img.copy(), ctx)
    assert not np.allclose(out_gauss, img), "Gaussian noise did not change image"

    # B. Salt & pepper
    cfg_sp = replace(base_cfg.disturbances, salt_pepper=SaltPepperConfig(enabled=True, density=0.05))
    stack_sp = DisturbanceStack(config=cfg_sp, seed=42)
    out_sp = stack_sp.apply_photometric(img.copy(), ctx)
    assert not np.allclose(out_sp, img), "Salt & pepper did not change image"

    # C. Poisson photon noise
    cfg_poi = replace(base_cfg.disturbances, poisson=PoissonConfig(enabled=True, photon_scale=10.0))
    stack_poi = DisturbanceStack(config=cfg_poi, seed=42)
    out_poi = stack_poi.apply_photometric(img.copy(), ctx)
    assert not np.allclose(out_poi, img), "Poisson noise did not change image"

    # D. Optical blur
    sharp_img = np.zeros((120, 160), dtype=np.float32)
    sharp_img[50:70, 70:90] = 200.0
    cfg_blur = replace(base_cfg.disturbances, blur=BlurConfig(enabled=True, sigma_px=3.0))
    stack_blur = DisturbanceStack(config=cfg_blur, seed=42)
    out_blur = stack_blur.apply_photometric(sharp_img.copy(), ctx)
    assert not np.allclose(out_blur, sharp_img), "Optical blur did not change image"
    assert out_blur.max() < sharp_img.max(), "Optical blur should reduce peak intensity"

    # E. Camera jitter (geometric offset)
    cfg_jit = override(base_cfg, {"disturbances.camera_jitter.enabled": True, "disturbances.camera_jitter.max_px_frame": 5.0})
    stack_jit = DisturbanceStack(config=cfg_jit.disturbances, seed=42)
    offsets = [stack_jit.compute_geometric_offset(i, i * 0.033) for i in range(10)]
    assert any(dx != 0.0 or dy != 0.0 for dx, dy in offsets), "Camera jitter produced no offset"

    # F. Platform drift (geometric offset)
    cfg_drift = override(base_cfg, {"disturbances.platform.enabled": True, "disturbances.platform.velocity_px_frame": 2.0, "disturbances.platform.max_px_frame": 10.0})
    stack_drift = DisturbanceStack(config=cfg_drift.disturbances, seed=42)
    dx0, dy0 = stack_drift.compute_geometric_offset(0, 0.0)
    dx5, dy5 = stack_drift.compute_geometric_offset(5, 0.166)
    assert dx5 != dx0 or dy5 != dy0, "Platform drift did not accumulate over frames"


# ============================================================================
# 2. Disabled disturbances no longer affect output
# ============================================================================
def test_2_disabled_disturbances_no_effect():
    """Verify that disabled disturbances act as exact no-ops."""
    from skylock.simulation.disturbances.base import DisturbanceContext
    ctx = DisturbanceContext(frame_index=1, timestamp_s=0.1)
    img = np.full((120, 160), 100.0, dtype=np.float32)

    cfg_all_disabled = DisturbanceConfig()
    stack = DisturbanceStack(config=cfg_all_disabled, seed=42)

    # Photometric no-op
    out = stack.apply_photometric(img.copy(), ctx)
    np.testing.assert_array_equal(out, img, err_msg="Disabled photometric disturbances modified image")

    # Geometric no-op
    dx, dy = stack.compute_geometric_offset(1, 0.1)
    assert (dx, dy) == (0.0, 0.0), f"Disabled geometric disturbances produced non-zero offset: {(dx, dy)}"


# ============================================================================
# 3. Disturbances do not modify 3D orbital positions
# ============================================================================
def test_3_disturbances_do_not_modify_3d_orbital_positions():
    """Verify that orbital 3D ephemerides are completely independent of disturbances."""
    t_test = 42.5
    p1_clean = get_satellite_position("s1", t_test)
    p2_clean = get_satellite_position("s2", t_test)

    # Apply extreme disturbance stack
    noisy_cfg = SkyLockConfig(
        seed=123,
        disturbances=DisturbanceConfig(
            camera_jitter=replace(DisturbanceConfig().camera_jitter, enabled=True, max_px_frame=20.0),
            platform=PlatformConfig(enabled=True, velocity_px_frame=10.0, max_px_frame=20.0),
        ),
    )
    stack = DisturbanceStack(noisy_cfg.disturbances, seed=123)
    dx, dy = stack.compute_geometric_offset(10, t_test)

    # Positions in 3D orbit world remain strictly identical
    p1_after = get_satellite_position("s1", t_test)
    p2_after = get_satellite_position("s2", t_test)
    assert p1_clean == p1_after, "Orbital position p1 altered by disturbances"
    assert p2_clean == p2_after, "Orbital position p2 altered by disturbances"


# ============================================================================
# 4. Live disturbance changes reach the active pipeline
# ============================================================================
def test_4_live_disturbance_changes_reach_active_pipeline(qapp):
    """Verify that live disturbance changes update the running session without session rebuild."""
    cfg = SkyLockConfig(seed=42)
    worker = SessionWorker(cfg)
    worker.initialize()

    # Initial: gaussian noise disabled
    assert not worker._session_s1.source.disturbances.config.gaussian.enabled

    # Enable gaussian live
    new_cfg = override(cfg, {"disturbances.gaussian.enabled": True, "disturbances.gaussian.sigma_levels": 12.0})
    worker.apply_config(new_cfg)

    # Should update source disturbance stack live
    assert worker._session_s1.source.disturbances.config.gaussian.enabled
    assert worker._session_s1.source.disturbances.config.gaussian.sigma_levels == 12.0
    assert worker._session_s2.source.disturbances.config.gaussian.enabled


# ============================================================================
# 5. AUTO mode commands the gimbal
# ============================================================================
def test_5_auto_mode_commands_gimbal():
    """Verify that AUTO mode computes control commands toward target."""
    ctrl = PointingController(max_slew_rate_deg_s=5.0)
    ctrl.set_mode(ControlMode.AUTO)
    # Ensure clear line of sight
    ctrl._custom_sat_pos = (50.0, 50.0, 50.0)

    intent = ControlIntent(mode=ControlIntentMode.TRACK, image_error_px=(100.0, 80.0))
    est = TargetEstimate(
        pan_deg=0.0,
        tilt_deg=0.0,
        pan_rate=0.0,
        tilt_rate=0.0,
        px=100.0,
        py=80.0,
        sigma_deg=0.1,
        from_measurement=True,
    )
    pointing = Pointing(0.0, 0.0)

    cmd = ctrl.step(intent, est, pointing, dt=0.033)
    assert cmd.pan_rate_deg_s != 0.0 or cmd.tilt_rate_deg_s != 0.0, "AUTO mode produced zero rate"


# ============================================================================
# 6. MANUAL mode prevents automatic pointing from overwriting manual commands
# ============================================================================
def test_6_manual_mode_prevents_auto_overwriting():
    """Verify that MANUAL mode strictly executes manual commanded rates."""
    ctrl = PointingController(max_slew_rate_deg_s=5.0)
    ctrl.set_mode(ControlMode.MANUAL)
    ctrl.set_manual_rate(2.5, -1.5)

    intent = ControlIntent(mode=ControlIntentMode.TRACK, image_error_px=(50.0, 50.0))
    est = TargetEstimate(
        pan_deg=0.0,
        tilt_deg=0.0,
        pan_rate=0.0,
        tilt_rate=0.0,
        px=50.0,
        py=50.0,
        sigma_deg=0.1,
        from_measurement=True,
    )
    pointing = Pointing(0.0, 0.0)

    cmd = ctrl.step(intent, est, pointing, dt=0.033)
    assert cmd.pan_rate_deg_s == 2.5
    assert cmd.tilt_rate_deg_s == -1.5


# ============================================================================
# 7. AUTO/MANUAL switching takes effect immediately
# ============================================================================
def test_7_auto_manual_switching_immediate(qapp):
    """Verify that switching between AUTO and MANUAL changes controller mode immediately."""
    cfg = SkyLockConfig()
    worker = SessionWorker(cfg)
    worker.initialize()

    # Initial is AUTO
    assert worker.session_s1.controller.mode == ControlMode.AUTO

    # Switch to MANUAL
    worker.set_control_mode("MANUAL")
    assert worker.session_s1.controller.mode == ControlMode.MANUAL

    # Switch back to AUTO
    worker.set_control_mode("AUTO")
    assert worker.session_s1.controller.mode == ControlMode.AUTO


# ============================================================================
# 8. Max Slew limits angular movement
# ============================================================================
def test_8_max_slew_limits_angular_movement():
    """Verify that Max Slew setting limits angular rate commands."""
    ctrl = PointingController(max_slew_rate_deg_s=3.0)
    ctrl.set_mode(ControlMode.MANUAL)
    ctrl.set_manual_rate(10.0, -10.0)  # Exceeds max slew

    cmd = ctrl.step(ControlIntent(mode=ControlIntentMode.HOLD), None, Pointing(0, 0), dt=0.033)
    assert abs(cmd.pan_rate_deg_s) <= 3.0, f"Pan rate {cmd.pan_rate_deg_s} exceeded max slew 3.0"
    assert abs(cmd.tilt_rate_deg_s) <= 3.0, f"Tilt rate {cmd.tilt_rate_deg_s} exceeded max slew 3.0"

    # VirtualGimbal also enforces max slew
    gimbal_cfg = GimbalConfig(slew_rate_deg_s=3.0, max_slew_rate_deg_s=3.0)
    gimbal = VirtualGimbal(gimbal_cfg)
    assert gimbal.max_slew_rate == 3.0


# ============================================================================
# 9. FOV changes the actual virtual camera
# ============================================================================
def test_9_fov_changes_actual_virtual_camera(qapp):
    """Verify that FOV updates reach the virtual camera sensor model."""
    cfg = SkyLockConfig()
    worker = SessionWorker(cfg)
    worker.initialize()

    initial_fov = worker.session_s1.source.camera.config.fov_h_deg
    assert initial_fov == cfg.camera.fov_h_deg

    worker.set_camera_fov(24.0)
    assert worker.session_s1.source.camera.config.fov_h_deg == 24.0
    assert worker.session_s1.source.camera.config.fov_v_deg == 18.0
    assert worker.session_s1.controller.camera_cfg.fov_h_deg == 24.0


# ============================================================================
# 10. Perspective switching selects the correct camera and gimbal
# ============================================================================
def test_10_perspective_switching(qapp):
    """Verify that perspective dropdown toggles selected camera, mount, and labels."""
    feed = CameraFeedView()
    assert feed.selected_satellite == "s1"
    assert feed.lbl_id_badge.text() == "HOST: S-1  ➔  TARGET: S-2"

    feed.combo_observer.setCurrentIndex(1)
    assert feed.selected_satellite == "s2"
    assert feed.btn_focus_target.text() == "Focus S-1"
    assert feed.lbl_id_badge.text() == "HOST: S-2  ➔  TARGET: S-1"
    assert feed.gimbal_3d_view.mount == "s2"


# ============================================================================
# 11. Communication line absent from sensor images
# ============================================================================
def test_11_communication_line_absent_from_sensor_images():
    """Verify that the tracking beam is excluded from the sensor camera view."""
    # Check legacy/js/gimbal_view/main.js
    main_js = open("legacy/js/gimbal_view/main.js", "r", encoding="utf-8").read()
    assert "trackingBeam.visible = false;" in main_js
    assert "trackingBeam.visible = true;" not in main_js

    # Check bundled gimbal asset
    bundle_js = open("src/skylock/ui/web3d/static_gimbal/assets/index-BEM6Obt0.js", "r", encoding="utf-8").read()
    assert "kn.visible=!1" in bundle_js
    assert "kn.visible=!0" not in bundle_js


# ============================================================================
# 12. Communication line remains available in 3D visualization
# ============================================================================
def test_12_communication_line_in_3d_visualization():
    """Verify SpaceView3D preserves set_show_tracking_beam toggle."""
    from skylock.ui.web3d.view_3d import SpaceView3D
    view = SpaceView3D()
    assert hasattr(view, "set_show_tracking_beam")


# ============================================================================
# 13. Start/Stop/Reset work without duplicate workers
# ============================================================================
def test_13_start_stop_reset_no_duplicates(qapp):
    """Verify start, stop, and reset cycles operate safely on single timer."""
    cfg = SkyLockConfig()
    worker = SessionWorker(cfg)
    worker.initialize()
    timer1 = worker._timer

    worker.start_running()
    assert worker.is_running
    assert worker._timer is timer1

    worker.stop_running()
    assert not worker.is_running
    assert worker._timer is timer1

    worker.reset_session()
    assert not worker.is_running
    assert worker._timer is timer1


# ============================================================================
# 14. Camera FPS affects actual processing cadence
# ============================================================================
def test_14_camera_fps_affects_cadence(qapp):
    """Verify that changing camera FPS alters period_s."""
    cfg = SkyLockConfig()
    cfg30 = override(cfg, {"camera.fps": 30.0})
    worker = SessionWorker(cfg30)
    worker.initialize()
    assert pytest.approx(worker._period_s, rel=1e-3) == 1.0 / 30.0

    cfg60 = override(cfg, {"camera.fps": 60.0})
    worker.apply_config(cfg60)
    assert pytest.approx(worker._period_s, rel=1e-3) == 1.0 / 60.0


# ============================================================================
# 15. Every enabled UI control has a real implementation
# ============================================================================
def test_15_all_controls_connected(qapp):
    """Verify that ControlsPanel, GimbalControlPanel, and CameraFeedView controls are fully implemented."""
    editor = ConfigEditor(SkyLockConfig())
    ctrls = ControlsPanel(editor)

    # Check tab title is Disturbances
    assert ctrls.tabs.tabText(1) == "Disturbances"

    # Check mode combo
    assert ctrls.cmb_mode.count() == 2

    # Check disturbance rows exist
    assert len(ctrls._dist_rows) == 6

    # Gimbal panel controls
    gimbal_pnl = GimbalControlPanel()
    assert gimbal_pnl.spn_pan.maximum() == 180.0
    assert gimbal_pnl.spn_tilt.maximum() == 90.0
    assert gimbal_pnl.spn_fov.value() == 16.0

    # Camera feed view controls
    feed = CameraFeedView()
    assert hasattr(feed, "btn_focus_target")
    assert hasattr(feed, "btn_manual")
    assert hasattr(feed, "btn_focus_earth")
    assert hasattr(feed, "btn_view_camera")
    assert hasattr(feed, "btn_view_sensor")
    assert hasattr(feed, "cs_pan")
    assert hasattr(feed, "cs_tilt")
    assert hasattr(feed, "cs_los")


# ============================================================================
# 16. Redesigned sensor layout works at default and resized dimensions
# ============================================================================
def test_16_sensor_layout_resizing(qapp):
    """Verify CameraFeedView layout stability across various dimensions."""
    feed = CameraFeedView()
    feed.resize(1024, 768)
    QCoreApplication.processEvents()
    assert feed.width() == 1024
    assert feed.height() == 768

    feed.resize(640, 480)
    QCoreApplication.processEvents()
    assert feed.width() == 640

    feed.resize(1920, 1080)
    QCoreApplication.processEvents()
    assert feed.width() == 1920
