"""Tests for Phase 4: Functional 3D Space, Camera/Gimbal, and Optical Tracking Integration."""

from __future__ import annotations

import math

from PySide6.QtWidgets import QApplication

from skylock.core.los import (
    EARTH_RADIUS,
    OrbitParams,
    calculate_look_angles,
    check_line_of_sight,
    is_target_in_fov,
    orbit_position_at_time,
    slew_toward_target,
)
from skylock.ui.main_window import MainWindow
from skylock.ui.panels.configuration_view import ConfigurationView
from skylock.ui.panels.gimbal_control import GimbalControlPanel
from skylock.ui.widgets.no_wheel import NoWheelDoubleSpinBox

# ===========================================================================
# 1. Configuration View Tests
# ===========================================================================

def test_configuration_tab_exists(qapp: QApplication) -> None:
    """Verify Configuration tab is mounted as Tab 2 in MainWindow view_tabs."""
    win = MainWindow()
    win.resize(1280, 720)
    win.show()
    qapp.processEvents()

    assert hasattr(win, "view_tabs")
    assert win.view_tabs.count() == 3
    assert win.view_tabs.tabText(0) == "3D Space Simulation"
    assert win.view_tabs.tabText(1) == "Camera Sensor Feed"
    assert win.view_tabs.tabText(2) == "Configuration"
    assert hasattr(win, "configuration_view")
    assert isinstance(win.configuration_view, ConfigurationView)

    win.space_view_3d.cleanup()
    win.close()


def test_configuration_view_toggles(qapp: QApplication) -> None:
    """Verify orbit lines and track line toggles on trimmed configuration tab."""
    config_view = ConfigurationView()
    config_view.show()
    qapp.processEvents()

    # Check toggles exist and default to True
    assert config_view.chk_orbit_lines.isChecked()
    assert config_view.chk_track_line.isChecked()
    assert config_view.chk_tracking_beam.isChecked()

    # Verify signals fire on toggle
    events: list[tuple[str, bool]] = []
    config_view.orbit_lines_toggled.connect(lambda v: events.append(("orbit", v)))
    config_view.track_line_toggled.connect(lambda v: events.append(("track", v)))

    config_view.chk_orbit_lines.setChecked(False)
    config_view.chk_track_line.setChecked(False)

    assert ("orbit", False) in events
    assert ("track", False) in events

    config_view.close()


# ===========================================================================
# 2. Satellite Placement and Physical Orbits Tests
# ===========================================================================

def test_satellite_orbital_physics() -> None:
    """Verify satellites remain strictly on their calculated physical orbits."""
    # Test circular inclined orbit: distance to origin must constantly equal radius
    orbit = OrbitParams(radius=24.0, speed=0.25, inclination_deg=45.0, phase_deg=30.0)

    for step in range(50):
        t_s = step * 0.5
        pos = orbit_position_at_time(orbit, t_s)
        dist = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        assert abs(dist - 24.0) < 1e-9, f"Satellite deviated from orbit radius at t={t_s}"


def test_satellite_manual_placement_parameters(qapp: QApplication) -> None:
    """Verify manual configuration of satellite orbit radius (S-1 and S-2)."""
    win = MainWindow()
    win.show()
    qapp.processEvents()

    cfg = win.configuration_view
    assert isinstance(cfg.spn_s1_radius, NoWheelDoubleSpinBox)
    assert isinstance(cfg.spn_s2_radius, NoWheelDoubleSpinBox)

    # Change S-1 and S-2 radius parameters
    cfg.spn_s1_radius.setValue(22.0)
    cfg.spn_s2_radius.setValue(28.0)

    cfg.btn_apply.click()
    qapp.processEvents()

    assert win._s1_orbit.radius == 22.0
    assert win._s2_orbit.radius == 28.0

    # Also test pause button on 3D Space Simulation tab toolbar
    assert hasattr(win, "btn_pause_3d")
    assert win.btn_pause_3d is not None
    assert win.btn_pause_3d.text() == "⏸ Pause"
    win.btn_pause_3d.click()
    assert win.btn_pause_3d.text() == "▶ Resume"
    win.btn_pause_3d.click()
    assert win.btn_pause_3d.text() == "⏸ Pause"

    win.space_view_3d.cleanup()
    win.close()


# ===========================================================================
# 3. Geometric Line-of-Sight and Earth Occlusion Tests
# ===========================================================================

def test_line_of_sight_earth_occlusion() -> None:
    """Verify geometric line-of-sight against Earth sphere (radius = 10.0)."""
    # Case 1: Clear direct optical path across space
    p1 = (20.0, 0.0, 0.0)
    p2 = (0.0, 26.0, 0.0)
    assert check_line_of_sight(p1, p2, body_radius=EARTH_RADIUS) is True

    # Case 2: Directly through Earth center (blocked)
    p1_opp = (20.0, 0.0, 0.0)
    p2_opp = (-26.0, 0.0, 0.0)
    assert check_line_of_sight(p1_opp, p2_opp, body_radius=EARTH_RADIUS) is False

    # Case 3: Grazing occultation near Earth horizon
    # Midpoint of segment from (20, 9.9, 0) to (-20, 9.9, 0) is (0, 9.9, 0) -> < 10.0 (blocked)
    assert check_line_of_sight((20.0, 9.9, 0.0), (-20.0, 9.9, 0.0), body_radius=10.0) is False

    # Midpoint of segment from (20, 10.1, 0) to (-20, 10.1, 0) is (0, 10.1, 0) -> > 10.0 (clear)
    assert check_line_of_sight((20.0, 10.1, 0.0), (-20.0, 10.1, 0.0), body_radius=10.0) is True


def test_target_in_fov_geometry() -> None:
    """Verify FOV containment test for rectangular camera sensor frustum."""
    current_pan = 0.0
    current_tilt = 0.0
    fov_deg = 20.0

    # Inside FOV (within 13.3° horiz and 10.0° vert)
    assert is_target_in_fov(current_pan, current_tilt, 5.0, 4.0, fov_deg) is True

    # Outside FOV (pan too large)
    assert is_target_in_fov(current_pan, current_tilt, 25.0, 0.0, fov_deg) is False

    # Outside FOV (tilt too large)
    assert is_target_in_fov(current_pan, current_tilt, 0.0, 15.0, fov_deg) is False


def test_look_angles_calculation() -> None:
    """Verify azimuth (pan) and elevation (tilt) look angles from observer to target."""
    obs = (0.0, 0.0, 0.0)
    target_ahead = (0.0, 0.0, -20.0)
    pan, tilt, dist = calculate_look_angles(obs, target_ahead)
    assert abs(pan - 0.0) < 1e-6
    assert abs(tilt - 0.0) < 1e-6
    assert abs(dist - 20.0) < 1e-6

    target_up = (0.0, 10.0, -10.0)
    pan, tilt, _ = calculate_look_angles(obs, target_up)
    assert abs(pan - 0.0) < 1e-6
    assert abs(tilt - 45.0) < 1e-6


# ===========================================================================
# 4. Gimbal Control Panel & Slew Limitation Tests
# ===========================================================================

def test_gimbal_control_panel_layout_and_signals(qapp: QApplication) -> None:
    """Verify dedicated GimbalControlPanel mounted below Telemetry in Right Dock."""
    win = MainWindow()
    win.show()
    qapp.processEvents()

    assert hasattr(win, "gimbal_control_panel")
    panel = win.gimbal_control_panel
    assert isinstance(panel, GimbalControlPanel)
    assert isinstance(panel.spn_pan, NoWheelDoubleSpinBox)
    assert isinstance(panel.spn_tilt, NoWheelDoubleSpinBox)
    assert isinstance(panel.spn_fov, NoWheelDoubleSpinBox)

    # Test manual command propagation
    events: list[tuple[str, float]] = []
    panel.pan_changed.connect(lambda v: events.append(("pan", v)))
    panel.tilt_changed.connect(lambda v: events.append(("tilt", v)))
    panel.fov_changed.connect(lambda v: events.append(("fov", v)))

    panel.spn_pan.setValue(15.5)
    panel.spn_tilt.setValue(-8.2)
    panel.spn_fov.setValue(25.0)

    assert ("pan", 15.5) in events
    assert ("tilt", -8.2) in events
    assert ("fov", 25.0) in events

    win.space_view_3d.cleanup()
    win.close()


def test_gimbal_reset_pose(qapp: QApplication) -> None:
    """Verify Reset Camera restores neutral pose (pan=0°, tilt=0°, FOV=20°)."""
    win = MainWindow()
    win.show()
    qapp.processEvents()

    # Move to non-neutral pose
    win.gimbal_control_panel.spn_pan.setValue(45.0)
    win.gimbal_control_panel.spn_tilt.setValue(-20.0)
    win.gimbal_control_panel.spn_fov.setValue(35.0)

    # Click Reset
    win.gimbal_control_panel.btn_reset.click()
    qapp.processEvents()

    assert win.gimbal_control_panel.spn_pan.value() == 0.0
    assert win.gimbal_control_panel.spn_tilt.value() == 0.0
    assert win.gimbal_control_panel.spn_fov.value() == 16.0
    assert win._current_pan == 0.0
    assert win._current_tilt == 0.0

    win.space_view_3d.cleanup()
    win.close()


def test_auto_gimbal_slew_rate_limiter() -> None:
    """Verify AUTO mode steps toward target respecting Max Slew (°/s) without teleporting."""
    current_pan = 0.0
    current_tilt = 0.0
    target_pan = 20.0
    target_tilt = 10.0
    max_slew = 5.0  # deg/s
    dt = 1.0  # 1 second step

    # Step 1: should advance by at most 5.0 deg
    pan1, tilt1, aligned = slew_toward_target(
        current_pan, current_tilt, target_pan, target_tilt, max_slew, dt
    )
    assert abs(pan1 - 5.0) < 1e-6
    assert abs(tilt1 - 5.0) < 1e-6
    assert not aligned

    # Step 2: 10.0 deg
    pan2, tilt2, aligned = slew_toward_target(pan1, tilt1, target_pan, target_tilt, max_slew, dt)
    assert abs(pan2 - 10.0) < 1e-6
    assert abs(tilt2 - 10.0) < 1e-6
    assert not aligned

    # Step 3: pan 15.0 deg, tilt reaches 10.0 deg
    pan3, tilt3, aligned = slew_toward_target(pan2, tilt2, target_pan, target_tilt, max_slew, dt)
    assert abs(pan3 - 15.0) < 1e-6
    assert abs(tilt3 - 10.0) < 1e-6
    assert not aligned

    # Step 4: reaches 20.0 deg and aligns
    pan4, tilt4, aligned = slew_toward_target(pan3, tilt3, target_pan, target_tilt, max_slew, dt)
    assert abs(pan4 - 20.0) < 1e-6
    assert abs(tilt4 - 10.0) < 1e-6
    assert aligned


# ===========================================================================
# 5. Authoritative Tracking State Machine Tests
# ===========================================================================

def test_authoritative_tracking_state_machine(qapp: QApplication) -> None:
    """Verify deterministic state transitions (LOST, SEARCH, ACQUIRE, TRACK)."""
    win = MainWindow()
    win.show()
    qapp.processEvents()

    class DummyFrame:
        frame_index = 10
        wall_fps = 30.0
        input_kind = "simulation"
        pointing_pan_deg = None
        pointing_tilt_deg = None

    # Case A: When satellites are on opposite sides of Earth (LOST / UNLOCKED)
    win._s1_orbit = OrbitParams(radius=20.0, speed=0.0, inclination_deg=0.0, phase_deg=0.0)
    win._s2_orbit = OrbitParams(radius=20.0, speed=0.0, inclination_deg=0.0, phase_deg=180.0)
    win._on_frame_ready(DummyFrame())
    qapp.processEvents()

    assert "UNLOCKED" in win.telemetry_panel.lbl_lock.text()
    assert win.lbl_status_state.text() == "LOST"

    # Case B: When target is visible and aligned within FOV (TRACK / ENGAGED)
    win._s1_orbit = OrbitParams(radius=20.0, speed=0.0, inclination_deg=0.0, phase_deg=0.0)
    win._s2_orbit = OrbitParams(radius=26.0, speed=0.0, inclination_deg=0.0, phase_deg=0.0)
    target_pan, target_tilt, _ = calculate_look_angles((20.0, 0.0, 0.0), (26.0, 0.0, 0.0))
    win._current_pan = target_pan
    win._current_tilt = target_tilt

    win._on_frame_ready(DummyFrame())
    qapp.processEvents()

    assert "ENGAGED" in win.telemetry_panel.lbl_lock.text()
    assert win.lbl_status_state.text() == "TRACK"

    win.space_view_3d.cleanup()
    win.close()
