"""PySide6 3D Space View widget embedding WebGL Three.js simulation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from skylock.ui import theme
from skylock.ui.web3d.server import Embedded3DServer

logger = logging.getLogger(__name__)


class SpaceView3D(QWidget):
    """3D Space Visualization Widget embedding Earth, Satellites, Gimbal, and FOV frustum."""

    scene_ready = Signal()
    state_updated = Signal(dict)
    pause_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)

        self._server = Embedded3DServer.get_shared_server()
        self._server.start()

        self._is_ready = False
        self._is_paused = False
        self._last_pan = 90.0
        self._last_tilt = 0.0
        self._last_fov = 20.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3D Space Simulation Toolbar
        self.toolbar = QWidget(self)
        tb_layout = QHBoxLayout(self.toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(8)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setToolTip("Pause or resume 3D orbital motion")
        self.btn_pause.setStyleSheet(
            f"QPushButton {{ font-weight: bold; padding: 4px 14px; background: {theme.SLATE_BG.name()}; "
            f"color: {theme.TEXT_PRIMARY.name()}; border: 1px solid {theme.SKY_ACCENT.name()}; border-radius: 4px; }}"
            f"QPushButton:checked {{ background: {theme.RED_ACTIVE.name()}; border-color: {theme.STATUS_ERROR.name()}; }}"
            f"QPushButton:hover {{ background: {theme.SLATE_HOVER.name()}; }}"
        )
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        tb_layout.addWidget(self.btn_pause)

        btn_action_style = (
            f"QPushButton {{ padding: 4px 10px; background: {theme.SLATE_BG.name()}; color: {theme.SLATE_TEXT.name()}; "
            f"border: 1px solid {theme.SLATE_BORDER.name()}; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: {theme.SLATE_HOVER.name()}; border-color: {theme.SKY_ACCENT.name()}; }}"
        )

        self.btn_reset_cam = QPushButton("⟲ Reset Camera")
        self.btn_reset_cam.setToolTip("Reset camera to Earth overview")
        self.btn_reset_cam.setStyleSheet(btn_action_style)
        self.btn_reset_cam.clicked.connect(self.reset_view)
        tb_layout.addWidget(self.btn_reset_cam)

        self.btn_focus_s1 = QPushButton("Focus S-1")
        self.btn_focus_s1.setToolTip("Focus camera on Satellite 1")
        self.btn_focus_s1.setStyleSheet(btn_action_style)
        self.btn_focus_s1.clicked.connect(self.focus_s1)
        tb_layout.addWidget(self.btn_focus_s1)

        self.btn_focus_s2 = QPushButton("Focus S-2")
        self.btn_focus_s2.setToolTip("Focus camera on Satellite 2")
        self.btn_focus_s2.setStyleSheet(btn_action_style)
        self.btn_focus_s2.clicked.connect(self.focus_s2)
        tb_layout.addWidget(self.btn_focus_s2)

        tb_layout.addStretch(1)

        # FPS readout (numeric count only, green at high fps, turning red at low fps)
        self.lbl_fps = QLabel("60")
        self.lbl_fps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_fps.setToolTip("3D Space Simulation Rendering FPS (Target: 60)")
        self.lbl_fps.setStyleSheet(
            f"QLabel {{ color: {theme.FPS_GREEN.name()}; font-weight: bold; font-family: 'Consolas', 'Courier New', monospace; "
            f"font-size: 13px; padding: 2px 8px; background: {theme.DARK_BG.name()}; border: 1px solid {theme.BORDER_NORMAL.name()}; "
            f"border-radius: 4px; min-width: 24px; }}"
        )
        tb_layout.addWidget(self.lbl_fps)

        layout.addWidget(self.toolbar)

        # FPS update timer (queries WebGL 3D simulation engine)
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(400)
        self._fps_timer.timeout.connect(self._query_fps)
        self._fps_timer.start()

        self._web_view = QWebEngineView(self)
        layout.addWidget(self._web_view, stretch=1)

        # Configure WebEngine settings for hardware acceleration
        settings = self._web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        self._web_view.loadFinished.connect(self._on_load_finished)

        # Load 3D scene from embedded local server
        url = self._server.get_url()
        self._web_view.load(QUrl(url))

    @property
    def is_ready(self) -> bool:
        """Whether the WebGL 3D scene is loaded and ready."""
        return self._is_ready

    @property
    def current_pan(self) -> float:
        """Current gimbal pan in degrees."""
        return self._last_pan

    @property
    def current_tilt(self) -> float:
        """Current gimbal tilt in degrees."""
        return self._last_tilt

    @property
    def current_fov(self) -> float:
        """Current camera FOV in degrees."""
        return self._last_fov

    def _on_load_finished(self, success: bool) -> None:
        """Invoked when web view finishes loading."""
        if success:
            self._is_ready = True
            logger.info("SpaceView3D WebGL scene loaded successfully")
            self.scene_ready.emit()
            # Push initial values
            self.set_external_clock(True)
            self.set_gimbal_pose(self._last_pan, self._last_tilt)
            self.set_camera_fov(self._last_fov)
            # Force-remove loading overlay from Python side as safety measure
            self._web_view.page().runJavaScript(
                "var ov=document.getElementById('loading-overlay');"
                "if(ov){ov.style.opacity='0';setTimeout(function(){ov.remove();},400);}"
            )
        else:
            logger.warning("SpaceView3D failed to load WebGL scene")
            # Still mark ready so the app doesn't hang
            self._is_ready = True
            self.scene_ready.emit()

    def set_time(self, time_sec: float) -> None:
        """Synchronize the 3D orbital simulation time directly from the shared clock."""
        js = f"window.skylock3d?.setTime({float(time_sec)});"
        self._web_view.page().runJavaScript(js)

    def set_external_clock(self, enabled: bool) -> None:
        """Enable or disable external-clock mode."""
        js_val = "true" if enabled else "false"
        self._web_view.page().runJavaScript(f"window.skylock3d?.setExternalClock({js_val});")

    def set_gimbal_pose(self, pan_deg: float, tilt_deg: float) -> None:
        """Command the 3D gimbal pan and tilt angles in degrees."""
        self._last_pan = float(pan_deg)
        self._last_tilt = float(tilt_deg)
        js = f"window.skylock3d?.setGimbalPose({self._last_pan}, {self._last_tilt});"
        self._web_view.page().runJavaScript(js)

    def set_camera_fov(self, fov_deg: float) -> None:
        """Update the 3D camera FOV and frustum geometry in degrees."""
        self._last_fov = float(fov_deg)
        js = f"window.skylock3d?.setCameraFov({self._last_fov});"
        self._web_view.page().runJavaScript(js)

    def update_state(self, state: dict[str, Any]) -> None:
        """Update 3D visualization state from authoritative Python dictionary."""
        if "pan" in state:
            self._last_pan = float(state["pan"])
        if "tilt" in state:
            self._last_tilt = float(state["tilt"])
        if "fov" in state:
            self._last_fov = float(state["fov"])

        state_json = json.dumps(state)
        js = f"window.skylock3d?.updateState({state_json});"
        self._web_view.page().runJavaScript(js)

    def set_paused(self, paused: bool) -> None:
        """Pause or resume the 3D orbit simulation."""
        new_val = bool(paused)
        if self._is_paused == new_val:
            return
        self._is_paused = new_val
        if hasattr(self, "btn_pause"):
            self.btn_pause.blockSignals(True)
            self.btn_pause.setChecked(self._is_paused)
            self.btn_pause.setText("▶ Resume" if self._is_paused else "⏸ Pause")
            self.btn_pause.blockSignals(False)
        js_val = "true" if self._is_paused else "false"
        js = f"window.skylock3d?.setPaused({js_val});"
        self._web_view.page().runJavaScript(js)
        self.pause_toggled.emit(self._is_paused)

    def _on_pause_toggled(self, checked: bool) -> None:
        """Handle toolbar pause button toggle."""
        self.set_paused(checked)

    def set_simulation_speed(self, speed: float) -> None:
        """Set simulation speed multiplier."""
        js = f"window.skylock3d?.setSimulationSpeed({float(speed)});"
        self._web_view.page().runJavaScript(js)

    def reset_view(self) -> None:
        """Reset the user orbit camera to default Earth view."""
        self._web_view.page().runJavaScript("window.skylock3d?.resetView();")

    def reset_camera(self) -> None:
        """Reset tracking gimbal camera to neutral pose (pan=0°, tilt=0°, fov=20°)."""
        self._last_pan = 0.0
        self._last_tilt = 0.0
        self._last_fov = 20.0
        self._web_view.page().runJavaScript("window.skylock3d?.resetCamera();")

    def focus_satellite(self, sat_id: str) -> None:
        """Focus scene viewing camera on satellite ('s1', 's2', or 'earth')."""
        sat_str = json.dumps(sat_id)
        self._web_view.page().runJavaScript(f"window.skylock3d?.focusSatellite({sat_str});")

    def focus_s1(self) -> None:
        """Focus scene viewing camera on Observer Satellite S-1."""
        self.focus_satellite("s1")

    def focus_s2(self) -> None:
        """Focus scene viewing camera on Target Satellite S-2."""
        self.focus_satellite("s2")

    def set_show_orbit_lines(self, show: bool) -> None:
        """Toggle visibility of orbital path lines in 3D space."""
        val = "true" if show else "false"
        self._web_view.page().runJavaScript(f"window.skylock3d?.setShowOrbitLines({val});")

    def set_show_camera_fov(self, show: bool) -> None:
        """Toggle visibility of 3D camera FOV frustum."""
        val = "true" if show else "false"
        self._web_view.page().runJavaScript(f"window.skylock3d?.setShowCameraFov({val});")

    def set_show_optical_axis(self, show: bool) -> None:
        """Toggle visibility of optical axis / boresight ray."""
        val = "true" if show else "false"
        self._web_view.page().runJavaScript(f"window.skylock3d?.setShowOpticalAxis({val});")

    def set_show_tracking_beam(self, show: bool) -> None:
        """Toggle visibility of optical tracking beam."""
        val = "true" if show else "false"
        self._web_view.page().runJavaScript(f"window.skylock3d?.setShowTrackingBeam({val});")

    def set_satellite_orbit(
        self, sat_id: str, radius: float, inc_deg: float, speed: float, phase_deg: float
    ) -> None:
        """Update satellite orbital parameters in 3D simulation."""
        sat_str = json.dumps(sat_id)
        js = (
            f"window.skylock3d?.setSatelliteOrbit({sat_str}, {float(radius)}, "
            f"{float(inc_deg)}, {float(speed)}, {float(phase_deg)});"
        )
        self._web_view.page().runJavaScript(js)

    def query_state(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Query current state from the JavaScript 3D layer asynchronously."""

        def _handle_result(res: Any) -> None:
            if isinstance(res, dict):
                self.state_updated.emit(res)
                callback(res)

        self._web_view.page().runJavaScript("window.skylock3d?.getState();", _handle_result)

    def _query_fps(self) -> None:
        """Poll current rendering FPS from WebGL simulation."""
        if self._is_ready:
            self._web_view.page().runJavaScript(
                "window.skylock3d?.getFps ? window.skylock3d.getFps() : 60;",
                self._update_fps_display,
            )

    def _update_fps_display(self, fps_val: Any) -> None:
        """Update FPS numeric badge and color dynamically."""
        if fps_val is None:
            return
        try:
            fps = int(round(float(fps_val)))
        except (ValueError, TypeError):
            return

        self.lbl_fps.setText(str(fps))
        if fps >= 55:
            color = theme.FPS_GREEN.name()
        elif fps >= 40:
            color = theme.FPS_LIME.name()
        elif fps >= 25:
            color = theme.FPS_AMBER.name()
        else:
            color = theme.FPS_RED.name()

        self.lbl_fps.setStyleSheet(
            f"QLabel {{ color: {color}; font-weight: bold; font-family: 'Consolas', 'Courier New', monospace; "
            f"font-size: 13px; padding: 2px 8px; background: {theme.DARK_BG.name()}; border: 1px solid {theme.BORDER_NORMAL.name()}; "
            f"border-radius: 4px; min-width: 24px; }}"
        )

    def closeEvent(self, event: Any) -> None:
        """Clean up embedded HTTP server when widget is closed."""
        self.cleanup()
        super().closeEvent(event)

    def cleanup(self) -> None:
        """Stop server and release resources."""
        if hasattr(self, "_fps_timer") and self._fps_timer is not None:
            self._fps_timer.stop()
        if hasattr(self, "_web_view") and self._web_view is not None:
            try:
                self._web_view.stop()
            except Exception:
                pass
        if self._server is not None:
            self._server.stop()
