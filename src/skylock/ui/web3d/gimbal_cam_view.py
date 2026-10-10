"""PySide6 3D Gimbal POV View widget embedding first-person Three.js simulation.

Also contains ``GimbalOverlayWidget``, a transparent Qt painter layer that draws
real-time tracking symbology (boresight, detections, gate, estimate) on top of
the 4:3 letterboxed WebGL canvas, driven by live ``FrameView`` data from the session pipeline.
"""

from __future__ import annotations

import base64
import logging
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QUrl, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from skylock.config.models import CameraConfig
from skylock.core.geometry import angle_offset_to_pixel, pixel_to_angle_offset
from skylock.core.orbital_world import pan_tilt_to_world_aim
from skylock.core.sim_clock import get_shared_clock
from skylock.ui import theme
from skylock.ui.web3d.server import Embedded3DServer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transparent Qt overlay drawing tracking symbology
# ---------------------------------------------------------------------------

class GimbalOverlayWidget(QWidget):
    """Transparent overlay that paints tracking symbology on the gimbal POV canvas.

    The widget covers exactly the 4:3 letterboxed render rectangle of the WebGL view
    and is kept invisible to mouse events so the underlying canvas still receives them.

    Coordinate conversion pipeline
    --------------------------------
    1. Grayscale pipeline outputs detections/estimate in sensor-pixel space (live camera config).
    2. ``_pixel_to_angle`` maps them to tangent-plane angular offsets from boresight.
    3. ``_angle_to_pixel`` remaps into *this overlay's* pixel space using the live 3D render
       camera FOV (4:3 aspect) — so the angular position always matches the 3D render exactly.
    """

    def __init__(
        self,
        parent: QWidget,
        camera_config: CameraConfig | None = None,
        render_fov_h_deg: float = 16.0,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._camera_cfg = camera_config if camera_config is not None else CameraConfig()
        self._render_fov_h = float(render_fov_h_deg)
        self._render_fov_v = 0.75 * self._render_fov_h

        # Payload — set via update_data()
        self._detections: tuple[tuple[float, float, float, float], ...] = ()
        self._estimate: tuple[float, float] | None = None
        self._gate_px: float = 0.0
        self._is_predicting: bool = False
        self._has_data: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_camera_config(self, cfg: CameraConfig) -> None:
        """Update live camera sensor hardware configuration."""
        self._camera_cfg = cfg
        self.update()

    def set_fov(self, fov_h_deg: float, fov_v_deg: float | None = None) -> None:
        """Update active 3D camera render FOV."""
        self._render_fov_h = float(fov_h_deg)
        self._render_fov_v = (
            float(fov_v_deg) if fov_v_deg is not None else 0.75 * self._render_fov_h
        )
        self.update()

    def update_data(
        self,
        detections: tuple[tuple[float, float, float, float], ...],
        estimate: tuple[float, float] | None,
        gate_px: float,
        is_predicting: bool,
    ) -> None:
        """Push new tracking data and schedule a repaint."""
        self._detections = detections
        self._estimate = estimate
        self._gate_px = gate_px
        self._is_predicting = is_predicting
        self._has_data = True
        self.update()

    def clear_data(self) -> None:
        """Clear all overlay data (e.g. on session reset)."""
        self._detections = ()
        self._estimate = None
        self._gate_px = 0.0
        self._has_data = False
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _sensor_to_overlay(self, spx: float, spy: float) -> tuple[float, float]:
        """Map sensor pixel coordinate to overlay pixel space matching 3D render FOV."""
        dpan, dtilt = pixel_to_angle_offset(spx, spy, self._camera_cfg)
        overlay_cam = CameraConfig(
            width=max(1, self.width()),
            height=max(1, self.height()),
            fov_h_deg=self._render_fov_h,
            fov_v_deg=self._render_fov_v,
        )
        return angle_offset_to_pixel(dpan, dtilt, overlay_cam)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, _event: Any) -> None:  # noqa: ANN401
        """Paint tracking symbology on top of the 4:3 letterboxed canvas."""
        w_w = self.width()
        w_h = self.height()
        if w_w <= 0 or w_h <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # --- 1. Boresight crosshair (always drawn) ---
        bx, by = w_w / 2.0, w_h / 2.0
        arm = 14.0
        gap = 4.0
        painter.setPen(QPen(theme.OVERLAY_BORESIGHT, 1.5, Qt.PenStyle.SolidLine))
        painter.drawLine(QPointF(bx - arm, by), QPointF(bx - gap, by))
        painter.drawLine(QPointF(bx + gap, by), QPointF(bx + arm, by))
        painter.drawLine(QPointF(bx, by - arm), QPointF(bx, by - gap))
        painter.drawLine(QPointF(bx, by + gap), QPointF(bx, by + arm))

        if not self._has_data:
            painter.end()
            return

        # --- 2. Detections (yellow centroid rings) ---
        painter.setPen(QPen(theme.OVERLAY_DETECTION, 1.5, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for scx, scy, _dw, _dh in self._detections:
            ox, oy = self._sensor_to_overlay(scx, scy)
            r = 8.0
            painter.drawEllipse(QPointF(ox, oy), r, r)

        # --- 3. Gate box & estimate cross ---
        if self._estimate is not None:
            ex, ey = self._sensor_to_overlay(self._estimate[0], self._estimate[1])

            # Gate box
            if self._gate_px > 0:
                gate_deg_h = self._gate_px * (self._camera_cfg.fov_h_deg / float(self._camera_cfg.width))
                gate_deg_v = self._gate_px * (self._camera_cfg.fov_v_deg / float(self._camera_cfg.height))
                g_w = gate_deg_h * (float(w_w) / self._render_fov_h) * 2.0
                g_h = gate_deg_v * (float(w_h) / self._render_fov_v) * 2.0
                gate_rect = QRectF(ex - g_w / 2.0, ey - g_h / 2.0, g_w, g_h)
                painter.setPen(QPen(theme.OVERLAY_GATE, 1.0, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(gate_rect)

            # Estimate cross (+)
            pen_style = Qt.PenStyle.DashLine if self._is_predicting else Qt.PenStyle.SolidLine
            painter.setPen(QPen(theme.OVERLAY_ESTIMATE, 2.0, pen_style))
            c_arm = 10.0
            painter.drawLine(QPointF(ex - c_arm, ey), QPointF(ex + c_arm, ey))
            painter.drawLine(QPointF(ex, ey - c_arm), QPointF(ex, ey + c_arm))

        painter.end()


# ---------------------------------------------------------------------------
# Main Gimbal POV view widget
# ---------------------------------------------------------------------------

class GimbalCamView(QWidget):
    """First-person perspective 3D gimbal view mounted on a satellite looking toward the target."""

    scene_ready = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        server: Embedded3DServer | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)

        self._server = server or Embedded3DServer.get_shared_server()
        self._camera_cfg = camera_config if camera_config is not None else CameraConfig()

        self._is_ready = False
        self._mount_sat = "s1"
        self._focus_mode = "TARGET"
        self._last_pan = 0.0
        self._last_tilt = 0.0
        self._last_fov_h = 16.0
        self._last_fov_v = 12.0
        self._is_paused = False

        self._last_forward = (1.0, 0.0, 0.0)
        self._last_up = (0.0, 1.0, 0.0)
        self._last_push_time = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._web_view = QWebEngineView(self)
        layout.addWidget(self._web_view, stretch=1)

        # Configure WebEngine settings for hardware acceleration
        settings = self._web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        # Transparent tracking-symbology overlay (covers exact letterboxed 4:3 render rect)
        self._overlay = GimbalOverlayWidget(
            self,
            camera_config=self._camera_cfg,
            render_fov_h_deg=self._last_fov_h,
        )
        self._overlay.raise_()

        self._web_view.loadFinished.connect(self._on_load_finished)

        # Load Gimbal POV 3D scene from embedded local server
        url = self._server.get_url("gimbal/index.html")
        self._web_view.load(QUrl(url))

    @property
    def is_ready(self) -> bool:
        """Whether the WebGL 3D scene is loaded and ready."""
        return self._is_ready

    @property
    def mount(self) -> str:
        """Currently mounted satellite ID ('s1' or 's2')."""
        return self._mount_sat

    @property
    def mount_satellite(self) -> str:
        """Currently mounted satellite ID ('s1' or 's2')."""
        return self._mount_sat

    @property
    def focus_mode(self) -> str:
        """Current camera aim focus mode ('TARGET' or 'EARTH')."""
        return self._focus_mode

    @property
    def is_paused(self) -> bool:
        """Whether the 3D orbital clock is paused."""
        return self._is_paused

    @property
    def current_pan(self) -> float:
        """Current gimbal pan in degrees."""
        return self._last_pan

    @property
    def last_pan(self) -> float:
        """Last commanded gimbal pan in degrees."""
        return self._last_pan

    @property
    def current_tilt(self) -> float:
        """Current gimbal tilt in degrees."""
        return self._last_tilt

    @property
    def last_tilt(self) -> float:
        """Last commanded gimbal tilt in degrees."""
        return self._last_tilt

    @property
    def current_fov(self) -> float:
        """Current camera horizontal FOV in degrees."""
        return self._last_fov_h

    @property
    def last_fov(self) -> float:
        """Last commanded camera horizontal FOV in degrees."""
        return self._last_fov_h

    @property
    def current_fov_h(self) -> float:
        """Current camera horizontal FOV in degrees."""
        return self._last_fov_h

    @property
    def current_fov_v(self) -> float:
        """Current camera vertical FOV in degrees."""
        return self._last_fov_v

    def _on_load_finished(self, success: bool) -> None:
        """Invoked when web view finishes loading."""
        if success:
            self._is_ready = True
            logger.info("GimbalCamView WebGL scene loaded successfully")
            self.scene_ready.emit()
            # Push initial values
            self._web_view.page().runJavaScript("window.gimbalcam?.setExternalClock(true);")
            self.set_mount(self._mount_sat)
            self.set_fov(self._last_fov_h)
            self.set_aim(self._last_forward, self._last_up, force=True)
            self.set_paused(self._is_paused)
            # Force-remove loading overlay from Python side as safety measure
            self._web_view.page().runJavaScript(
                "var ov=document.getElementById('loading-overlay');"
                "if(ov){ov.style.opacity='0';setTimeout(function(){ov.remove();},400);}"
            )
        else:
            logger.warning("GimbalCamView failed to load WebGL scene")
            # Still mark ready so the app doesn't hang
            self._is_ready = True
            self.scene_ready.emit()

    def resizeEvent(self, event: Any) -> None:  # noqa: ANN401
        """Keep tracking overlay exactly covering the letterboxed 4:3 render rect."""
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        target_aspect = 4.0 / 3.0
        if w <= 0 or h <= 0:
            return

        if w / h > target_aspect:
            vh = h
            vw = int(round(h * target_aspect))
            vx = int(round((w - vw) / 2.0))
            vy = 0
        else:
            vw = w
            vh = int(round(w / target_aspect))
            vx = 0
            vy = int(round((h - vh) / 2.0))

        self._overlay.setGeometry(vx, vy, vw, vh)
        self._overlay.raise_()

    def set_aim(
        self,
        forward: tuple[float, float, float] | np.ndarray,
        up: tuple[float, float, float] | np.ndarray,
        force: bool = False,
        pan_deg: float | None = None,
        tilt_deg: float | None = None,
    ) -> None:
        """Command 3D gimbal camera forward and up unit vectors in world space.

        Throttled to <= 60 Hz to avoid flooding the WebEngine bridge.
        """
        fx = float(forward[0])
        fy = float(forward[1])
        fz = float(forward[2])
        ux = float(up[0])
        uy = float(up[1])
        uz = float(up[2])

        self._last_forward = (fx, fy, fz)
        self._last_up = (ux, uy, uz)
        if pan_deg is not None:
            self._last_pan = float(pan_deg)
        if tilt_deg is not None:
            self._last_tilt = float(tilt_deg)

        now = time.monotonic()
        if not force and (now - self._last_push_time < 1.0 / 60.0):
            return
        self._last_push_time = now

        js = f"window.gimbalcam?.setAim({fx:.6f}, {fy:.6f}, {fz:.6f}, {ux:.6f}, {uy:.6f}, {uz:.6f});"
        self._web_view.page().runJavaScript(js)

    def set_pose(self, pan_deg: float, tilt_deg: float) -> None:
        """Command gimbal pan/tilt angles (converts via observer frame math into world aim)."""
        self._last_pan = float(pan_deg)
        self._last_tilt = float(tilt_deg)
        t_now = get_shared_clock().now()
        f_vec, u_vec = pan_tilt_to_world_aim(self._mount_sat, self._last_pan, self._last_tilt, t_now)
        self.set_aim(f_vec, u_vec)

    def set_fov(self, fov_h_deg: float) -> None:
        """Update 3D gimbal camera horizontal FOV in degrees (vertical is 0.75x at 4:3)."""
        self._last_fov_h = float(fov_h_deg)
        self._last_fov_v = 0.75 * self._last_fov_h
        self._overlay.set_fov(self._last_fov_h, self._last_fov_v)
        js = f"window.gimbalcam?.setFov({self._last_fov_h});"
        self._web_view.page().runJavaScript(js)

    def set_camera_config(self, cfg: CameraConfig) -> None:
        """Update live camera sensor hardware configuration for the overlay."""
        self._camera_cfg = cfg
        self._overlay.set_camera_config(cfg)

    def set_mount(self, sat_id: str) -> None:
        """Set which satellite hosts the first-person gimbal camera ('s1' or 's2')."""
        clean_id = str(sat_id).lower().replace("-", "")
        self._mount_sat = "s2" if clean_id in ("s2", "sat2", "2") else "s1"
        js = f"window.gimbalcam?.setMount('{self._mount_sat}');"
        self._web_view.page().runJavaScript(js)

    def set_focus_mode(self, mode: str) -> None:
        """Command the camera aim focus mode ('TARGET', 'EARTH', or 'MANUAL')."""
        m = str(mode).upper()
        self._focus_mode = "EARTH" if m in ("EARTH", "EARTH_BORESIGHT") else "TARGET"

    def set_paused(self, paused: bool) -> None:
        """Pause or resume the 3D orbital clock."""
        self._is_paused = bool(paused)
        js = f"window.gimbalcam?.setPaused({str(self._is_paused).lower()});"
        self._web_view.page().runJavaScript(js)

    def set_simulation_speed(self, speed: float) -> None:
        """Set simulation speed multiplier."""
        js = f"window.gimbalcam?.setSimulationSpeed({float(speed)});"
        self._web_view.page().runJavaScript(js)

    def set_satellite_orbit(
        self,
        sat_id: str,
        radius: float,
        inc_deg: float,
        speed: float,
        phase_deg: float,
    ) -> None:
        """Configure orbit parameters for a satellite."""
        js = (
            f"window.gimbalcam?.setSatelliteOrbit('{sat_id}', {float(radius)}, "
            f"{float(inc_deg)}, {float(speed)}, {float(phase_deg)});"
        )
        self._web_view.page().runJavaScript(js)

    def set_time(self, time_sec: float) -> None:
        """Synchronize the simulation time directly (seconds)."""
        js = f"window.gimbalcam?.setTime({float(time_sec)});"
        self._web_view.page().runJavaScript(js)

    def reset_pose(self) -> None:
        """Reset gimbal camera to neutral pose (pan=0°, tilt=0°, default FOV=16°x12°)."""
        self._last_pan = 0.0
        self._last_tilt = 0.0
        self._last_fov_h = 16.0
        self._last_fov_v = 12.0
        self.set_fov(16.0)
        self.set_pose(0.0, 0.0)
        self._overlay.clear_data()

    # ------------------------------------------------------------------
    # Overlay / tracking-symbology API
    # ------------------------------------------------------------------
    def update_overlay(
        self,
        detections: tuple[tuple[float, float, float, float], ...],
        estimate: tuple[float, float] | None,
        gate_px: float,
        is_predicting: bool,
    ) -> None:
        """Push real pipeline data to the tracking overlay.

        Args:
            detections: Sequence of (cx, cy, w, h) in sensor pixel space.
            estimate:   Kalman estimate (px, py) in sensor pixel space, or ``None``.
            gate_px:    Gate half-width in sensor pixels.
            is_predicting: True when state is LOST/REACQUIRE (dashed estimate cross).
        """
        self._overlay.update_data(detections, estimate, gate_px, is_predicting)

    @property
    def overlay(self) -> GimbalOverlayWidget:
        """The transparent tracking-symbology overlay widget."""
        return self._overlay

    def get_state(
        self,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        """Query internal state of the 3D gimbal scene."""
        if callback is not None:
            js = "window.gimbalcam ? window.gimbalcam.getState() : null;"
            self._web_view.page().runJavaScript(js, callback)
            return None
        return {
            "mount": self._mount_sat,
            "focus_mode": self._focus_mode,
            "pan": self._last_pan,
            "tilt": self._last_tilt,
            "fov": self._last_fov_h,
            "fov_h": self._last_fov_h,
            "fov_v": self._last_fov_v,
            "is_paused": self._is_paused,
        }

    def capture_frame(self, callback: Callable[[QImage | None], None]) -> None:
        """Capture current 3D WebGL render with tracking overlay composited on top."""

        def on_js_done(data_url: Any) -> None:
            if not data_url or not isinstance(data_url, str) or not data_url.startswith("data:image"):
                pix = self.grab()
                callback(pix.toImage())
                return
            try:
                base64_data = data_url.split(",", 1)[1]
                img_bytes = base64.b64decode(base64_data)
                img = QImage()
                img.loadFromData(img_bytes)
                # Composite the Qt tracking overlay on top
                overlay_pix = self._overlay.grab()
                painter = QPainter(img)
                painter.drawPixmap(self._overlay.x(), self._overlay.y(), overlay_pix)
                painter.end()
                callback(img)
            except Exception as e:
                logger.error("Error compositing gimbal frame: %s", e)
                callback(self.grab().toImage())

        self._web_view.page().runJavaScript(
            "window.gimbalcam ? window.gimbalcam.captureDataUrl() : null;",
            on_js_done,
        )

    def cleanup(self) -> None:
        """Clean up view resources."""
        self._web_view.stop()


__all__ = ("GimbalCamView", "GimbalOverlayWidget")
