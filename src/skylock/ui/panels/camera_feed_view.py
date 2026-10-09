"""Composite panel for the Camera Sensor Feed tab.

Combines Section 1 (Observer-switchable sensor imagery with Earth/AUTO/MANUAL focus controls,
and CAMERA (3D) | SENSOR (mono) view toggle) and Section 2 (Real-time communications, lock
transitions, LOS occlusion, and handshake event log driven by the shared simulation clock).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from skylock.core.enums import ControlMode, TrackState
from skylock.core.los import check_line_of_sight, get_satellite_position
from skylock.core.orbital_world import pan_tilt_to_world_aim
from skylock.core.sim_clock import get_shared_clock
from skylock.ui import theme
from skylock.ui.web3d.gimbal_cam_view import GimbalCamView
from skylock.ui.widgets.camera_view import CameraView
from skylock.ui.worker import FrameView, SessionWorker


class CameraFeedView(QWidget):
    """Integrated Camera Sensor Feed tab containing sensor view and live status log."""

    frame_painted = Signal()  # Forwarded from inner views for back-pressure
    pause_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: SessionWorker | None = None
        self._steering_filter: Any = None
        self._selected_sat: str = "s1"  # "s1" or "s2"
        self._is_paused: bool = False
        self._active_view_mode: str = "CAMERA"  # "CAMERA" or "SENSOR"

        # State tracking for events
        self._s1_locked = False
        self._s2_locked = False
        self._handshake_active = False
        self._prev_los_blocked: bool | None = None
        self._log_history: deque[str] = deque(maxlen=100)

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)

        # --------------------------------------------------------------------
        # SECTION 1: Camera Feed + Controls
        # --------------------------------------------------------------------
        section1_widget = QWidget(splitter)
        sec1_layout = QVBoxLayout(section1_widget)
        sec1_layout.setContentsMargins(2, 2, 2, 2)
        sec1_layout.setSpacing(4)

        # Control Bar
        control_bar = QFrame(section1_widget)
        control_bar.setStyleSheet(
            f"QFrame {{ background-color: {theme.BASE_BG.name()}; "
            f"border: 1px solid {theme.BORDER_NORMAL.name()}; border-radius: 4px; padding: 2px; }}"
        )
        bar_layout = QHBoxLayout(control_bar)
        bar_layout.setContentsMargins(6, 4, 6, 4)
        bar_layout.setSpacing(8)

        # Observer selector
        lbl_obs = QLabel("OBSERVER:", control_bar)
        lbl_obs.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY.name()}; font-weight: 700; font-size: 11px; border: none;"
        )
        bar_layout.addWidget(lbl_obs)

        self.combo_observer = QComboBox(control_bar)
        self.combo_observer.addItem("View S-2 from S-1", "s1")
        self.combo_observer.addItem("View S-1 from S-2", "s2")
        self.combo_observer.setToolTip("Select which satellite camera feeds this view")
        self.combo_observer.setStyleSheet(
            f"QComboBox {{ background-color: {theme.ALT_BASE_BG.name()}; color: {theme.TEXT_PRIMARY.name()}; "
            f"border: 1px solid {theme.BORDER_NORMAL.name()}; border-radius: 4px; padding: 4px 8px; "
            f"font-weight: 600; font-size: 11px; }}"
            f"QComboBox:hover {{ border-color: {theme.HIGHLIGHT_BG.name()}; }}"
            f"QComboBox QAbstractItemView {{ background-color: {theme.BASE_BG.name()}; "
            f"color: {theme.TEXT_PRIMARY.name()}; selection-background-color: {theme.HIGHLIGHT_BG.name()}; }}"
        )
        self.combo_observer.currentIndexChanged.connect(self._on_observer_changed)
        bar_layout.addWidget(self.combo_observer)

        # Divider
        div1 = QFrame(control_bar)
        div1.setFrameShape(QFrame.Shape.VLine)
        div1.setStyleSheet(f"color: {theme.BORDER_NORMAL.name()};")
        bar_layout.addWidget(div1)

        # Focus mode controls
        lbl_mode = QLabel("FOCUS:", control_bar)
        lbl_mode.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY.name()}; font-weight: 700; font-size: 11px; border: none;"
        )
        bar_layout.addWidget(lbl_mode)

        self.mode_group = QButtonGroup(control_bar)
        self.mode_group.setExclusive(True)

        btn_style = (
            f"QPushButton {{ background-color: {theme.ALT_BASE_BG.name()}; color: {theme.TEXT_SECONDARY.name()}; "
            f"border: 1px solid {theme.BORDER_NORMAL.name()}; border-radius: 4px; padding: 4px 10px; "
            f"font-weight: 600; font-size: 11px; }}"
            f"QPushButton:hover {{ background-color: {theme.BTN_HOVER_BG.name()}; color: {theme.TEXT_PRIMARY.name()}; }}"
            f"QPushButton:checked {{ background-color: {theme.HIGHLIGHT_BG.name()}; color: {theme.COLOR_WHITE.name()}; "
            f"border-color: {theme.BTN_BORDER_ACTIVE.name()}; }}"
        )

        self.btn_focus_earth = QPushButton("Focus Earth", control_bar)
        self.btn_focus_earth.setCheckable(True)
        self.btn_focus_earth.setToolTip("Point gimbal continuously at Earth center")
        self.btn_focus_earth.setStyleSheet(btn_style)
        self.mode_group.addButton(self.btn_focus_earth, 0)
        bar_layout.addWidget(self.btn_focus_earth)

        self.btn_focus_target = QPushButton("Focus S-2", control_bar)
        self.btn_focus_target.setCheckable(True)
        self.btn_focus_target.setChecked(True)  # Default session start mode: AUTO
        self.btn_focus_target.setToolTip("Active pipeline tracking on opposing satellite")
        self.btn_focus_target.setStyleSheet(btn_style)
        self.mode_group.addButton(self.btn_focus_target, 1)
        bar_layout.addWidget(self.btn_focus_target)

        self.btn_manual = QPushButton("Manual", control_bar)
        self.btn_manual.setCheckable(True)
        self.btn_manual.setToolTip("Manual slew control via Arrow keys / WASD")
        self.btn_manual.setStyleSheet(btn_style)
        self.mode_group.addButton(self.btn_manual, 2)
        bar_layout.addWidget(self.btn_manual)

        self.btn_focus_earth.clicked.connect(lambda: self._set_pointing_mode("EARTH"))
        self.btn_focus_target.clicked.connect(lambda: self._set_pointing_mode("AUTO"))
        self.btn_manual.clicked.connect(lambda: self._set_pointing_mode("MANUAL"))

        # Divider
        div2 = QFrame(control_bar)
        div2.setFrameShape(QFrame.Shape.VLine)
        div2.setStyleSheet(f"color: {theme.BORDER_NORMAL.name()};")
        bar_layout.addWidget(div2)

        # View Toggle: CAMERA (3D) | SENSOR (mono)
        lbl_view = QLabel("VIEW:", control_bar)
        lbl_view.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY.name()}; font-weight: 700; font-size: 11px; border: none;"
        )
        bar_layout.addWidget(lbl_view)

        self.view_toggle_group = QButtonGroup(control_bar)
        self.view_toggle_group.setExclusive(True)

        self.btn_view_camera = QPushButton("CAMERA (3D)", control_bar)
        self.btn_view_camera.setCheckable(True)
        self.btn_view_camera.setChecked(True)
        self.btn_view_camera.setToolTip("First-person 3D simulation with live tracking symbology overlay")
        self.btn_view_camera.setStyleSheet(btn_style)
        self.view_toggle_group.addButton(self.btn_view_camera, 0)
        bar_layout.addWidget(self.btn_view_camera)

        self.btn_view_sensor = QPushButton("SENSOR (mono)", control_bar)
        self.btn_view_sensor.setCheckable(True)
        self.btn_view_sensor.setToolTip("Raw monochrome detector camera frame from the tracking pipeline")
        self.btn_view_sensor.setStyleSheet(btn_style)
        self.view_toggle_group.addButton(self.btn_view_sensor, 1)
        bar_layout.addWidget(self.btn_view_sensor)

        self.btn_view_camera.clicked.connect(lambda: self._set_active_view("CAMERA"))
        self.btn_view_sensor.clicked.connect(lambda: self._set_active_view("SENSOR"))

        bar_layout.addStretch(1)

        # Status badge for selected session
        self.lbl_selected_status = QLabel("SEARCH", control_bar)
        self.lbl_selected_status.setStyleSheet(
            f"background-color: {theme.STATE_SEARCH_BG.name()}; color: {theme.STATE_SEARCH_TEXT.name()}; "
            f"border: 1px solid {theme.STATE_SEARCH_PRIMARY.name()}; border-radius: 3px; "
            f"padding: 2px 6px; font-weight: 700; font-size: 10px;"
        )
        bar_layout.addWidget(self.lbl_selected_status)

        sec1_layout.addWidget(control_bar)

        # Stacked display: CAMERA (3D) on page 0, SENSOR (mono) on page 1
        self.view_stack = QStackedWidget(section1_widget)

        self.gimbal_3d_view = GimbalCamView(self.view_stack)
        self.view_stack.addWidget(self.gimbal_3d_view)

        self._camera_view = CameraView(self.view_stack)
        self._camera_view.frame_painted.connect(self.frame_painted.emit)
        self.view_stack.addWidget(self._camera_view)

        self.view_stack.setCurrentIndex(0)
        sec1_layout.addWidget(self.view_stack, stretch=1)

        # --------------------------------------------------------------------
        # SECTION 2: Live Communications & Status Text Feed
        # --------------------------------------------------------------------
        section2_widget = QWidget(splitter)
        sec2_layout = QVBoxLayout(section2_widget)
        sec2_layout.setContentsMargins(2, 2, 2, 2)
        sec2_layout.setSpacing(4)

        # Header
        header_bar = QFrame(section2_widget)
        header_bar.setStyleSheet(
            f"QFrame {{ background-color: {theme.BASE_BG.name()}; "
            f"border: 1px solid {theme.BORDER_NORMAL.name()}; border-radius: 4px; padding: 2px; }}"
        )
        hdr_layout = QHBoxLayout(header_bar)
        hdr_layout.setContentsMargins(6, 4, 6, 4)
        hdr_layout.setSpacing(6)

        lbl_hdr = QLabel("LIVE COMMS & STATUS", header_bar)
        lbl_hdr.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY.name()}; font-weight: 700; font-size: 11px; border: none;"
        )
        hdr_layout.addWidget(lbl_hdr)
        hdr_layout.addStretch(1)

        # Link indicator
        self.lbl_link_indicator = QLabel("LINK CLEAR", header_bar)
        self.lbl_link_indicator.setStyleSheet(
            f"background-color: {theme.STATE_TRACK_BG.name()}; color: {theme.STATE_TRACK_TEXT.name()}; "
            f"border: 1px solid {theme.STATE_TRACK_PRIMARY.name()}; "
            f"border-radius: 3px; padding: 2px 6px; font-weight: 700; font-size: 10px;"
        )
        hdr_layout.addWidget(self.lbl_link_indicator)

        # Handshake indicator
        self.lbl_handshake_indicator = QLabel("HANDSHAKE: OFF", header_bar)
        self.lbl_handshake_indicator.setStyleSheet(
            f"background-color: {theme.ALT_BASE_BG.name()}; color: {theme.TEXT_SECONDARY.name()}; "
            f"border: 1px solid {theme.BORDER_NORMAL.name()}; border-radius: 3px; "
            f"padding: 2px 6px; font-weight: 700; font-size: 10px;"
        )
        hdr_layout.addWidget(self.lbl_handshake_indicator)

        sec2_layout.addWidget(header_bar)

        # Monospace Text Feed
        self.log_text = QTextEdit(section2_widget)
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_text.setFont(font)
        self.log_text.setStyleSheet(
            f"QTextEdit {{ background-color: {theme.DARK_BG.name()}; color: {theme.TEXT_PRIMARY.name()}; "
            f"border: 1px solid {theme.BORDER_NORMAL.name()}; border-radius: 4px; padding: 6px; "
            f"line-height: 1.4; }}"
        )
        sec2_layout.addWidget(self.log_text, stretch=1)

        # Splitter proportions: 70% left, 30% right
        splitter.addWidget(section1_widget)
        splitter.addWidget(section2_widget)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([750, 320])

        # Initial log message
        t_init = get_shared_clock().now()
        self.append_log(f"{t_init:.1f}s — SYSTEM READY — Dual-Direction Tracking Initialized", "info")

    # ------------------------------------------------------------------------
    # Public API & Proxies (for 100% test & main_window compatibility)
    # ------------------------------------------------------------------------
    @property
    def camera_view(self) -> CameraView:
        """Underlying canvas CameraView instance."""
        return self._camera_view

    @property
    def gimbal_cam(self) -> GimbalCamView:
        """Gimbal 3D POV camera view widget."""
        return self.gimbal_3d_view

    def set_paused(self, paused: bool) -> None:
        """Pause or resume the 3D gimbal camera POV."""
        new_val = bool(paused)
        if self._is_paused == new_val:
            return
        self._is_paused = new_val
        self.gimbal_3d_view.set_paused(self._is_paused)
        self.pause_toggled.emit(self._is_paused)

    def set_fov(self, fov_deg: float) -> None:
        """Set horizontal FOV for the 3D gimbal camera POV (vertical derived at 0.75x)."""
        self.gimbal_3d_view.set_fov(fov_deg)

    @property
    def selected_satellite(self) -> str:
        """Currently selected observer satellite ('s1' or 's2')."""
        return self._selected_sat

    def set_worker(
        self,
        worker: SessionWorker,
        steering_filter: Any = None,
    ) -> None:
        """Bind SessionWorker and ManualSteeringFilter to this feed panel."""
        self._worker = worker
        self._steering_filter = steering_filter
        worker.dual_frame_ready.connect(self.on_dual_frame_ready)
        worker.session_rebuilt.connect(self._on_session_rebuilt)

        sess = worker.get_session(self._selected_sat)
        if sess is not None and hasattr(sess, "config"):
            self.gimbal_3d_view.set_camera_config(sess.config.camera)

    @property
    def _frame_view(self) -> FrameView | None:
        """Proxy to inner camera view FrameView."""
        return self._camera_view._frame_view

    @_frame_view.setter
    def _frame_view(self, val: FrameView | None) -> None:
        self._camera_view._frame_view = val

    @property
    def show_ground_truth(self) -> bool:
        """Proxy to inner camera view show_ground_truth."""
        return self._camera_view.show_ground_truth

    @show_ground_truth.setter
    def show_ground_truth(self, value: bool) -> None:
        self._camera_view.show_ground_truth = value

    @property
    def show_legend(self) -> bool:
        """Proxy to inner camera view show_legend."""
        return self._camera_view.show_legend

    @show_legend.setter
    def show_legend(self, value: bool) -> None:
        self._camera_view.show_legend = value

    def clear(self) -> None:
        """Clear canvas and reset state tracking."""
        self._camera_view.clear()
        self.gimbal_3d_view.reset_pose()
        self._s1_locked = False
        self._s2_locked = False
        self._handshake_active = False
        self._prev_los_blocked = None
        self._update_handshake_badge(False)
        t_now = get_shared_clock().now()
        self.append_log(f"{t_now:.1f}s — SYSTEM RESET — Dual tracking reset", "info")

    def get_log_lines(self) -> list[str]:
        """Return list of recent plain-text log lines."""
        return list(self._log_history)

    def clear_log(self) -> None:
        """Clear log history display."""
        self._log_history.clear()
        self.log_text.clear()

    # ------------------------------------------------------------------------
    # Control Actions
    # ------------------------------------------------------------------------
    def _set_active_view(self, mode: str) -> None:
        """Toggle between CAMERA (3D render + overlay) and SENSOR (mono pipeline)."""
        self._active_view_mode = mode
        if mode == "SENSOR":
            self.btn_view_sensor.setChecked(True)
            self.view_stack.setCurrentIndex(1)
            self._camera_view.show()
            self._camera_view.update()
        else:
            self.btn_view_camera.setChecked(True)
            self.view_stack.setCurrentIndex(0)
            self.gimbal_3d_view.show()

    @Slot(int)
    def _on_observer_changed(self, index: int) -> None:
        """Handle observer switch without disturbing the non-displayed session."""
        sat_id = self.combo_observer.itemData(index)
        if not sat_id:
            sat_id = "s1" if index == 0 else "s2"
        self._selected_sat = sat_id

        # Update dynamic label for target focus button
        target_sat = "S-2" if sat_id == "s1" else "S-1"
        self.btn_focus_target.setText(f"Focus {target_sat}")
        self.gimbal_3d_view.set_mount(sat_id)

        # Update worker active satellite
        if self._worker is not None:
            self._worker.set_active_satellite(sat_id)
            sess = self._worker.get_session(sat_id)
            if sess is not None and hasattr(sess, "config"):
                self.gimbal_3d_view.set_camera_config(sess.config.camera)

            # Query newly selected session's actual mode and sync button state
            if sess is not None and hasattr(sess, "controller"):
                current_mode = sess.controller.mode
            else:
                current_mode = (
                    self._worker._mode_str_s1 if sat_id == "s1" else self._worker._mode_str_s2
                )

            mode_str = str(current_mode).upper()
            if mode_str in ("EARTH", "EARTH_BORESIGHT"):
                self.btn_focus_earth.setChecked(True)
                self.gimbal_3d_view.set_focus_mode("EARTH")
                if self._steering_filter is not None:
                    self._steering_filter.set_manual_mode(False)
            elif mode_str == "MANUAL":
                self.btn_manual.setChecked(True)
                self.gimbal_3d_view.set_focus_mode("TARGET")
                if self._steering_filter is not None:
                    self._steering_filter.set_manual_mode(True)
            else:
                self.btn_focus_target.setChecked(True)
                self.gimbal_3d_view.set_focus_mode("TARGET")
                if self._steering_filter is not None:
                    self._steering_filter.set_manual_mode(False)

            # Immediately update canvas with latest frame of newly selected sat
            latest_fv = self._worker.get_latest_frame_view(sat_id)
            if latest_fv is not None:
                self._camera_view.update_frame(latest_fv)
                t_sim = get_shared_clock().now()
                f_vec, u_vec = pan_tilt_to_world_aim(
                    sat_id, latest_fv.pointing_pan_deg, latest_fv.pointing_tilt_deg, t_sim
                )
                self.gimbal_3d_view.set_aim(f_vec, u_vec, force=True)
                self.gimbal_3d_view.set_time(t_sim)

    def _set_pointing_mode(self, mode: str) -> None:
        """Set pointing mode for the currently selected session."""
        self.gimbal_3d_view.set_focus_mode(mode)
        if self._worker is not None:
            self._worker.set_mode(self._selected_sat, mode)

        if self._steering_filter is not None:
            self._steering_filter.set_manual_mode(mode == "MANUAL")

    def sync_mode_from_external(self, mode: str) -> None:
        """Synchronize UI buttons when mode is changed externally (e.g. controls panel)."""
        m = mode.upper()
        self.gimbal_3d_view.set_focus_mode(m)
        if m in ("EARTH", "EARTH_BORESIGHT"):
            self.btn_focus_earth.setChecked(True)
        elif m == "MANUAL":
            self.btn_manual.setChecked(True)
        else:
            self.btn_focus_target.setChecked(True)

    @Slot(str)
    def _on_session_rebuilt(self, _msg: str) -> None:
        """Handle session rebuild by refreshing mode state."""
        self._on_observer_changed(self.combo_observer.currentIndex())

    # ------------------------------------------------------------------------
    # Real-Time Frame & Event Processing
    # ------------------------------------------------------------------------
    @Slot(object)
    def update_frame(self, fv: FrameView) -> None:
        """Receive a FrameView and route it through dual frame processing."""
        self.on_dual_frame_ready(fv.sat_id, fv)

    @Slot(str, object)
    def on_dual_frame_ready(self, sat_id: str, fv: FrameView) -> None:
        """Process incoming frame view for display and real-time state events."""
        if self._worker is None and hasattr(fv, "timestamp_s"):
            get_shared_clock().reset(fv.timestamp_s)
            t_sim = float(fv.timestamp_s)
        else:
            t_sim = get_shared_clock().now()

        # 1. Update display if frame is from the selected observer
        if sat_id == self._selected_sat:
            self._camera_view.update_frame(fv)
            self._update_selected_badge(fv.track_state)

            # Push aim derived from Phase 1A observer frame math and shared clock
            f_vec, u_vec = pan_tilt_to_world_aim(
                self._selected_sat, fv.pointing_pan_deg, fv.pointing_tilt_deg, t_sim
            )
            self.gimbal_3d_view.set_aim(
                f_vec, u_vec, pan_deg=fv.pointing_pan_deg, tilt_deg=fv.pointing_tilt_deg
            )
            self.gimbal_3d_view.set_time(t_sim)

            # Push real pipeline symbology to the 3D gimbal overlay
            is_predicting = fv.track_state in (TrackState.LOST, TrackState.REACQUIRE)
            self.gimbal_3d_view.update_overlay(
                fv.detections,
                fv.estimate,
                fv.gate_px,
                is_predicting,
            )
            self.frame_painted.emit()

        # 2. Lock state transitions
        if sat_id == "s1":
            if fv.is_locked and not self._s1_locked:
                self._s1_locked = True
                self.append_log(f"{t_sim:.1f}s — S-1 LOCKED on S-2", "lock")
            elif not fv.is_locked and self._s1_locked:
                self._s1_locked = False
                self.append_log(f"{t_sim:.1f}s — S-1 LOST S-2", "lost")

        elif sat_id == "s2":
            if fv.is_locked and not self._s2_locked:
                self._s2_locked = True
                self.append_log(f"{t_sim:.1f}s — S-2 LOCKED on S-1", "lock")
            elif not fv.is_locked and self._s2_locked:
                self._s2_locked = False
                self.append_log(f"{t_sim:.1f}s — S-2 LOST S-1", "lost")

        # 3. Two-way Handshake Detection
        both_locked = self._s1_locked and self._s2_locked
        if both_locked and not self._handshake_active:
            self._handshake_active = True
            self.append_log(
                f"{t_sim:.1f}s — HANDSHAKE ESTABLISHED (Dual Lock)", "handshake"
            )
            self._update_handshake_badge(True)
        elif not both_locked and self._handshake_active:
            self._handshake_active = False
            self.append_log(f"{t_sim:.1f}s — HANDSHAKE LOST", "lost")
            self._update_handshake_badge(False)

        # 4. Geometric Line-of-Sight & Earth Occlusion
        p1 = get_satellite_position("s1", t_sim)
        p2 = get_satellite_position("s2", t_sim)
        los_clear = check_line_of_sight(p1, p2)
        los_blocked = not los_clear

        if self._prev_los_blocked is None:
            self._prev_los_blocked = los_blocked
            if los_blocked:
                self.append_log(
                    f"{t_sim:.1f}s — LINK BLOCKED (Earth occlusion) — BLOCKED - gimbal holding", "occluded"
                )
                self._update_link_badge(False)
            else:
                self.append_log(f"{t_sim:.1f}s — LINK ACTIVE (LOS clear)", "clear")
                self._update_link_badge(True)
        elif los_blocked != self._prev_los_blocked:
            self._prev_los_blocked = los_blocked
            if los_blocked:
                self.append_log(
                    f"{t_sim:.1f}s — LINK BLOCKED (Earth occlusion) — BLOCKED - gimbal holding", "occluded"
                )
                self._update_link_badge(False)
            else:
                self.append_log(
                    f"{t_sim:.1f}s — LINK RESTORED (LOS clear) — LOS restored - slewing to ephemeris cue", "clear"
                )
                self._update_link_badge(True)

    # ------------------------------------------------------------------------
    # Status & Logging Helpers
    # ------------------------------------------------------------------------
    def append_log(self, text: str, tag: str = "info") -> None:
        """Append a formatted event line to the status text feed."""
        self._log_history.append(text)

        # Color mapping based on theme
        if tag == "lock":
            color = theme.COLOR_EMERALD.name()
            prefix = "✔ "
        elif tag == "lost":
            color = theme.COLOR_ROSE.name()
            prefix = "✖ "
        elif tag == "handshake":
            color = theme.COLOR_LIGHT_EMERALD.name()
            prefix = "★ "
        elif tag == "occluded":
            color = theme.COLOR_AMBER.name()
            prefix = "▲ "
        elif tag == "clear":
            color = theme.COLOR_BLUE.name()
            prefix = "● "
        else:
            color = theme.COLOR_GREY.name()
            prefix = "▪ "

        # Format with timestamp muted
        parts = text.split(" — ", 1)
        if len(parts) == 2:
            time_part, msg_part = parts
            html = (
                f'<div style="margin: 2px 0;">'
                f'<span style="color: {theme.COLOR_TIME_MUTED.name()}; font-weight: 600;">{time_part}</span>'
                f'<span style="color: {theme.COLOR_DOT_MUTED.name()};"> — </span>'
                f'<span style="color: {color}; font-weight: 600;">{prefix}{msg_part}</span>'
                f'</div>'
            )
        else:
            html = f'<div style="color: {color}; margin: 2px 0;">{prefix}{text}</div>'

        self.log_text.append(html)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def _update_selected_badge(self, state: TrackState) -> None:
        """Update the tracking state pill next to focus buttons."""
        name = state.name if hasattr(state, "name") else str(state)
        bg = theme.STATE_SEARCH_BG.name()
        fg = theme.STATE_SEARCH_TEXT.name()
        border = theme.STATE_SEARCH_PRIMARY.name()
        if state == TrackState.TRACK:
            bg = theme.STATE_TRACK_BG.name()
            fg = theme.STATE_TRACK_TEXT.name()
            border = theme.STATE_TRACK_PRIMARY.name()
        elif state in (TrackState.LOST, TrackState.REACQUIRE):
            bg = theme.STATE_LOST_BG.name()
            fg = theme.STATE_LOST_TEXT.name()
            border = theme.STATE_LOST_PRIMARY.name()
        elif state == TrackState.ACQUIRE:
            bg = theme.STATE_ACQUIRE_BG.name()
            fg = theme.STATE_ACQUIRE_TEXT.name()
            border = theme.STATE_ACQUIRE_PRIMARY.name()

        self.lbl_selected_status.setText(name)
        self.lbl_selected_status.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
            f"border-radius: 3px; padding: 2px 6px; font-weight: 700; font-size: 10px;"
        )

    def _update_link_badge(self, is_clear: bool) -> None:
        """Update top link indicator badge."""
        if is_clear:
            self.lbl_link_indicator.setText("LINK CLEAR")
            self.lbl_link_indicator.setStyleSheet(
                f"background-color: {theme.STATE_TRACK_BG.name()}; color: {theme.STATE_TRACK_TEXT.name()}; "
                f"border: 1px solid {theme.STATE_TRACK_PRIMARY.name()}; "
                f"border-radius: 3px; padding: 2px 6px; font-weight: 700; font-size: 10px;"
            )
        else:
            self.lbl_link_indicator.setText("LINK BLOCKED")
            self.lbl_link_indicator.setStyleSheet(
                f"background-color: {theme.STATE_LOST_BG.name()}; color: {theme.STATE_LOST_TEXT.name()}; "
                f"border: 1px solid {theme.STATE_LOST_PRIMARY.name()}; "
                f"border-radius: 3px; padding: 2px 6px; font-weight: 700; font-size: 10px;"
            )

    def _update_handshake_badge(self, is_handshake: bool) -> None:
        """Update top handshake indicator badge."""
        if is_handshake:
            self.lbl_handshake_indicator.setText("HANDSHAKE: ACTIVE")
            self.lbl_handshake_indicator.setStyleSheet(
                f"background-color: {theme.STATE_TRACK_BG.name()}; color: {theme.STATE_TRACK_TEXT.name()}; "
                f"border: 1px solid {theme.STATE_TRACK_PRIMARY.name()}; "
                f"border-radius: 3px; padding: 2px 6px; font-weight: 700; font-size: 10px;"
            )
        else:
            self.lbl_handshake_indicator.setText("HANDSHAKE: OFF")
            self.lbl_handshake_indicator.setStyleSheet(
                f"background-color: {theme.ALT_BASE_BG.name()}; color: {theme.TEXT_SECONDARY.name()}; "
                f"border: 1px solid {theme.BORDER_NORMAL.name()}; border-radius: 3px; "
                f"padding: 2px 6px; font-weight: 700; font-size: 10px;"
            )


__all__ = ("CameraFeedView",)
