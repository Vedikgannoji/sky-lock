"""Primary application window coordinating control panels, video display, and telemetry."""

from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import QEvent, QMetaObject, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import skylock
from skylock.config.models import InputConfig, SkyLockConfig
from skylock.core.geometry import angular_diff_deg
from skylock.core.los import (
    OrbitParams,
    calculate_look_angles,
    check_line_of_sight,
    is_target_in_fov,
    orbit_position_at_time,
    slew_toward_target,
)
from skylock.core.orbital_world import ORBIT_SPEED_SCALE
from skylock.core.sim_clock import get_shared_clock
from skylock.ui import theme
from skylock.benchmark.runner import BenchmarkRunner
from skylock.ui.config_editor import ConfigEditor
from skylock.ui.panels.benchmark import BenchmarkPanel
from skylock.ui.panels.configuration_view import ConfigurationView
from skylock.ui.panels.controls import ControlsPanel
from skylock.ui.panels.gimbal_control import GimbalControlPanel
from skylock.ui.panels.telemetry import TelemetryPanel
from skylock.ui.settings import AppSettings
from skylock.ui.web3d.view_3d import SpaceView3D
from skylock.ui.panels.camera_feed_view import CameraFeedView
from skylock.ui.widgets.camera_view import CameraView
from skylock.ui.widgets.state_timeline import StateTimeline
from skylock.ui.worker import SessionWorker


class ManualSteeringFilter(QObject):
    """Application-level event filter for manual steering with arrow keys and WASD.

    Handles key press/release events globally while respecting focus context:
    - Only active when control mode is MANUAL
    - Ignores auto-repeat events
    - Works even when child widgets (spinboxes, combos) have focus
    - Exception: ignores events when QLineEdit has focus
      and is actively being edited
    - Tracks multiple simultaneous key presses (e.g., Up + Right)
    - Sends (0, 0) rates when window loses focus
    """

    rate_changed = Signal(float, float)  # (pan_rate, tilt_rate)
    is_manual_mode: bool = False

    def __init__(
        self,
        parent: QObject | None = None,
        manual_rate_deg_s: float = 2.0,
    ) -> None:
        self.is_manual_mode = False
        super().__init__(parent)
        self.manual_rate_deg_s = manual_rate_deg_s
        self._pressed_keys: set[int] = set()

        # Key mappings
        self._pan_keys = {
            Qt.Key.Key_Left: -1.0,
            Qt.Key.Key_Right: 1.0,
            Qt.Key.Key_A: -1.0,
            Qt.Key.Key_D: 1.0,
        }
        self._tilt_keys = {
            Qt.Key.Key_Up: 1.0,
            Qt.Key.Key_Down: -1.0,
            Qt.Key.Key_W: 1.0,
            Qt.Key.Key_S: -1.0,
        }

    def set_manual_mode(self, is_manual: bool) -> None:
        """Update whether manual mode is active."""
        self.is_manual_mode = is_manual
        if not is_manual:
            self._pressed_keys.clear()
            self.rate_changed.emit(0.0, 0.0)

    def set_manual_rate(self, rate_deg_s: float) -> None:
        """Update the manual rate magnitude."""
        self.manual_rate_deg_s = max(0.0, rate_deg_s)
        self._emit_current_rates()

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        """Filter key events for manual steering."""
        if event is None or not getattr(self, "is_manual_mode", False):
            return False

        event_type = event.type()

        # Handle window deactivation
        if event_type == QEvent.Type.WindowDeactivate:
            self._pressed_keys.clear()
            self.rate_changed.emit(0.0, 0.0)
            return False

        # Only process key events
        if event_type not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False

        # Check if focus widget should block steering
        focused = watched if isinstance(watched, QWidget) else QApplication.focusWidget()
        if self._should_ignore_focus(focused):
            return False

        # Ignore auto-repeat
        if event.isAutoRepeat():  # type: ignore[attr-defined]
            return False

        key = event.key()  # type: ignore[attr-defined]
        all_keys = set(self._pan_keys.keys()) | set(self._tilt_keys.keys())

        if key not in all_keys:
            return False

        # Update pressed keys set
        if event_type == QEvent.Type.KeyPress:
            self._pressed_keys.add(key)
        elif event_type == QEvent.Type.KeyRelease:
            self._pressed_keys.discard(key)

        self._emit_current_rates()
        return True  # Consume the event

    def _should_ignore_focus(self, widget: QWidget | None) -> bool:
        """Check if the focused widget should block steering keys."""
        if widget is None:
            return False

        # Block standalone text input, but allow the internal editor used by spinboxes.
        if isinstance(widget, QLineEdit):
            parent = widget.parentWidget()
            while parent is not None:
                if isinstance(parent, QAbstractSpinBox):
                    return False
                parent = parent.parentWidget()
            return True

        # Block for combo boxes with open popups
        return bool(
            isinstance(widget, QComboBox)
            and hasattr(widget, "view")
            and widget.view().isVisible()
        )

    def _emit_current_rates(self) -> None:
        """Calculate and emit rates based on currently pressed keys."""
        pan_rate = 0.0
        tilt_rate = 0.0

        for key in self._pressed_keys:
            if key in self._pan_keys:
                pan_rate += self._pan_keys[key] * self.manual_rate_deg_s
            if key in self._tilt_keys:
                tilt_rate += self._tilt_keys[key] * self.manual_rate_deg_s

        self.rate_changed.emit(pan_rate, tilt_rate)


class MainWindow(QMainWindow):
    """Main application window for the SkyLock tracking interface."""

    def __init__(
        self,
        initial_config: SkyLockConfig | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SkyLock — Electro-Optical Tracking System")
        self.setMinimumSize(1100, 700)

        # Settings manager
        self.settings = AppSettings()

        # Single source of truth for configuration
        initial = initial_config if initial_config is not None else SkyLockConfig()
        self.editor = ConfigEditor(initial)
        self.manual_rate_deg_s = 2.0  # Default manual steering rate

        # Initialize worker and thread
        self._worker_thread = QThread(self)
        self._worker = SessionWorker(self.editor.config)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.initialize)

        # Manual steering event filter
        max_slew = self.editor.config.gimbal.slew_rate_deg_s
        self._steering_filter = ManualSteeringFilter(
            self, manual_rate_deg_s=min(self.manual_rate_deg_s, max_slew)
        )
        QApplication.instance().installEventFilter(self._steering_filter)  # type: ignore[union-attr]

        # Phase 4 Authoritative orbital & gimbal state with scaled speeds
        self._s1_orbit = OrbitParams(
            radius=20.0, speed=0.3 * ORBIT_SPEED_SCALE, inclination_deg=25.0, phase_deg=0.0
        )
        self._s2_orbit = OrbitParams(
            radius=26.0, speed=0.2 * ORBIT_SPEED_SCALE, inclination_deg=65.0, phase_deg=45.0
        )
        self._sim_time_s = 0.0
        self._current_pan = 0.0
        self._current_tilt = 0.0
        self._current_fov = 20.0
        self._is_auto_tracking = False

        self._build_ui()
        self._build_menus()
        self._connect_signals()

        # Restore settings or apply defaults
        restored = self.settings.restore_window_state(self)
        if not restored:
            self.resize(1280, 800)
            self._apply_default_layout()

        # Restore UI flags
        show_gt = self.settings.get_show_ground_truth()
        show_legend = self.settings.get_show_legend()
        self.chk_debug_gt.setChecked(show_gt)
        self.chk_legend.setChecked(show_legend)
        self.camera_view.show_ground_truth = show_gt
        self.camera_view.show_legend = show_legend

        self._worker_thread.start()

    def _build_ui(self) -> None:
        # Central widget: Vertical splitter with camera/timeline top, tabs bottom
        central_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section: 3D space visualization + camera view + state timeline
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.setSpacing(4)

        self.view_tabs = QTabWidget(top_widget)
        self.space_view_3d = SpaceView3D(self.view_tabs)
        self.toolbar_3d = self.space_view_3d.toolbar
        self.btn_pause_3d = self.space_view_3d.btn_pause
        self.camera_view = CameraFeedView(self.view_tabs)
        self.camera_view.set_fov(self.editor.config.camera.fov_v_deg)
        self.configuration_view = ConfigurationView(self.view_tabs)
        self.view_tabs.addTab(self.space_view_3d, "3D Space Simulation")
        self.view_tabs.addTab(self.camera_view, "Camera Sensor Feed")
        self.view_tabs.addTab(self.configuration_view, "Configuration")
        # 4th tab: Connection Feed (event log with link/handshake badges)
        self.connection_feed_widget = self.camera_view.create_connection_feed_widget(self.view_tabs)
        self.view_tabs.addTab(self.connection_feed_widget, "Connection Feed")
        top_layout.addWidget(self.view_tabs, stretch=10)

        # State timeline mounted under camera view (hidden by default per Phase 2 Rule 5)
        self.state_timeline = StateTimeline()
        self.state_timeline.hide()
        top_layout.addWidget(self.state_timeline)

        # Checkable debug toggles retained for headless/settings/test compatibility,
        # but moved out of primary visualization space into View -> Debug menu
        self.chk_debug_gt = QCheckBox("Show ground truth (debug)")
        self.chk_debug_gt.setChecked(False)
        self.chk_debug_gt.setToolTip("Display ground truth overlay (simulation only)")
        self.chk_debug_gt.toggled.connect(self._on_toggle_gt)
        self.chk_debug_gt.hide()

        self.chk_legend = QCheckBox("Legend")
        self.chk_legend.setChecked(False)
        self.chk_legend.setToolTip("Show symbology legend")
        self.chk_legend.toggled.connect(self._on_toggle_legend)
        self.chk_legend.hide()

        central_splitter.addWidget(top_widget)

        # Bottom section: Tabs for Benchmark and other future panels
        self.tab_widget = QTabWidget()
        self.bench_panel = BenchmarkPanel(runner=BenchmarkRunner(self.editor.config))
        self.bench_panel._mp4_path = self.editor.config.input.mp4_path
        self.tab_widget.addTab(self.bench_panel, "Benchmark")

        central_splitter.addWidget(self.tab_widget)

        # Set splitter stretch factors: camera area gets most space
        central_splitter.setStretchFactor(0, 10)
        central_splitter.setStretchFactor(1, 1)

        # Set default sizes (will be overridden by settings if restored)
        central_splitter.setSizes([600, 220])

        self.central_splitter = central_splitter
        self.setCentralWidget(central_splitter)

        # Left Dock: Controls
        self.dock_controls = QDockWidget("Controls", self)
        self.controls_panel = ControlsPanel(self.editor)
        self.dock_controls.setWidget(self.controls_panel)
        self.dock_controls.setMinimumWidth(340)
        self.dock_controls.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_controls)

        # Right Dock: Telemetry & Gimbal Control
        self.dock_telemetry = QDockWidget("Telemetry", self)
        right_container = QWidget()
        right_container.setMinimumWidth(0)
        from PySide6.QtCore import QSize
        right_container.minimumSizeHint = lambda: QSize(280, 0)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.telemetry_panel = TelemetryPanel()
        self.gimbal_control_panel = GimbalControlPanel()

        right_layout.addWidget(self.telemetry_panel)
        right_layout.addWidget(self.gimbal_control_panel)
        right_layout.addStretch(1)

        self.dock_telemetry.setWidget(right_container)
        self.dock_telemetry.setMinimumWidth(280)
        self.dock_telemetry.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_telemetry)

        # Status Bar with permanent widgets
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Permanent status widgets (hidden to remove redundant live diagnostics per Phase 2 Rule 6)
        self.lbl_status_state = QLabel("SEARCH")
        self.lbl_status_state.setToolTip("Current tracking state")
        self.lbl_status_state.hide()
        self.status_bar.addPermanentWidget(self.lbl_status_state)

        self.lbl_status_source = QLabel("simulation")
        self.lbl_status_source.setToolTip("Input source")
        self.lbl_status_source.hide()
        self.status_bar.addPermanentWidget(self.lbl_status_source)

        self.lbl_status_frame = QLabel("frame 0")
        self.lbl_status_frame.setToolTip("Current frame number")
        self.lbl_status_frame.hide()
        self.status_bar.addPermanentWidget(self.lbl_status_frame)

        self.lbl_status_fps = QLabel("0.0 fps")
        self.lbl_status_fps.setToolTip("Wall-clock rendering FPS")
        self.lbl_status_fps.hide()
        self.status_bar.addPermanentWidget(self.lbl_status_fps)

        self.lbl_status_pending = QLabel("")
        self.status_bar.addPermanentWidget(self.lbl_status_pending)

        self.status_bar.showMessage("Ready")

    def _build_menus(self) -> None:
        """Build menu bar with File, Run, View, and Help menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        act_load_config = QAction("&Load Config...", self)
        act_load_config.setToolTip("Load configuration from JSON file")
        act_load_config.triggered.connect(self._on_load_config)
        file_menu.addAction(act_load_config)

        act_save_config = QAction("&Save Config...", self)
        act_save_config.setToolTip("Save current configuration to JSON file")
        act_save_config.triggered.connect(self._on_save_config)
        file_menu.addAction(act_save_config)

        file_menu.addSeparator()

        act_export_json = QAction("Export Benchmark &JSON...", self)
        act_export_json.setToolTip("Export benchmark results as JSON")
        act_export_json.triggered.connect(self.bench_panel._export_json)
        file_menu.addAction(act_export_json)

        act_export_md = QAction("Export Benchmark &Markdown...", self)
        act_export_md.setToolTip("Export benchmark results as Markdown")
        act_export_md.triggered.connect(self.bench_panel._export_md)
        file_menu.addAction(act_export_md)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Run menu
        run_menu = menubar.addMenu("&Run")

        self.act_start = QAction("&Start", self)
        self.act_start.setShortcut(QKeySequence("Ctrl+R"))
        self.act_start.setToolTip("Start tracking session (Ctrl+R)")
        self.act_start.triggered.connect(self.controls_panel._on_start)
        run_menu.addAction(self.act_start)

        self.act_stop = QAction("S&top", self)
        self.act_stop.setShortcut(QKeySequence("Ctrl+."))
        self.act_stop.setToolTip("Stop tracking session (Ctrl+.)")
        self.act_stop.triggered.connect(lambda: self.controls_panel.stop_clicked.emit())
        run_menu.addAction(self.act_stop)

        self.act_reset = QAction("&Reset", self)
        self.act_reset.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.act_reset.setToolTip("Reset session (Ctrl+Shift+R)")
        self.act_reset.triggered.connect(lambda: self.controls_panel.reset_clicked.emit())
        run_menu.addAction(self.act_reset)

        # View menu
        view_menu = menubar.addMenu("&View")

        act_toggle_controls = self.dock_controls.toggleViewAction()
        act_toggle_controls.setText("&Controls")
        view_menu.addAction(act_toggle_controls)

        act_toggle_telemetry = self.dock_telemetry.toggleViewAction()
        act_toggle_telemetry.setText("&Telemetry")
        view_menu.addAction(act_toggle_telemetry)

        view_menu.addSeparator()

        act_view_3d = QAction("3D &Space Simulation", self)
        act_view_3d.setToolTip("Switch central view to 3D Space Simulation")
        act_view_3d.triggered.connect(lambda: self.view_tabs.setCurrentWidget(self.space_view_3d))
        view_menu.addAction(act_view_3d)

        act_view_sensor = QAction("&Camera Sensor Feed", self)
        act_view_sensor.setToolTip("Switch central view to 2D Camera Sensor Feed")
        act_view_sensor.triggered.connect(lambda: self.view_tabs.setCurrentWidget(self.camera_view))
        view_menu.addAction(act_view_sensor)

        act_view_config = QAction("3D &Configuration", self)
        act_view_config.setToolTip("Switch central view to 3D Space & Orbit Configuration")
        act_view_config.triggered.connect(
            lambda: self.view_tabs.setCurrentWidget(self.configuration_view)
        )
        view_menu.addAction(act_view_config)

        view_menu.addSeparator()

        act_focus_s1 = QAction("Focus S-1 (Observer)", self)
        act_focus_s1.setToolTip("Focus 3D viewing camera on Observer Satellite S-1")
        act_focus_s1.triggered.connect(self.space_view_3d.focus_s1)
        view_menu.addAction(act_focus_s1)

        act_focus_s2 = QAction("Focus S-2 (Target)", self)
        act_focus_s2.setToolTip("Focus 3D viewing camera on Target Satellite S-2")
        act_focus_s2.triggered.connect(self.space_view_3d.focus_s2)
        view_menu.addAction(act_focus_s2)

        act_reset_cam = QAction("Reset Camera Pose", self)
        act_reset_cam.setToolTip("Reset gimbal camera to neutral pose (pan=0°, tilt=0°, FOV=20°)")
        act_reset_cam.triggered.connect(self._on_gimbal_reset)
        view_menu.addAction(act_reset_cam)

        view_menu.addSeparator()

        act_demo_3d = QAction("Run 3D &Gimbal Demo", self)
        act_demo_3d.setToolTip("Run deterministic 3D gimbal pan/tilt validation sequence")
        act_demo_3d.triggered.connect(self.start_3d_demonstration)
        view_menu.addAction(act_demo_3d)

        view_menu.addSeparator()

        # Debug submenu (Phase 2 Rule 5)
        debug_menu = view_menu.addMenu("&Debug")
        self.debug_menu = debug_menu

        act_show_gt = QAction("Show &Ground Truth", self, checkable=True)
        act_show_gt.setChecked(self.chk_debug_gt.isChecked())
        act_show_gt.toggled.connect(self.chk_debug_gt.setChecked)
        debug_menu.addAction(act_show_gt)

        act_show_legend = QAction("Show &Legend", self, checkable=True)
        act_show_legend.setChecked(self.chk_legend.isChecked())
        act_show_legend.toggled.connect(self.chk_legend.setChecked)
        debug_menu.addAction(act_show_legend)

        act_show_timeline = QAction("Show State &Timeline", self, checkable=True)
        act_show_timeline.setChecked(False)
        act_show_timeline.toggled.connect(self.state_timeline.setVisible)
        debug_menu.addAction(act_show_timeline)

        # Retain top-level actions for test compatibility (test_g6_theme_layout)
        view_menu.addAction(act_show_gt)
        view_menu.addAction(act_show_legend)

        view_menu.addSeparator()

        act_reset_layout = QAction("Reset &Layout", self)
        act_reset_layout.setToolTip("Reset window layout to defaults")
        act_reset_layout.triggered.connect(self._on_reset_layout)
        view_menu.addAction(act_reset_layout)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        act_about = QAction("&About SkyLock", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

        act_shortcuts = QAction("&Keyboard Shortcuts", self)
        act_shortcuts.triggered.connect(self._on_shortcuts)
        help_menu.addAction(act_shortcuts)

    def _apply_default_layout(self) -> None:
        """Apply default window layout (called when settings not restored)."""
        # Default splitter sizes: camera gets ~60% of height
        height = self.height()
        camera_height = int(height * 0.6)
        tabs_height = height - camera_height
        self.central_splitter.setSizes([camera_height, tabs_height])

    def _connect_signals(self) -> None:
        # Controls -> Worker
        self.controls_panel.start_clicked.connect(self._worker.start_running)
        self.controls_panel.stop_clicked.connect(self._worker.stop_running)
        self.controls_panel.reset_clicked.connect(self._worker.reset_session)
        self.controls_panel.reset_clicked.connect(self._on_reset_ui)
        self.controls_panel.config_changed.connect(self._worker.apply_config)
        self.controls_panel.config_changed.connect(self._on_controls_config_changed)
        self.controls_panel.mode_changed.connect(self._on_mode_changed)

        # Worker -> Views
        self.camera_view.set_worker(self._worker, self._steering_filter)
        self._worker.frame_ready.connect(self.camera_view.update_frame)
        self._worker.frame_ready.connect(self._on_telemetry_frame)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.session_error.connect(self._on_session_error)
        self._worker.running_changed.connect(self._on_running_changed)
        self._worker.session_rebuilt.connect(self._on_session_rebuilt)
        self._worker.session_finished.connect(self._on_session_finished)

        # Manual steering
        self._steering_filter.rate_changed.connect(self._worker.set_manual_rates)

        # Configuration View -> 3D Space View & MainWindow
        self.configuration_view.orbit_lines_toggled.connect(self.space_view_3d.set_show_orbit_lines)
        self.configuration_view.tracking_beam_toggled.connect(self.space_view_3d.set_show_tracking_beam)
        self.configuration_view.orbits_changed.connect(self._on_orbits_changed)

        # Gimbal Control Panel -> State & 3D Space View
        self.gimbal_control_panel.pan_changed.connect(self._on_gimbal_manual_pan)
        self.gimbal_control_panel.tilt_changed.connect(self._on_gimbal_manual_tilt)
        self.gimbal_control_panel.fov_changed.connect(self._on_gimbal_manual_fov)
        self.gimbal_control_panel.reset_clicked.connect(self._on_gimbal_reset)
        self.gimbal_control_panel.track_target_clicked.connect(self._on_gimbal_track_target)

        # Seed link: sync controls panel seed to benchmark panel
        self.controls_panel.spn_seed.valueChanged.connect(
            self.bench_panel.sync_controls_seed
        )
        self.bench_panel.sync_controls_seed(self.controls_panel.spn_seed.value())

        # Back-pressure: camera view acknowledges frames
        self.camera_view.frame_painted.connect(self._worker.ack_frame)
        self.view_tabs.currentChanged.connect(lambda _: self._worker.ack_frame())

        # Synchronized pause across both 3D views (Space Simulation tab & Gimbal Camera tab) and shared sim clock
        self.space_view_3d.pause_toggled.connect(self.camera_view.set_paused)
        self.camera_view.pause_toggled.connect(self.space_view_3d.set_paused)
        self.space_view_3d.pause_toggled.connect(self._on_pause_clock_sync)
        self.camera_view.pause_toggled.connect(self._on_pause_clock_sync)

        # Push scaled orbit parameters to both 3D views when loaded
        self.space_view_3d.scene_ready.connect(self._push_orbits_to_views)
        self.camera_view.gimbal_cam.scene_ready.connect(self._push_orbits_to_views)

    def _on_telemetry_frame(self, fv: Any) -> None:  # noqa: ANN401
        """Forward frame to telemetry panel only if worker is active."""
        if self._worker.is_running:
            self.telemetry_panel.update_telemetry(fv)

    def _on_frame_ready(self, fv: Any) -> None:  # noqa: ANN401
        """Update timeline, status bar, and 3D simulation with latest state."""
        if hasattr(fv, "state_history_tail"):
            self.state_timeline.set_history(fv.state_history_tail)

        # Update status bar permanent widgets
        if hasattr(fv, "track_state"):
            state_name = (
                fv.track_state.name
                if hasattr(fv.track_state, "name")
                else str(fv.track_state)
            )
            self.lbl_status_state.setText(state_name)
        elif hasattr(self, "_s1_orbit") and self._s1_orbit is not None:
            from skylock.core.geometry import angular_diff_deg
            from skylock.core.los import calculate_look_angles, check_line_of_sight, orbit_position_at_time
            t_s = getattr(self, "_sim_time_s", 0.0)
            p1 = orbit_position_at_time(self._s1_orbit, t_s)
            p2 = orbit_position_at_time(self._s2_orbit, t_s)
            has_los = check_line_of_sight(p1, p2, body_radius=10.0)
            if not has_los:
                self.lbl_status_state.setText("LOST")
                self.telemetry_panel.lbl_lock.setText("UNLOCKED")
            else:
                target_pan, target_tilt, _ = calculate_look_angles(p1, p2)
                d_pan = abs(angular_diff_deg(target_pan, self._current_pan))
                d_tilt = abs(angular_diff_deg(target_tilt, self._current_tilt))
                if d_pan < 2.0 and d_tilt < 2.0:
                    self.lbl_status_state.setText("TRACK")
                    self.telemetry_panel.lbl_lock.setText("ENGAGED")
                else:
                    self.lbl_status_state.setText("ACQUIRE")
                    self.telemetry_panel.lbl_lock.setText("ACQUIRING")
        if hasattr(fv, "frame_index"):
            self.lbl_status_frame.setText(f"frame {fv.frame_index}")
        if hasattr(fv, "wall_fps") and fv.wall_fps is not None:
            self.lbl_status_fps.setText(f"{fv.wall_fps:.1f} fps")
        if hasattr(fv, "input_kind"):
            self.lbl_status_source.setText(fv.input_kind)

        # Synchronize simulation time to 3D Space View from shared clock
        t_sim = get_shared_clock().now()
        if hasattr(self, "space_view_3d"):
            self.space_view_3d.set_time(t_sim)

        # Forward actual gimbal pointing to right-panel display & telemetry & 3D Space View
        if (
            hasattr(fv, "pointing_pan_deg")
            and hasattr(fv, "pointing_tilt_deg")
            and fv.pointing_pan_deg is not None
            and fv.pointing_tilt_deg is not None
        ):
            self._current_pan = float(fv.pointing_pan_deg)
            self._current_tilt = float(fv.pointing_tilt_deg)
            if hasattr(self, "space_view_3d"):
                self.space_view_3d.set_gimbal_pose(self._current_pan, self._current_tilt)
            if hasattr(self, "gimbal_control_panel"):
                self.gimbal_control_panel.set_values(
                    self._current_pan, self._current_tilt, emit_signals=False
                )
            if hasattr(self, "telemetry_panel"):
                self.telemetry_panel.update_gimbal(self._current_pan, self._current_tilt)

        # If camera_view is not the current active tab (e.g. 3D Space Simulation is active),
        # acknowledge the frame directly because camera_view.paintEvent will not trigger.
        if hasattr(self, "view_tabs") and self.view_tabs.currentWidget() != self.camera_view:
            self._worker.ack_frame()

    def _on_gimbal_manual_pan(self, pan: float) -> None:
        sel_sat = self.camera_view.selected_satellite
        self._current_pan = float(pan)
        if self._is_auto_tracking:
            self._on_mode_changed("MANUAL")
            self.controls_panel.cmb_mode.setCurrentText("MANUAL")
        self._worker.set_gimbal_pointing(sel_sat, self._current_pan, self._current_tilt)

    def _on_gimbal_manual_tilt(self, tilt: float) -> None:
        sel_sat = self.camera_view.selected_satellite
        self._current_tilt = float(tilt)
        if self._is_auto_tracking:
            self._on_mode_changed("MANUAL")
            self.controls_panel.cmb_mode.setCurrentText("MANUAL")
        self._worker.set_gimbal_pointing(sel_sat, self._current_pan, self._current_tilt)

    def _on_gimbal_manual_fov(self, fov: float) -> None:
        self._current_fov = float(fov)
        self.space_view_3d.set_camera_fov(self._current_fov)
        self.camera_view.set_fov(self._current_fov)
        self._worker.set_camera_fov(self._current_fov)

    def _on_gimbal_reset(self) -> None:
        sel_sat = self.camera_view.selected_satellite
        self._current_pan = 0.0
        self._current_tilt = 0.0
        self._current_fov = 16.0
        self._worker.set_gimbal_pointing(sel_sat, 0.0, 0.0)
        self._worker.set_camera_fov(16.0)
        self.gimbal_control_panel.set_values(0.0, 0.0, 16.0, emit_signals=False)
        self.space_view_3d.reset_camera()
        self.camera_view.set_fov(16.0)
        self.camera_view.gimbal_cam.reset_pose()
        self.telemetry_panel.update_gimbal(0.0, 0.0)

    def _on_gimbal_track_target(self) -> None:
        sel_sat = self.camera_view.selected_satellite
        self._is_auto_tracking = True
        self._worker.set_mode(sel_sat, "AUTO")
        self.camera_view.sync_mode_from_external("AUTO")
        self.controls_panel.cmb_mode.setCurrentText("AUTO")

    def _on_orbits_changed(self, data: dict[str, Any]) -> None:
        if "s1" in data:
            s1 = data["s1"]
            r = float(s1.get("radius", self._s1_orbit.radius))
            inc = float(s1.get("inclination", self._s1_orbit.inclination_deg))
            spd = float(s1.get("speed", self._s1_orbit.speed))
            ph = float(s1.get("phase", self._s1_orbit.phase_deg))
            self._s1_orbit = OrbitParams(
                radius=r,
                speed=spd,
                inclination_deg=inc,
                phase_deg=ph,
            )
            self.space_view_3d.set_satellite_orbit("s1", r, inc, spd, ph)
            self.camera_view.gimbal_cam.set_satellite_orbit("s1", r, inc, spd, ph)
        if "s2" in data:
            s2 = data["s2"]
            r = float(s2.get("radius", self._s2_orbit.radius))
            inc = float(s2.get("inclination", self._s2_orbit.inclination_deg))
            spd = float(s2.get("speed", self._s2_orbit.speed))
            ph = float(s2.get("phase", self._s2_orbit.phase_deg))
            self._s2_orbit = OrbitParams(
                radius=r,
                speed=spd,
                inclination_deg=inc,
                phase_deg=ph,
            )
            self.space_view_3d.set_satellite_orbit("s2", r, inc, spd, ph)
            self.camera_view.gimbal_cam.set_satellite_orbit("s2", r, inc, spd, ph)

    def _on_mode_changed(self, mode: str) -> None:
        """Handle mode change: update worker and steering filter."""
        self._worker.set_control_mode(mode)
        self._steering_filter.set_manual_mode(mode == "MANUAL")
        self._is_auto_tracking = (mode == "AUTO")
        self.camera_view.sync_mode_from_external(mode)

    def _on_controls_config_changed(self, cfg: SkyLockConfig) -> None:
        """Sync updated configuration to the benchmark runner and internal state."""
        if hasattr(self, "bench_panel") and self.bench_panel is not None:
            self.bench_panel.runner.base_config = cfg
            self.bench_panel._mp4_path = cfg.input.mp4_path

    def _on_toggle_gt(self, checked: bool) -> None:
        self.camera_view.show_ground_truth = checked
        self.settings.set_show_ground_truth(checked)

    def _on_toggle_legend(self, checked: bool) -> None:
        self.camera_view.show_legend = checked
        self.settings.set_show_legend(checked)

    def _on_reset_ui(self) -> None:
        """Clear camera, timeline, and telemetry displays on reset."""
        get_shared_clock().reset()
        self.camera_view.clear()
        self.state_timeline.clear()
        self.telemetry_panel.clear()

    def _on_pause_clock_sync(self, paused: bool) -> None:
        """Synchronize pause state to the authoritative simulation clock."""
        clock = get_shared_clock()
        if paused:
            clock.pause()
        else:
            clock.resume()

    def _push_orbits_to_views(self) -> None:
        """Push scaled orbit parameters to both 3D visualization views."""
        self.space_view_3d.set_satellite_orbit(
            "s1",
            self._s1_orbit.radius,
            self._s1_orbit.inclination_deg,
            self._s1_orbit.speed,
            self._s1_orbit.phase_deg,
        )
        self.space_view_3d.set_satellite_orbit(
            "s2",
            self._s2_orbit.radius,
            self._s2_orbit.inclination_deg,
            self._s2_orbit.speed,
            self._s2_orbit.phase_deg,
        )
        self.camera_view.gimbal_cam.set_satellite_orbit(
            "s1",
            self._s1_orbit.radius,
            self._s1_orbit.inclination_deg,
            self._s1_orbit.speed,
            self._s1_orbit.phase_deg,
        )
        self.camera_view.gimbal_cam.set_satellite_orbit(
            "s2",
            self._s2_orbit.radius,
            self._s2_orbit.inclination_deg,
            self._s2_orbit.speed,
            self._s2_orbit.phase_deg,
        )

    def _on_session_error(self, err: str) -> None:
        self.status_bar.showMessage(f"Error: {err}", 5000)

    def _on_running_changed(self, running: bool) -> None:
        state_str = "Running" if running else "Stopped"
        self.status_bar.showMessage(f"Session {state_str}")

    def _on_session_rebuilt(self, msg: str) -> None:
        """Show session rebuilt notification."""
        self.status_bar.showMessage(msg, 3000)

    def _on_session_finished(self, msg: str) -> None:
        """Show persistent end-of-stream message."""
        self.status_bar.showMessage(msg)  # No timeout - persistent

    def _on_load_config(self) -> None:
        """Load configuration from JSON file."""
        last_dir = self.settings.get_last_config_directory()
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", last_dir or "", "JSON Files (*.json);;All Files (*.*)"
        )
        if path:
            try:
                import json
                from pathlib import Path
                with Path(path).open() as f:
                    data = json.load(f)
                from skylock.config.io import from_dict
                new_config = from_dict(data)
                self.editor.replace_config(new_config)
                self.settings.set_last_config_directory(path)
                self.status_bar.showMessage(f"Loaded config from {Path(path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Load Config Error", f"Failed to load config:\n{e}")

    def _on_save_config(self) -> None:
        """Save current configuration to JSON file."""
        last_dir = self.settings.get_last_config_directory()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", last_dir or "", "JSON Files (*.json);;All Files (*.*)"
        )
        if path:
            try:
                import json
                from pathlib import Path

                from skylock.config.io import to_dict
                with Path(path).open("w") as f:
                    json.dump(to_dict(self.editor.config), f, indent=2)
                self.settings.set_last_config_directory(path)
                self.status_bar.showMessage(f"Saved config to {Path(path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Save Config Error", f"Failed to save config:\n{e}")

    def _on_reset_layout(self) -> None:
        """Reset window layout to defaults."""
        self.settings.reset_layout()
        QMessageBox.information(
            self,
            "Layout Reset",
            "Window layout will be reset to defaults on next launch."
        )

    def _on_about(self) -> None:
        """Show About dialog."""
        QMessageBox.about(
            self,
            "About SkyLock",
            f"<h3>SkyLock</h3>"
            f"<p>Version {skylock.__version__}</p>"
            f"<p>Electro-Optical Tracking System</p>"
            f"<p>A precision target tracking simulation and control interface.</p>"
        )

    def _on_shortcuts(self) -> None:
        """Show keyboard shortcuts dialog."""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <table cellpadding="4">
        <tr><td><b>Ctrl+R</b></td><td>Start tracking</td></tr>
        <tr><td><b>Ctrl+.</b></td><td>Stop tracking</td></tr>
        <tr><td><b>Ctrl+Shift+R</b></td><td>Reset session</td></tr>
        <tr><td colspan="2">&nbsp;</td></tr>
        <tr><td colspan="2"><b>Manual Control Mode:</b></td></tr>
        <tr><td><b>Arrow Keys</b></td><td>Pan/tilt gimbal</td></tr>
        <tr><td><b>W/A/S/D</b></td><td>Pan/tilt gimbal (alternative)</td></tr>
        <tr><td><b>Up/W</b></td><td>Tilt up</td></tr>
        <tr><td><b>Down/S</b></td><td>Tilt down</td></tr>
        <tr><td><b>Left/A</b></td><td>Pan left</td></tr>
        <tr><td><b>Right/D</b></td><td>Pan right</td></tr>
        </table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    def start_3d_demonstration(self) -> None:
        """Run deterministic 3D visualization demonstration sequence (Phase 3 Section 16)."""
        self.view_tabs.setCurrentWidget(self.space_view_3d)

        demo_steps: list[tuple[float, float, int]] = [
            (0.0, 0.0, 0),
            (10.0, -5.0, 1200),
            (25.0, 10.0, 2400),
            (-15.0, -10.0, 3600),
            (90.0, 0.0, 4800),
        ]

        for pan, tilt, delay in demo_steps:
            if delay == 0:
                self.space_view_3d.set_gimbal_pose(pan, tilt)
            else:
                QTimer.singleShot(
                    delay, lambda p=pan, t=tilt: self.space_view_3d.set_gimbal_pose(p, t)
                )

    def closeEvent(self, event: Any) -> None:  # noqa: ANN401
        """Cleanly shut down worker thread and save settings on application close."""
        # Save settings
        self.settings.save_window_state(self)
        self.settings.save_splitter_sizes(self.central_splitter, "main")

        # Cleanup 3D Space View
        if hasattr(self, "space_view_3d"):
            self.space_view_3d.cleanup()

        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self._steering_filter)

        # Shutdown benchmark panel
        self.bench_panel.shutdown()

        # Stop worker using blocking call from worker thread
        QMetaObject.invokeMethod(
            self._worker,
            "shutdown",
            Qt.ConnectionType.BlockingQueuedConnection,
        )

        # Stop and wait for thread
        self._worker_thread.quit()
        if not self._worker_thread.wait(3000):
            # Thread didn't stop in time - log and terminate as last resort
            self.status_bar.showMessage("Warning: Worker thread forced termination")
            self._worker_thread.terminate()
            self._worker_thread.wait(1000)

        event.accept()


def run_app(
    argv: list[str] | None = None,
    initial_config: SkyLockConfig | None = None,
) -> int:
    """Launch the SkyLock graphical user interface."""
    app = QApplication(argv if argv is not None else sys.argv)
    theme.apply_theme(app)
    if initial_config is None:
        initial_config = SkyLockConfig(input=InputConfig(kind="orbital"))
    window = MainWindow(initial_config=initial_config)
    window.show()
    return app.exec()


__all__ = ("MainWindow", "run_app")
