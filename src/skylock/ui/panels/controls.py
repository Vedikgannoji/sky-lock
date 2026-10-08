"""Configuration and runtime control panel for the SkyLock tracking interface."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from skylock.config.validation import ConfigError
from skylock.core.enums import InputKind
from skylock.ui import theme
from skylock.ui.config_editor import ConfigEditor
from skylock.ui.config_model import (
    MOTION_KINDS,
    SHAPES,
    disturbance_toggle_override,
    slew_override,
    target_count_override,
    target_override,
)
from skylock.ui.panels.controls_sections import (
    DisturbanceMagnitudeRow,
    MotionParamsStack,
    Mp4InputSection,
    PresetsDropdown,
)
from skylock.ui.widgets.no_wheel import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)

# Disabled-button stylesheet (G-07)
_BTN_START_STYLE = (
    f"QPushButton {{ background-color: {theme.BUTTON_START_BG.name()}; "
    f"color: white; font-weight: bold; }}\n"
    f"QPushButton:disabled {{ background-color: {theme.BUTTON_DISABLED_BG.name()}; "
    f"color: {theme.BUTTON_DISABLED_TEXT.name()}; }}"
)


class ControlsPanel(QWidget):
    """Left dock control panel providing interactive system configuration."""

    config_changed = Signal(object)  # Emits SkyLockConfig
    start_clicked = Signal()
    stop_clicked = Signal()
    reset_clicked = Signal()
    mode_changed = Signal(str)  # Emits "AUTO" or "MANUAL" for live mode changes

    def __init__(self, editor: ConfigEditor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.editor = editor
        self._block_signals = True
        self._mp4_probe_ok = False  # Track whether MP4 probe succeeded
        self._error_labels: dict[str, QLabel] = {}  # key-substring -> inline error label
        self._build_ui()
        self._block_signals = False
        self.sync_from_config(self.editor.config)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)

        # 1. Execution controls (Start / Stop / Reset)
        run_box = QGroupBox("Run Control")
        r_layout = QHBoxLayout(run_box)
        self.btn_start = QPushButton("Start")
        self.btn_start.setFixedHeight(28)
        self.btn_start.setStyleSheet(_BTN_START_STYLE)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(28)
        self.btn_stop.setStyleSheet(
            f"background-color: {theme.BUTTON_STOP_BG.name()}; color: white; font-weight: bold;"
        )
        self.btn_stop.clicked.connect(lambda: self.stop_clicked.emit())
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setFixedHeight(28)
        self.btn_reset.clicked.connect(lambda: self.reset_clicked.emit())
        r_layout.addWidget(self.btn_start)
        r_layout.addWidget(self.btn_stop)
        r_layout.addWidget(self.btn_reset)
        layout.addWidget(run_box)

        # Summary Error Display (fallback)
        self.lbl_error = QLabel()
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet(
            f"color: {theme.STATUS_ERROR.name()}; font-weight: bold; padding: 4px;"
        )
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

        # Tabs: Operations (primary operational controls) vs Advanced (simulation tuning)
        self.tabs = QTabWidget()

        # Tab 1: Operations
        tab_ops = QWidget()
        layout_ops = QVBoxLayout(tab_ops)
        layout_ops.setContentsMargins(2, 6, 2, 2)
        layout_ops.setSpacing(8)
        self._build_input_section(layout_ops)
        self._build_mode_section(layout_ops)
        self._build_camera_gimbal_section(layout_ops)
        self._build_target_section(layout_ops)
        self._build_motion_section(layout_ops)
        layout_ops.addStretch()
        self.tabs.addTab(tab_ops, "Operations")

        # Tab 2: Advanced (Disturbances, Presets, RNG Seed)
        tab_adv = QWidget()
        layout_adv = QVBoxLayout(tab_adv)
        layout_adv.setContentsMargins(2, 6, 2, 2)
        layout_adv.setSpacing(8)
        self._build_disturbance_section(layout_adv)
        self._build_presets_section(layout_adv)
        self._build_seed_section(layout_adv)
        layout_adv.addStretch()
        self.tabs.addTab(tab_adv, "Advanced")

        layout.addWidget(self.tabs)
        layout.addStretch()
        self.scroll_area.setWidget(container)
        main_layout.addWidget(self.scroll_area)

    def _build_input_section(self, layout: QVBoxLayout) -> None:
        input_box = QGroupBox("Input Source")
        i_layout = QFormLayout(input_box)
        self.cmb_input = NoWheelComboBox()
        self.cmb_input.addItems(["Orbital (S-1 <-> S-2)", "Simulation", "MP4 Video"])
        self.cmb_input.currentIndexChanged.connect(self._on_input_changed)
        self.btn_browse_mp4 = QPushButton("Browse...")
        self.btn_browse_mp4.clicked.connect(self._browse_mp4)
        self.btn_browse_mp4.hide()

        self.mp4_section = Mp4InputSection(self)
        self.mp4_section.hide()
        self.mp4_section.fps_override_changed.connect(self._on_mp4_fps_override)
        self.mp4_section.loop_changed.connect(self._on_mp4_loop)

        i_layout.addRow("Source:", self.cmb_input)
        i_layout.addRow(self.btn_browse_mp4)
        i_layout.addRow(self.mp4_section)

        self._add_error_label(i_layout, "input.", "input_error")
        layout.addWidget(input_box)

    def _build_mode_section(self, layout: QVBoxLayout) -> None:
        """Gimbal control mode selector (AUTO/MANUAL). Manual rate not exposed in UI."""
        mode_box = QGroupBox("Gimbal Control Mode")
        m_layout = QFormLayout(mode_box)
        m_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        m_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.cmb_mode = NoWheelComboBox()
        self.cmb_mode.addItems(["AUTO", "MANUAL"])
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        m_layout.addRow("Mode:", self.cmb_mode)
        layout.addWidget(mode_box)

    def _build_camera_gimbal_section(self, layout: QVBoxLayout) -> None:
        cam_box = QGroupBox("Camera & Gimbal")
        c_layout = QFormLayout(cam_box)
        c_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        c_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.spn_fps = NoWheelDoubleSpinBox()
        self.spn_fps.setFixedWidth(110)
        self.spn_fps.setRange(30.0, 120.0)
        self.spn_fps.setValue(30.0)
        self.spn_fps.setToolTip("Camera FPS [30..120] Hz (PS §2: min 30)")
        self.spn_fps.valueChanged.connect(self._on_fps_changed)

        self.spn_slew = NoWheelDoubleSpinBox()
        self.spn_slew.setFixedWidth(110)
        self.spn_slew.setRange(0.1, 10.0)
        self.spn_slew.setValue(5.0)
        self.spn_slew.setToolTip("Max slew rate [0.1..10.0] °/s (PS §3)")
        self.spn_slew.valueChanged.connect(self._on_slew_changed)

        c_layout.addRow("Camera FPS:", self.spn_fps)
        c_layout.addRow("Max Slew (°/s):", self.spn_slew)
        self._add_error_label(c_layout, "camera.", "camera_error")
        self._add_error_label(c_layout, "gimbal.", "gimbal_error")
        layout.addWidget(cam_box)

    def _build_target_section(self, layout: QVBoxLayout) -> None:
        tgt_box = QGroupBox("Target Settings")
        t_layout = QFormLayout(tgt_box)
        t_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        t_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.spn_target_count = NoWheelSpinBox()
        self.spn_target_count.setFixedWidth(110)
        self.spn_target_count.setRange(1, 4)
        self.spn_target_count.setValue(1)
        self.spn_target_count.setToolTip("Number of targets [1..4]")
        self.spn_target_count.valueChanged.connect(self._on_target_count_changed)

        self.cmb_tgt_motion = NoWheelComboBox()
        self.cmb_tgt_motion.addItems([k for k in MOTION_KINDS])
        self.cmb_tgt_motion.currentIndexChanged.connect(self._on_target_changed)
        self.spn_tgt_size = NoWheelSpinBox()
        self.spn_tgt_size.setFixedWidth(110)
        self.spn_tgt_size.setRange(5, 20)
        self.spn_tgt_size.setValue(10)
        self.spn_tgt_size.setToolTip("Target size [5..20] px (PS §4)")
        self.spn_tgt_size.valueChanged.connect(self._on_target_changed)
        self.cmb_tgt_shape = NoWheelComboBox()
        self.cmb_tgt_shape.addItems([s for s in SHAPES])
        self.cmb_tgt_shape.currentIndexChanged.connect(self._on_target_changed)
        self.chk_rand_pos = QCheckBox("Random Initial Position")
        self.chk_rand_pos.toggled.connect(self._on_target_changed)

        t_layout.addRow("Count:", self.spn_target_count)
        t_layout.addRow("Motion:", self.cmb_tgt_motion)
        t_layout.addRow("Size (px):", self.spn_tgt_size)
        t_layout.addRow("Shape:", self.cmb_tgt_shape)
        t_layout.addRow(self.chk_rand_pos)

        self._add_error_label(t_layout, "target", "target_error")
        layout.addWidget(tgt_box)

    def _build_motion_section(self, layout: QVBoxLayout) -> None:
        motion_box = QGroupBox("Motion Parameters")
        ml = QVBoxLayout(motion_box)
        self.motion_stack = MotionParamsStack(self)
        self.motion_stack.params_changed.connect(self._on_motion_params_changed)
        ml.addWidget(self.motion_stack)
        self._add_error_label(ml, "motion", "motion_error")
        layout.addWidget(motion_box)

    def _build_disturbance_section(self, layout: QVBoxLayout) -> None:
        dist_box = QGroupBox("Disturbances")
        d_layout = QVBoxLayout(dist_box)

        # Individual disturbance rows: [checkbox][slider][spinbox][unit]
        self.dist_gaussian = DisturbanceMagnitudeRow(
            "gaussian", "sigma_levels",
            label="Gaussian noise", unit="σ levels", minimum=0.0, maximum=20.0,
            default=0.0, decimals=1, step=0.5, slider_scale=10, parent=self,
        )
        self.dist_salt_pepper = DisturbanceMagnitudeRow(
            "salt_pepper", "density",
            label="Salt & pepper", unit="density", minimum=0.0, maximum=1.0,
            default=0.0, decimals=3, step=0.001, slider_scale=1000, parent=self,
        )
        self.dist_poisson = DisturbanceMagnitudeRow(
            "poisson", "photon_scale",
            label="Poisson (photon)", unit="scale", minimum=0.01, maximum=100.0,
            default=1.0, decimals=2, step=0.1, slider_scale=100, parent=self,
        )
        self.dist_jitter = DisturbanceMagnitudeRow(
            "camera_jitter", "max_px_frame",
            label="Camera jitter", unit="px/frame", minimum=0.0, maximum=20.0,
            default=0.0, decimals=1, step=0.5, slider_scale=10, parent=self,
        )
        self.dist_drift = DisturbanceMagnitudeRow(
            "platform", "max_px_frame",
            label="Platform drift", unit="px/frame", minimum=0.0, maximum=20.0,
            default=0.0, decimals=1, step=0.5, slider_scale=10, parent=self,
        )
        self.dist_blur = DisturbanceMagnitudeRow(
            "blur", "sigma_px",
            label="Optical blur", unit="σ px", minimum=0.0, maximum=10.0,
            default=0.0, decimals=1, step=0.1, slider_scale=10, parent=self,
        )

        self._dist_rows = [
            self.dist_gaussian, self.dist_salt_pepper, self.dist_poisson,
            self.dist_jitter, self.dist_drift, self.dist_blur,
        ]

        for row in self._dist_rows:
            row.enabled_changed.connect(self._on_dist_row_enabled)
            row.value_changed.connect(self._on_dist_row_value)
            d_layout.addWidget(row)

        self._add_error_label(d_layout, "disturbances", "disturbances_error")
        layout.addWidget(dist_box)

    def _build_presets_section(self, layout: QVBoxLayout) -> None:
        preset_box = QGroupBox("Presets && Config I/O")
        p_layout = QVBoxLayout(preset_box)

        self.presets_dropdown = PresetsDropdown(self)
        self.presets_dropdown.preset_selected.connect(self._on_preset_selected)
        p_layout.addWidget(self.presets_dropdown)

        # Save / Load buttons
        btn_row = QHBoxLayout()
        self.btn_save_config = QPushButton("Save Config...")
        self.btn_save_config.setFixedHeight(26)
        self.btn_save_config.clicked.connect(self._save_config)
        self.btn_load_config = QPushButton("Load Config...")
        self.btn_load_config.setFixedHeight(26)
        self.btn_load_config.clicked.connect(self._load_config)
        btn_row.addWidget(self.btn_save_config)
        btn_row.addWidget(self.btn_load_config)
        p_layout.addLayout(btn_row)

        layout.addWidget(preset_box)

    def _build_seed_section(self, layout: QVBoxLayout) -> None:
        seed_box = QGroupBox("RNG Seed")
        s_layout = QHBoxLayout(seed_box)
        self.spn_seed = NoWheelSpinBox()
        self.spn_seed.setFixedWidth(100)
        self.spn_seed.setRange(0, 999999)
        self.spn_seed.setValue(42)
        self.spn_seed.setToolTip("Random seed [0..999999]")
        self.spn_seed.valueChanged.connect(self._on_seed_changed)
        self.btn_random_seed = QPushButton("Randomise")
        self.btn_random_seed.setFixedHeight(26)
        self.btn_random_seed.clicked.connect(self._randomise_seed)
        s_layout.addWidget(self.spn_seed)
        s_layout.addWidget(self.btn_random_seed)
        layout.addWidget(seed_box)

    def _add_error_label(
        self, parent_layout: QFormLayout | QVBoxLayout, key_substr: str, tag: str,
    ) -> None:
        lbl = QLabel()
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.STATUS_ERROR.name()}; font-size: 11px; padding: 2px;")
        lbl.hide()
        if isinstance(parent_layout, QFormLayout):
            parent_layout.addRow(lbl)
        else:
            parent_layout.addWidget(lbl)
        self._error_labels[key_substr] = lbl

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sync from config (widget ← config)
    # ------------------------------------------------------------------

    def sync_from_config(self, cfg: Any) -> None:  # noqa: ANN401
        """Update all widgets from the given config (idempotent, no signals emitted)."""
        self._block_signals = True
        try:
            # Camera & Gimbal
            self.spn_fps.setValue(cfg.camera.fps)
            self.spn_slew.setValue(cfg.gimbal.slew_rate_deg_s)

            # Update FPS range based on allow_below_spec_fps
            if cfg.camera.allow_below_spec_fps:
                self.spn_fps.setRange(1.0, 120.0)
            else:
                self.spn_fps.setRange(30.0, 120.0)

            # Control mode
            mode_idx = 0 if cfg.control.mode == "AUTO" else 1
            self.cmb_mode.setCurrentIndex(mode_idx)

            # Input source
            if cfg.input.kind == "mp4":
                self.cmb_input.setCurrentText("MP4 Video")
                self.btn_browse_mp4.show()
                self.mp4_section.show()
                self.mp4_section.sync(
                    cfg.input.mp4_path, cfg.input.loop, cfg.input.fps_override
                )
            elif cfg.input.kind in (InputKind.ORBITAL, "orbital"):
                self.cmb_input.setCurrentText("Orbital (S-1 <-> S-2)")
                self.btn_browse_mp4.hide()
                self.mp4_section.hide()
            else:
                self.cmb_input.setCurrentText("Simulation")
                self.btn_browse_mp4.hide()
                self.mp4_section.hide()

            # Target
            self.spn_target_count.setValue(cfg.target.count)
            if cfg.target.targets:
                t0 = cfg.target.targets[0]
                self.spn_tgt_size.setValue(t0.size_px)

                # Shape
                shape_idx = self.cmb_tgt_shape.findText(t0.shape)
                if shape_idx >= 0:
                    self.cmb_tgt_shape.setCurrentIndex(shape_idx)

                # Motion
                motion_kind = t0.motion.kind
                motion_idx = self.cmb_tgt_motion.findText(motion_kind)
                if motion_idx >= 0:
                    self.cmb_tgt_motion.setCurrentIndex(motion_idx)

                # Motion parameters
                import dataclasses
                motion_params = dataclasses.asdict(t0.motion)
                motion_params.pop("kind", None)
                self.motion_stack.sync(motion_kind, motion_params)

                # Initial mode
                self.chk_rand_pos.setChecked(t0.initial == "random")

            # Disturbance magnitudes
            self.dist_gaussian.set_enabled_checked(cfg.disturbances.gaussian.enabled)
            self.dist_gaussian.set_value(cfg.disturbances.gaussian.sigma_levels)
            self.dist_salt_pepper.set_enabled_checked(cfg.disturbances.salt_pepper.enabled)
            self.dist_salt_pepper.set_value(cfg.disturbances.salt_pepper.density)
            self.dist_poisson.set_enabled_checked(cfg.disturbances.poisson.enabled)
            self.dist_poisson.set_value(cfg.disturbances.poisson.photon_scale)
            self.dist_jitter.set_enabled_checked(cfg.disturbances.camera_jitter.enabled)
            self.dist_jitter.set_value(cfg.disturbances.camera_jitter.max_px_frame)
            self.dist_drift.set_enabled_checked(cfg.disturbances.platform.enabled)
            self.dist_drift.set_value(cfg.disturbances.platform.max_px_frame)
            self.dist_blur.set_enabled_checked(cfg.disturbances.blur.enabled)
            self.dist_blur.set_value(cfg.disturbances.blur.sigma_px)

            # Seed
            self.spn_seed.setValue(cfg.seed)

            # Preset dropdown reset
            self.presets_dropdown.reset_selection()

        finally:
            self._block_signals = False

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        """Handle Start button: only starts if configuration is valid."""
        if not self.btn_start.isEnabled():
            return
        self.start_clicked.emit()

    def _on_input_changed(self) -> None:
        if self._block_signals:
            return
        txt = self.cmb_input.currentText()
        is_mp4 = txt == "MP4 Video"
        self.btn_browse_mp4.setVisible(is_mp4)
        self.mp4_section.setVisible(is_mp4)
        if is_mp4:
            kind = InputKind.MP4.value
        elif txt == "Orbital (S-1 <-> S-2)":
            kind = InputKind.ORBITAL.value
        else:
            kind = InputKind.SIMULATION.value
        self._mp4_probe_ok = False
        self._apply_dict({"input.kind": kind})
        self._update_start_enabled()

    def _browse_mp4(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)",
        )
        if path:
            self.mp4_section.set_path(path)
            self._probe_mp4(path)
            self._apply_dict({"input.mp4_path": path})

    def _probe_mp4(self, path: str) -> None:
        """Probe the MP4 file without importing cv2 in this module."""
        from skylock.app.video_probe import probe_video

        info = probe_video(path)
        self._mp4_probe_ok = info.ok
        self.mp4_section.show_probe_result(
            ok=info.ok,
            width=info.width,
            height=info.height,
            fps=info.fps,
            frame_count=info.frame_count,
            error=info.error,
        )
        # If fps unreadable, set an fps_override automatically
        if info.ok and info.fps is None:
            self._apply_dict({"input.fps_override": self.mp4_section.spn_fps_override.value()})
        self._update_start_enabled()

    def _on_mp4_fps_override(self, val: object) -> None:
        if self._block_signals:
            return
        self._apply_dict({"input.fps_override": float(val)})  # type: ignore[arg-type]

    def _on_mp4_loop(self, checked: bool) -> None:
        if self._block_signals:
            return
        self._apply_dict({"input.loop": checked})

    def _on_mode_changed(self) -> None:
        """Handle mode combo change - emit for live mode change, don't rebuild."""
        if self._block_signals:
            return
        mode_text = self.cmb_mode.currentText()
        self.mode_changed.emit(mode_text)

    def _on_fps_changed(self) -> None:
        if self._block_signals:
            return
        self._apply_dict({"camera.fps": self.spn_fps.value()})

    def _on_slew_changed(self) -> None:
        if self._block_signals:
            return
        overrides = slew_override(self.editor.config, self.spn_slew.value())
        self._apply_dict(overrides)

    def _on_target_count_changed(self) -> None:
        if self._block_signals:
            return
        n = self.spn_target_count.value()
        overrides = target_count_override(self.editor.config, n)
        self._apply_dict(overrides)

    def _on_target_changed(self) -> None:
        if self._block_signals:
            return

        motion_kind = self.cmb_tgt_motion.currentText()
        shape = self.cmb_tgt_shape.currentText()
        size_px = self.spn_tgt_size.value()
        initial = "random" if self.chk_rand_pos.isChecked() else "fixed"

        # Sync motion stack page
        self.motion_stack.set_kind(motion_kind)

        overrides = target_override(
            self.editor.config,
            size_px=size_px,
            shape=shape,
            motion_kind=motion_kind,
            initial=initial,
        )
        self._apply_dict(overrides)

    def _on_motion_params_changed(self, kind: str, params: dict[str, float]) -> None:
        """Update target motion parameters from the motion stack."""
        if self._block_signals:
            return
        # Build a motion dict with kind + params
        from skylock.config.io import to_dict

        cfg_dict = to_dict(self.editor.config)
        current_target = cfg_dict["target"]["targets"][0].copy()
        motion = {"kind": kind}
        motion.update(params)

        # For random motion, preserve bounds_deg from current config
        if kind == "random":
            current_motion = current_target.get("motion", {})
            if "bounds_deg" in current_motion:
                motion["bounds_deg"] = current_motion["bounds_deg"]

        current_target["motion"] = motion
        self._apply_dict({"target.targets": [current_target]})

    def _on_dist_row_enabled(self, dist_name: str, enabled: bool) -> None:
        """Handle individual disturbance enable/disable."""
        if self._block_signals:
            return
        overrides = disturbance_toggle_override(self.editor.config, dist_name, enabled)
        self._apply_dict(overrides)

    def _on_dist_row_value(self, dist_name: str, field_key: str, val: float) -> None:
        """Handle individual disturbance magnitude change."""
        if self._block_signals:
            return
        self._apply_dict({f"disturbances.{dist_name}.{field_key}": val})

    def _on_preset_selected(self, fn_name: str) -> None:
        """Apply a disturbance preset."""
        if self._block_signals:
            return

        from skylock.config import presets
        from skylock.config.io import to_dict

        preset_fn = getattr(presets, fn_name, None)
        if preset_fn is None:
            return

        dist_cfg = preset_fn()
        dist_dict = to_dict(
            type(self.editor.config)(disturbances=dist_cfg)
        )["disturbances"]

        # Build a single override dict with all disturbance keys
        overrides: dict[str, Any] = {}
        for group_name, group_vals in dist_dict.items():
            if isinstance(group_vals, dict):
                for k, v in group_vals.items():
                    overrides[f"disturbances.{group_name}.{k}"] = v

        self._apply_dict(overrides)
        # Sync widgets back after applying preset
        self.sync_from_config(self.editor.config)

    def _save_config(self) -> None:
        """Save current config to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)",
        )
        if path:
            from skylock.config.io import to_json

            try:
                json_str = to_json(self.editor.config)
                Path(path).write_text(json_str, encoding="utf-8")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _load_config(self) -> None:
        """Load config from a JSON file, replacing the editor's config."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)",
        )
        if path:
            try:
                from skylock.config.io import from_json

                text = Path(path).read_text(encoding="utf-8")
                new_cfg = from_json(text)
                self.editor.reset_to(new_cfg)
                self.sync_from_config(new_cfg)
                self._clear_all_errors()
                self.config_changed.emit(new_cfg)
            except ConfigError as e:
                QMessageBox.critical(
                    self, "Configuration Error",
                    "Configuration validation failed:\n" + "\n".join(e.violations),
                )
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _on_seed_changed(self) -> None:
        if self._block_signals:
            return
        self._apply_dict({"seed": self.spn_seed.value()})

    def _randomise_seed(self) -> None:
        new_seed = random.randint(1, 999999)
        self.spn_seed.setValue(new_seed)

    # ------------------------------------------------------------------
    # Override Application & Error Display
    # ------------------------------------------------------------------

    def _apply_dict(self, overrides: dict[str, Any]) -> None:
        """Apply overrides through ConfigEditor, updating inline errors."""
        new_cfg, violations = self.editor.apply_overrides(overrides)
        if violations:
            self._show_violations(violations)
            self._update_start_enabled()
        else:
            self._clear_all_errors()
            if new_cfg is not None:
                self.config_changed.emit(new_cfg)
            self._update_start_enabled()

    def _show_violations(self, violations: list[str]) -> None:
        """Route violations to per-section inline labels and fallback summary."""
        # Reset all inline labels
        for lbl in self._error_labels.values():
            lbl.hide()
            lbl.setText("")

        unmatched: list[str] = []
        for v in violations:
            matched = False
            for key_substr, lbl in self._error_labels.items():
                if key_substr in v:
                    current = lbl.text()
                    lbl.setText(f"{current}\n{v}" if current else v)
                    lbl.show()
                    matched = True
                    break
            if not matched:
                unmatched.append(v)

        if unmatched:
            self.lbl_error.setText("Configuration Error:\n" + "\n".join(unmatched))
            self.lbl_error.show()
        else:
            self.lbl_error.hide()

    def _clear_all_errors(self) -> None:
        for lbl in self._error_labels.values():
            lbl.hide()
            lbl.setText("")
        self.lbl_error.hide()

    def _has_violations(self) -> bool:
        """Check if any error label is currently visible."""
        if self.lbl_error.isVisible():
            return True
        return any(lbl.isVisible() for lbl in self._error_labels.values())

    def _update_start_enabled(self) -> None:
        """Start = no violations AND (simulation OR probe ok)."""
        has_errors = self._has_violations()
        is_mp4 = self.cmb_input.currentText() == "MP4 Video"
        can_start = not has_errors and (not is_mp4 or self._mp4_probe_ok)
        self.btn_start.setEnabled(can_start)


__all__ = ("ControlsPanel",)
