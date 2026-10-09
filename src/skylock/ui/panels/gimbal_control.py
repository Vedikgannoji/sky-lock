"""Dedicated functional Gimbal & Camera Control panel located below Telemetry."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from skylock.ui.widgets.no_wheel import NoWheelDoubleSpinBox


class GimbalControlPanel(QWidget):
    """Interactive control module for steering the S-1 gimbal and camera FOV."""

    pan_changed = Signal(float)  # deg
    tilt_changed = Signal(float)  # deg
    fov_changed = Signal(float)  # deg
    reset_clicked = Signal()
    track_target_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._block_signals = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.setMinimumWidth(0)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        group = QGroupBox("Gimbal Control")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        # Pan control (-180° to +180°)
        self.spn_pan = NoWheelDoubleSpinBox()
        self.spn_pan.setRange(-180.0, 180.0)
        self.spn_pan.setDecimals(2)
        self.spn_pan.setSingleStep(1.0)
        self.spn_pan.setSuffix("°")
        self.spn_pan.setValue(0.0)
        self.spn_pan.setToolTip("Gimbal azimuth angle (-180° to +180°)")
        self.spn_pan.valueChanged.connect(self._on_pan_changed)
        form.addRow("Pan:", self.spn_pan)

        # Tilt control (-90° to +90°)
        self.spn_tilt = NoWheelDoubleSpinBox()
        self.spn_tilt.setRange(-90.0, 90.0)
        self.spn_tilt.setDecimals(2)
        self.spn_tilt.setSingleStep(1.0)
        self.spn_tilt.setSuffix("°")
        self.spn_tilt.setValue(0.0)
        self.spn_tilt.setToolTip("Gimbal elevation angle (-90° to +90°)")
        self.spn_tilt.valueChanged.connect(self._on_tilt_changed)
        form.addRow("Tilt:", self.spn_tilt)

        # FOV control (horizontal FOV, vertical = 0.75 x; default 16 x 12)
        self.lbl_fov = QLabel("FOV (16.0° × 12.0°):")
        self.spn_fov = NoWheelDoubleSpinBox()
        self.spn_fov.setRange(2.0, 90.0)
        self.spn_fov.setDecimals(1)
        self.spn_fov.setSingleStep(1.0)
        self.spn_fov.setSuffix("°")
        self.spn_fov.setValue(16.0)
        self.spn_fov.setToolTip("Camera horizontal field-of-view (vertical is 0.75× at 4:3)")
        self.spn_fov.valueChanged.connect(self._on_fov_changed)
        form.addRow(self.lbl_fov, self.spn_fov)

        layout.addLayout(form)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 2, 0, 0)
        btn_layout.setSpacing(6)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset gimbal to neutral pose (0°, 0°) and default FOV (16° × 12°)")
        self.btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.btn_reset)

        self.btn_track = QPushButton("Track Target")
        self.btn_track.setToolTip("Point gimbal toward target and engage optical tracking")
        self.btn_track.setStyleSheet("QPushButton { font-weight: bold; }")
        self.btn_track.clicked.connect(self._on_track_target)
        btn_layout.addWidget(self.btn_track)

        layout.addLayout(btn_layout)
        main_layout.addWidget(group)

    def _update_fov_label(self, fov_h: float) -> None:
        fov_v = 0.75 * fov_h
        self.lbl_fov.setText(f"FOV ({fov_h:.1f}° × {fov_v:.1f}°):")

    def _on_pan_changed(self, val: float) -> None:
        if not self._block_signals:
            self.pan_changed.emit(val)

    def _on_tilt_changed(self, val: float) -> None:
        if not self._block_signals:
            self.tilt_changed.emit(val)

    def _on_fov_changed(self, val: float) -> None:
        self._update_fov_label(val)
        if not self._block_signals:
            self.fov_changed.emit(val)

    def _on_reset(self) -> None:
        self.set_values(0.0, 0.0, 16.0, emit_signals=True)
        self.reset_clicked.emit()

    def _on_track_target(self) -> None:
        self.track_target_clicked.emit()

    def set_values(
        self, pan: float, tilt: float, fov: float | None = None, emit_signals: bool = False
    ) -> None:
        """Set gimbal UI values programmatically."""
        old_block = self._block_signals
        self._block_signals = not emit_signals
        try:
            self.spn_pan.setValue(float(pan))
            self.spn_tilt.setValue(float(tilt))
            if fov is not None:
                self.spn_fov.setValue(float(fov))
                self._update_fov_label(float(fov))
        finally:
            self._block_signals = old_block
