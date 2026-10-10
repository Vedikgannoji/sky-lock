"""Benchmark execution panel - Phase 1 UI cleanup.

Removed from UI (not from core):
  - MP4 file browser / path display (S16 still works via _mp4_path internal state)
  - Status column from table
  - Frames column from table
  - Wall Time column from table
  - Long horizontal button row replaced with 2-row compact layout
  - Secondary actions (Details, Export, Load) moved to a More menu button
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from skylock.benchmark.catalog import builtin_scenarios
from skylock.benchmark.report import to_json_report, to_markdown_report
from skylock.benchmark.runner import BenchmarkRunner, RunRecord
from skylock.benchmark.scenario import Scenario
from skylock.config.models import SkyLockConfig
from skylock.ui import theme
from skylock.ui.panels.benchmark_detail import BenchmarkDetailDialog
from skylock.ui.widgets.no_wheel import NoWheelComboBox, NoWheelSpinBox

_EM_DASH = "\u2014"


def _get_verdict_color(verdict: str) -> QColor:
    """Return background color corresponding to a verdict string."""
    if verdict == "PASS":
        return theme.VERDICT_PASS_BG
    if verdict == "FAIL":
        return theme.VERDICT_FAIL_BG
    if verdict == "INDETERMINATE":
        return theme.VERDICT_INDETERMINATE_BG
    if verdict == "NOT_RUN":
        return theme.VERDICT_NOT_RUN_BG
    return theme.VERDICT_UNKNOWN_BG


class _BenchWorker(QObject):
    """Worker running benchmark scenarios sequentially off the main GUI thread."""

    record_ready = Signal(object)
    progress = Signal(int, int)  # completed, total
    finished = Signal()

    def __init__(
        self,
        tasks: list[tuple[Scenario, int]],
        runner: BenchmarkRunner,
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.runner = runner
        self._is_cancelled = False

    @Slot()
    def run(self) -> None:
        total = len(self.tasks)
        for i, (sc, seed) in enumerate(self.tasks):
            if self._is_cancelled:
                break
            try:
                record = self.runner.run(sc, seed=seed)
                self.record_ready.emit(record)
            except Exception as e:
                import platform
                import sys
                import cv2
                import numpy as np
                import skylock
                rec = RunRecord(
                    run_id="error",
                    scenario_id=sc.id,
                    seed=seed,
                    software_version=skylock.__version__,
                    python_version=sys.version,
                    numpy_version=np.__version__,
                    opencv_version=cv2.__version__,
                    platform_info=platform.platform(),
                    config_snapshot={},
                    config_hash="",
                    input_source=sc.input_kind,
                    duration_s=sc.duration_s,
                    frames=0,
                    metrics={},
                    verdicts={"overall": "FAIL"},
                    overall_verdict="FAIL",
                    started_at_utc="",
                    wall_time_s=0.0,
                    status="FAILED",
                    error=str(e),
                )
                self.record_ready.emit(rec)
            self.progress.emit(i + 1, total)
        self.finished.emit()

    def cancel(self) -> None:
        self._is_cancelled = True


class BenchmarkPanel(QWidget):
    """Bottom tab / dock panel for running test scenarios and displaying results."""

    def __init__(
        self,
        runner: BenchmarkRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.runner = runner if runner is not None else BenchmarkRunner(SkyLockConfig())
        self._records: list[RunRecord] = []
        self._worker_thread: QThread | None = None
        self._worker: _BenchWorker | None = None
        # Internal state for S16 mp4 path (not exposed in UI per Phase 1 rules)
        self._mp4_path: str | None = None

        self._build_ui()
        self._populate_scenarios()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # ── Row 1: Scenario and Seed selection ──────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        row1.addWidget(QLabel("Scenario:"))
        self.cmb_scenarios = NoWheelComboBox()
        self.cmb_scenarios.setMinimumWidth(150)
        self.cmb_scenarios.setMaximumWidth(220)
        row1.addWidget(self.cmb_scenarios)

        row1.addWidget(QLabel("Seed:"))
        self.spn_seed = NoWheelSpinBox()
        self.spn_seed.setRange(0, 999999)
        self.spn_seed.setValue(42)
        self.spn_seed.setReadOnly(True)
        self.spn_seed.setMaximumWidth(70)
        row1.addWidget(self.spn_seed)

        self.chk_link_seed = QCheckBox("Link seed")
        self.chk_link_seed.setChecked(True)
        self.chk_link_seed.setToolTip("Mirror seed from the Controls panel")
        self.chk_link_seed.toggled.connect(self._on_toggle_link_seed)
        row1.addWidget(self.chk_link_seed)

        row1.addWidget(QLabel("Seeds:"))
        self.txt_seeds = QLineEdit()
        self.txt_seeds.setPlaceholderText("e.g. 42,100,2024")
        self.txt_seeds.setToolTip("Comma-separated integers; leave blank to use single seed")
        self.txt_seeds.setMaximumWidth(120)
        row1.addWidget(self.txt_seeds)

        row1.addStretch()
        layout.addLayout(row1)

        # ── Row 2: Action buttons + status & summary ───────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self.btn_run = QPushButton("Run Scenario")
        self.btn_run.setToolTip("Run selected scenario")
        self.btn_run.clicked.connect(self._run_selected)
        row2.addWidget(self.btn_run)

        self.btn_run_all = QPushButton("Run All")
        self.btn_run_all.setToolTip("Run all scenarios")
        self.btn_run_all.clicked.connect(self._run_all)
        row2.addWidget(self.btn_run_all)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_benchmark)
        row2.addWidget(self.btn_cancel)

        # More Actions menu (secondary actions)
        self.btn_more = QPushButton("More Actions \u25be")
        self.btn_more.setToolTip("Secondary actions: Details, Export, Load Report")
        self._more_menu = QMenu(self)
        act_details = self._more_menu.addAction("Details")
        act_details.triggered.connect(self._open_selected_details)
        self._more_menu.addSeparator()
        act_export_json = self._more_menu.addAction("Export JSON")
        act_export_json.triggered.connect(self._export_json)
        act_export_md = self._more_menu.addAction("Export Markdown")
        act_export_md.triggered.connect(self._export_md)
        self._more_menu.addSeparator()
        act_load = self._more_menu.addAction("Load Report")
        act_load.triggered.connect(self._load_report)
        self.btn_more.setMenu(self._more_menu)
        row2.addWidget(self.btn_more)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMaximumWidth(120)
        row2.addWidget(self.progress_bar)

        row2.addSpacing(10)
        self.lbl_summary = QLabel("0 runs | PASS: 0 | FAIL: 0 | INDET: 0")
        self.lbl_summary.setStyleSheet(
            f"font-weight: bold; color: {theme.TEXT_PRIMARY.name()};"
        )
        row2.addWidget(self.lbl_summary)

        row2.addStretch()

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY.name()}; font-style: italic;"
        )
        row2.addWidget(self.lbl_status)

        layout.addLayout(row2)

        # ── Results table (cleaned columns) ──────────────────────────────
        self.table = QTableWidget()
        # Primary columns only (Rules 5 & 6):
        #   0=Scenario, 1=Seed, 2=Verdict, 3=Acq(s), 4=Track RMS(px),
        #   5=Loss Rate, 6=Reacq(s), 7=Proc FPS
        cols = [
            "Scenario",
            "Seed",
            "Verdict",
            "Acquisition (s)",
            "Tracking Error (px)",
            "Loss Rate",
            "Reacquisition (s)",
            "FPS",
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSortingEnabled(True)
        self.table.itemDoubleClicked.connect(self._on_table_double_clicked)
        layout.addWidget(self.table)

    def _populate_scenarios(self) -> None:
        self.scenarios_list = list(builtin_scenarios())
        for sc in self.scenarios_list:
            self.cmb_scenarios.addItem(sc.id, sc)
        self.cmb_scenarios.currentIndexChanged.connect(self._on_scenario_changed)
        if self.scenarios_list:
            self._on_scenario_changed(0)

    def _on_scenario_changed(self, index: int) -> None:
        sc = self.cmb_scenarios.currentData()
        if sc is not None:
            self.cmb_scenarios.setToolTip(
                f"{sc.id}\nDuration: {sc.duration_s}s\nTags: {', '.join(sc.tags)}\n\n{sc.description}"
            )
            if self._worker_thread is None or not self._worker_thread.isRunning():
                self.lbl_status.setText(f"Ready: {sc.id} ({sc.duration_s:.1f}s)")

    def _on_toggle_link_seed(self, checked: bool) -> None:
        self.spn_seed.setReadOnly(checked)

    def sync_controls_seed(self, seed: int) -> None:
        """Called when controls seed changes to mirror it if linked."""
        if self.chk_link_seed.isChecked():
            self.spn_seed.setValue(seed)

    def _parse_seeds(self) -> list[int] | None:
        raw = self.txt_seeds.text().strip()
        if not raw:
            return [self.spn_seed.value()]
        seeds: list[int] = []
        for part in raw.split(","):
            part_str = part.strip()
            if not part_str:
                continue
            try:
                val = int(part_str)
                seeds.append(val)
            except ValueError:
                self.lbl_status.setText(f"Error: Invalid seed '{part_str}'")
                return None
        return seeds if seeds else [self.spn_seed.value()]

    def _run_selected(self) -> None:
        seeds = self._parse_seeds()
        if seeds is None:
            return
        sc = self.cmb_scenarios.currentData()
        if sc:
            self._start_benchmark([sc], seeds)

    def _run_all(self) -> None:
        seeds = self._parse_seeds()
        if seeds is None:
            return
        self._start_benchmark(self.scenarios_list, seeds)

    def _start_benchmark(self, scenarios: list[Scenario], seeds: list[int]) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        self._was_cancelled = False
        tasks: list[tuple[Scenario, int]] = []
        skipped_s16 = False

        for sc in scenarios:
            for s in seeds:
                if sc.input_kind == "mp4" or sc.id == "S16_mp4":
                    if not self._mp4_path:
                        skipped_s16 = True
                        continue
                    sc_with_mp4 = dataclasses.replace(sc, mp4_path=self._mp4_path)
                    tasks.append((sc_with_mp4, s))
                else:
                    tasks.append((sc, s))

        if not tasks:
            if skipped_s16:
                self.lbl_status.setText("S16_mp4 skipped: no MP4 selected")
            return

        if skipped_s16:
            self.lbl_status.setText("S16_mp4 skipped: no MP4 selected")

        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(tasks))

        self._worker_thread = QThread()
        self._worker = _BenchWorker(tasks, self.runner)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.record_ready.connect(self.add_record)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _cancel_benchmark(self) -> None:
        if self._worker is not None:
            self._was_cancelled = True
            self._worker.cancel()
            self.lbl_status.setText("Cancelling after current run...")
            self.btn_cancel.setEnabled(False)

    @Slot(int, int)
    def _on_progress(self, completed: int, total: int) -> None:
        self.progress_bar.setValue(completed)
        self.lbl_status.setText(f"Running task {completed} / {total}...")

    def _on_finished(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run_all.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        if getattr(self, "_was_cancelled", False):
            self.lbl_status.setText(f"Cancelled after {len(self._records)} run(s)")
        else:
            self.lbl_status.setText(f"Completed {len(self._records)} benchmark run(s)")
        self._update_summary_label()

        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
            self._worker_thread = None

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Cancel and safely wait for benchmark worker thread to terminate."""
        if self._worker is not None:
            self._worker.cancel()
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            if not self._worker_thread.wait(5000):
                self._worker_thread.terminate()
                self._worker_thread.wait(1000)
            self._worker_thread.deleteLater()
            self._worker_thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    @Slot(object)
    def add_record(self, record: RunRecord) -> None:
        """Add a RunRecord to the table with verdict color formatting."""
        self._records.append(record)
        self._insert_record_row(record)
        self._update_summary_label()

    def _insert_record_row(self, record: RunRecord) -> None:
        self.table.setSortingEnabled(False)
        row = self.table.rowCount()
        self.table.insertRow(row)

        metrics = record.metrics or {}

        # 1. Acquisition time
        acq_entry = metrics.get("acquisition_time_from_observable_s", {})
        acq_str = _EM_DASH
        acq_tooltip = ""
        if isinstance(acq_entry, dict):
            if acq_entry.get("status") == "MEASURED" and acq_entry.get("value") is not None:
                acq_str = f"{acq_entry['value']:.3f}"
            else:
                acq_tooltip = acq_entry.get("reason", acq_entry.get("status", ""))

        # 2. Tracking error RMS
        trk_entry = metrics.get("tracking_error_px", {})
        trk_str = _EM_DASH
        trk_tooltip = ""
        if isinstance(trk_entry, dict):
            val = trk_entry.get("value")
            if trk_entry.get("status") == "MEASURED" and isinstance(val, dict) and "rms" in val:
                trk_str = f"{val['rms']:.2f}"
            else:
                trk_tooltip = trk_entry.get("reason", trk_entry.get("status", ""))

        # 3. Target loss rate
        loss_entry = metrics.get("target_loss_rate", {})
        loss_str = _EM_DASH
        loss_tooltip = ""
        if isinstance(loss_entry, dict):
            if loss_entry.get("status") == "MEASURED" and loss_entry.get("value") is not None:
                loss_str = f"{loss_entry['value']:.4f}"
            else:
                loss_tooltip = loss_entry.get("reason", loss_entry.get("status", ""))

        # 4. Reacquisition time
        reacq_entry = metrics.get("reacquisition_time_s", {})
        reacq_str = _EM_DASH
        reacq_tooltip = ""
        if isinstance(reacq_entry, dict):
            val = reacq_entry.get("value")
            if reacq_entry.get("status") == "MEASURED" and isinstance(val, dict):
                r_val = val.get("mean_s", val.get("max_s"))
                if r_val is not None:
                    reacq_str = f"{r_val:.3f}"
            else:
                reacq_tooltip = reacq_entry.get("reason", reacq_entry.get("status", ""))

        # 5. Processing FPS
        fps_entry = metrics.get("fps_pipeline", {})
        fps_str = _EM_DASH
        fps_tooltip = ""
        if isinstance(fps_entry, dict):
            if fps_entry.get("status") == "MEASURED" and fps_entry.get("value") is not None:
                fps_str = f"{fps_entry['value']:.1f}"
            else:
                fps_tooltip = fps_entry.get("reason", fps_entry.get("status", ""))

        # Table columns: Scenario, Seed, Verdict, Acq, TrkRMS, Loss, Reacq, FPS
        items = [
            (record.scenario_id, ""),
            (str(record.seed), ""),
            (record.overall_verdict, ""),
            (acq_str, acq_tooltip),
            (trk_str, trk_tooltip),
            (loss_str, loss_tooltip),
            (reacq_str, reacq_tooltip),
            (fps_str, fps_tooltip),
        ]

        color = _get_verdict_color(record.overall_verdict)
        for col_idx, (text, tooltip) in enumerate(items):
            item = QTableWidgetItem(text)
            if tooltip:
                item.setToolTip(tooltip)
            if col_idx == 2:  # Verdict column
                item.setBackground(color)
                item.setForeground(theme.VERDICT_TEXT)
                item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col_idx, item)

        self.table.setSortingEnabled(True)

    def _update_summary_label(self) -> None:
        counts = {"PASS": 0, "FAIL": 0, "INDETERMINATE": 0, "NOT_RUN": 0}
        for r in self._records:
            v = r.overall_verdict
            counts[v] = counts.get(v, 0) + 1
        not_run_str = f" | NOT_RUN: {counts['NOT_RUN']}" if counts.get("NOT_RUN", 0) > 0 else ""
        self.lbl_summary.setText(
            f"{len(self._records)} runs | "
            f"PASS: {counts.get('PASS', 0)} | "
            f"FAIL: {counts.get('FAIL', 0)} | "
            f"INDET: {counts.get('INDETERMINATE', 0)}"
            f"{not_run_str}"
        )

    def _on_table_double_clicked(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if 0 <= row < len(self._records):
            dlg = BenchmarkDetailDialog(self._records[row], self)
            dlg.exec()

    def _open_selected_details(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._records):
            dlg = BenchmarkDetailDialog(self._records[row], self)
            dlg.exec()
        else:
            QMessageBox.information(self, "Details", "Please select a run in the table.")

    def _load_report(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Benchmark JSON Report", "", "JSON (*.json);;All Files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") != "1.0" or "runs" not in data:
                raise ValueError("Incompatible report format or missing schema_version 1.0")

            records: list[RunRecord] = []
            for r_dict in data["runs"]:
                rec = RunRecord(
                    run_id=r_dict.get("run_id", ""),
                    scenario_id=r_dict.get("scenario_id", ""),
                    seed=int(r_dict.get("seed", 0)),
                    software_version=r_dict.get("software_version", ""),
                    python_version=r_dict.get("python_version", ""),
                    numpy_version=r_dict.get("numpy_version", ""),
                    opencv_version=r_dict.get("opencv_version", ""),
                    platform_info=r_dict.get("platform_info", ""),
                    config_snapshot=r_dict.get("config_snapshot", {}),
                    config_hash=r_dict.get("config_hash", ""),
                    input_source=r_dict.get("input_source", ""),
                    duration_s=float(r_dict.get("duration_s", 0.0)),
                    frames=int(r_dict.get("frames", 0)),
                    metrics=r_dict.get("metrics", {}),
                    verdicts=r_dict.get("verdicts", {}),
                    overall_verdict=r_dict.get("overall_verdict", "INDETERMINATE"),
                    started_at_utc=r_dict.get("started_at_utc", ""),
                    wall_time_s=float(r_dict.get("wall_time_s", 0.0)),
                    git_commit=r_dict.get("git_commit"),
                    status=r_dict.get("status", "COMPLETED"),
                    error=r_dict.get("error"),
                    state_timeline_hash=r_dict.get("state_timeline_hash"),
                    estimates_hash=r_dict.get("estimates_hash"),
                )
                records.append(rec)

            self._records = records
            self.table.setRowCount(0)
            for r in records:
                self._insert_record_row(r)
            self._update_summary_label()
            self.lbl_status.setText(f"Loaded {len(records)} run(s) from {Path(path).name}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load report: {e}")

    def _export_json(self) -> None:
        if not self._records:
            QMessageBox.information(self, "Export", "No benchmark records to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Report", "report.json", "JSON (*.json)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(to_json_report(self._records))
                self.lbl_status.setText(f"Exported JSON to {Path(path).name}")
            except OSError as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to write JSON report: {e}")

    def _export_md(self) -> None:
        if not self._records:
            QMessageBox.information(self, "Export", "No benchmark records to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Markdown Report", "report.md", "Markdown (*.md)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(to_markdown_report(self._records))
                self.lbl_status.setText(f"Exported Markdown to {Path(path).name}")
            except OSError as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to write Markdown report: {e}")


__all__ = ("BenchmarkPanel",)
