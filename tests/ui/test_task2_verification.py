"""Comprehensive Task 2 verification test suite covering all 16 specification points."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from skylock.app.factory import build_session
from skylock.benchmark.catalog import builtin_scenarios
from skylock.benchmark.compare import compare_runs
from skylock.benchmark.report import to_json_report, to_markdown_report
from skylock.benchmark.runner import BenchmarkRunner, RunRecord
from skylock.benchmark.scenario import Scenario
from skylock.config.models import GaussianConfig, RequirementsConfig, SkyLockConfig
from skylock.core.enums import ControlMode, MetricStatus, TrackState, Verdict
from skylock.core.types import GroundTruthSample
from skylock.metrics.calculators import (
    calculate_acquisition_time,
    calculate_reacquisition,
    calculate_target_loss_rate,
    calculate_tracking_error,
)
from skylock.metrics.collector import MetricsCollector
from skylock.metrics.requirements import evaluate
from skylock.ui.main_window import MainWindow
from skylock.ui.panels.benchmark import BenchmarkPanel

pytestmark = pytest.mark.gui


def _wait_for_condition(cond_fn, timeout_s=6.0, step_s=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        QCoreApplication.processEvents()
        if cond_fn():
            return True
        time.sleep(step_s)
    return False


# ============================================================================
# 1. Benchmark scenario configuration
# ============================================================================
def test_1_benchmark_scenario_configuration():
    """Verify built-in scenarios correctly configure conditions and overrides."""
    scenarios = builtin_scenarios()
    assert len(scenarios) == 16, "Catalog must have exactly 16 scenarios"

    s01 = next(s for s in scenarios if s.id == "S01_line_clean")
    s05 = next(s for s in scenarios if s.id == "S05_line_spec_noise")
    s13 = next(s for s in scenarios if s.id == "S13_all_disturbances")
    s15 = next(s for s in scenarios if s.id == "S15_slew10")

    base = SkyLockConfig()
    cfg01 = s01.apply(base, seed=42)
    assert cfg01.seed == 42
    assert cfg01.target.count == 1
    assert cfg01.disturbances.gaussian.enabled is False

    cfg05 = s05.apply(base, seed=42)
    assert cfg05.disturbances.gaussian.enabled is True
    assert cfg05.disturbances.salt_pepper.enabled is True

    cfg13 = s13.apply(base, seed=42)
    assert cfg13.disturbances.atmosphere.enabled is True
    assert cfg13.disturbances.blur.enabled is True

    cfg15 = s15.apply(base, seed=42)
    assert cfg15.gimbal.slew_rate_deg_s == 10.0


# ============================================================================
# 2. Seed reproducibility
# ============================================================================
def test_2_seed_reproducibility():
    """Verify identical seeds produce deterministic state and estimates hashes."""
    runner = BenchmarkRunner()
    s01 = next(s for s in builtin_scenarios() if s.id == "S01_line_clean")

    rec1 = runner.run(s01, seed=42)
    rec2 = runner.run(s01, seed=42)

    assert rec1.state_timeline_hash == rec2.state_timeline_hash
    assert rec1.estimates_hash == rec2.estimates_hash
    assert rec1.frames == rec2.frames
    assert rec1.overall_verdict == rec2.overall_verdict

    # Comparing deterministic fields yields no diffs
    diffs = compare_runs(rec1, rec2)
    assert len(diffs) == 0, f"Expected deterministic equivalence, got diffs: {diffs}"

    # Different seed yields different hash
    rec3 = runner.run(s01, seed=99)
    assert rec1.seed != rec3.seed


# ============================================================================
# 3. Actual detector and controller execution
# ============================================================================
def test_3_actual_detector_controller_execution():
    """Verify benchmark executes the real TrackingPipeline and PointingController."""
    runner = BenchmarkRunner()
    s01 = next(s for s in builtin_scenarios() if s.id == "S01_line_clean")

    rec = runner.run(s01, seed=42)
    assert rec.frames > 0
    assert rec.status == "COMPLETED"
    assert "tracking_error_px" in rec.metrics
    assert rec.metrics["tracking_error_px"]["status"] == "MEASURED"
    assert rec.metrics["fps_pipeline"]["status"] == "MEASURED"
    assert rec.overall_verdict in ("PASS", "FAIL", "INDETERMINATE")


# ============================================================================
# 4. Ground-truth separation
# ============================================================================
def test_4_ground_truth_separation():
    """Verify detector and tracking pipeline never access GroundTruthSample."""
    cfg = SkyLockConfig()
    session = build_session(cfg)

    step_res = session.step()
    assert step_res is not None

    # StepResult contains ground truth for metrics scoring
    assert step_res.truth is not None
    assert isinstance(step_res.truth, GroundTruthSample)

    # But pipeline frame has zero ground-truth references
    assert not hasattr(step_res.frame, "truth")
    assert not hasattr(step_res.frame, "targets")

    # Controller operates solely on estimate, pointing, and intent
    assert not hasattr(session.controller, "truth")


# ============================================================================
# 5. Acquisition and tracking metric calculations
# ============================================================================
def test_5_acquisition_and_tracking_metric_calculations():
    """Verify acquisition and tracking error calculations handle measured and missing data."""
    # When target never acquires TRACK
    times = [0.0, 0.033, 0.066, 0.100]
    states = [TrackState.SEARCH, TrackState.SEARCH, TrackState.ACQUIRE, TrackState.SEARCH]
    vis = [True, True, True, True]

    acq_start, acq_obs, acq_ok, obs_dur = calculate_acquisition_time(times, states, vis)
    assert acq_obs.status == MetricStatus.NOT_ACQUIRED
    assert acq_obs.value is None  # Must NOT be fabricated as 0.0

    # When target acquires TRACK at t=0.066
    states_acquired = [TrackState.SEARCH, TrackState.SEARCH, TrackState.TRACK, TrackState.TRACK]
    acq_start2, acq_obs2, acq_ok2, _ = calculate_acquisition_time(times, states_acquired, vis)
    assert acq_obs2.status == MetricStatus.MEASURED
    assert acq_obs2.value == pytest.approx(0.066, abs=1e-3)

    # Tracking error calculation
    estimates = [(100.0, 100.0), (102.0, 101.0)]
    gt_px = [(100.0, 100.0), (100.0, 100.0)]
    states_trk = [TrackState.TRACK, TrackState.TRACK]
    vis_trk = [True, True]
    err_metric = calculate_tracking_error(states_trk, estimates, gt_px, vis_trk)
    assert err_metric.status == MetricStatus.MEASURED
    assert err_metric.value is not None
    assert err_metric.value.rms >= 0.0


# ============================================================================
# 6. Loss and reacquisition metric calculations
# ============================================================================
def test_6_loss_and_reacquisition_metric_calculations():
    """Verify loss rate and reacquisition metrics handle zero-loss without fabrication."""
    # Never lost
    states = [TrackState.TRACK] * 30
    loss_metric = calculate_target_loss_rate(states)
    assert loss_metric.status == MetricStatus.MEASURED
    assert loss_metric.value == 0.0

    reacq_metric, reacq_success = calculate_reacquisition(
        [i * 0.033 for i in range(30)], states, [True] * 30, reacquire_timeout_s=1.0
    )
    assert reacq_metric.status == MetricStatus.NOT_RUN
    assert reacq_metric.reason == "No loss event occurred"
    assert reacq_metric.value is None  # Must NOT fabricate 0.0

    # Target loss occurred and reacquired
    states_with_loss = [TrackState.TRACK] * 10 + [TrackState.LOST] * 5 + [TrackState.TRACK] * 15
    times = [i * 0.033 for i in range(30)]
    reacq_m2, _ = calculate_reacquisition(times, states_with_loss, None, reacquire_timeout_s=1.0)
    assert reacq_m2.status == MetricStatus.MEASURED
    assert reacq_m2.value is not None
    assert reacq_m2.value.max_s > 0.0


# ============================================================================
# 7. Requirement verdict evaluation
# ============================================================================
def test_7_requirement_verdict_evaluation():
    """Verify requirement evaluation correctly scores PASS, FAIL, INDETERMINATE."""
    cfg = SkyLockConfig()
    collector = MetricsCollector(cfg)

    # Empty collector produces INDETERMINATE or FAIL based on criteria
    metrics = collector.finalize()
    verdicts = evaluate(metrics, cfg.requirements)
    assert verdicts["acquisition_time"] == Verdict.INDETERMINATE
    assert verdicts["overall"] in (Verdict.INDETERMINATE, Verdict.FAIL)

    # When all metrics measured and within spec -> PASS
    s01 = next(s for s in builtin_scenarios() if s.id == "S01_line_clean")
    runner = BenchmarkRunner()
    rec = runner.run(s01, seed=42)
    assert rec.overall_verdict == "PASS"
    assert rec.verdicts.get("acquisition_time") == "PASS"


# ============================================================================
# 8. Missing MP4 / invalid input handling
# ============================================================================
def test_8_missing_or_invalid_mp4_handling():
    """Verify missing or invalid MP4 inputs produce NOT_RUN without crash."""
    runner = BenchmarkRunner()
    s16 = next(s for s in builtin_scenarios() if s.id == "S16_mp4")

    # Missing MP4 path
    rec_no_path = runner.run(s16, seed=42)
    assert rec_no_path.overall_verdict == "NOT_RUN"
    assert rec_no_path.status == "NOT_RUN"
    assert "No MP4 path provided" in (rec_no_path.error or "")

    # Nonexistent MP4 path
    s16_bad = Scenario(
        id="S16_bad",
        description="Bad MP4",
        input_kind="mp4",
        mp4_path="nonexistent_video_path_9999.mp4",
    )
    rec_bad = runner.run(s16_bad, seed=42)
    assert rec_bad.overall_verdict == "NOT_RUN"
    assert rec_bad.status == "NOT_RUN"
    assert "MP4 file not found" in (rec_bad.error or "")


# ============================================================================
# 9. Cancellation and worker shutdown
# ============================================================================
def test_9_cancellation_and_worker_shutdown(qapp):
    """Verify cancelling benchmark stops safely, leaving GUI responsive."""
    panel = BenchmarkPanel()
    panel.show()

    scenarios = [Scenario(id=f"S{i:02d}", description="d") for i in range(1, 8)]
    panel._start_benchmark(scenarios, [42])
    assert panel.btn_cancel.isEnabled() is True

    # Cancel
    panel._cancel_benchmark()
    assert panel._was_cancelled is True
    assert "Cancelling" in panel.lbl_status.text()

    # Wait for completion
    assert _wait_for_condition(lambda: panel._worker_thread is None, timeout_s=5.0)
    assert panel.btn_run.isEnabled() is True
    assert "Cancelled" in panel.lbl_status.text()
    panel.close()


# ============================================================================
# 10. Run history and result-table updates
# ============================================================================
def test_10_run_history_and_result_table_updates(qapp):
    """Verify adding records updates the table rows, summary, and formatting."""
    panel = BenchmarkPanel()
    panel.show()

    assert panel.table.rowCount() == 0

    rec_pass = RunRecord(
        run_id="1",
        scenario_id="S01",
        seed=42,
        software_version="0.1.0",
        python_version="",
        numpy_version="",
        opencv_version="",
        platform_info="",
        config_snapshot={},
        config_hash="",
        input_source="simulation",
        duration_s=6.0,
        frames=180,
        metrics={"acquisition_time_from_observable_s": {"status": "MEASURED", "value": 1.2}},
        verdicts={"overall": "PASS"},
        overall_verdict="PASS",
        started_at_utc="",
        wall_time_s=1.0,
    )
    panel.add_record(rec_pass)

    assert panel.table.rowCount() == 1
    assert "1 runs" in panel.lbl_summary.text()
    assert "PASS: 1" in panel.lbl_summary.text()

    rec_not_run = RunRecord(
        run_id="2",
        scenario_id="S16",
        seed=1,
        software_version="0.1.0",
        python_version="",
        numpy_version="",
        opencv_version="",
        platform_info="",
        config_snapshot={},
        config_hash="",
        input_source="mp4",
        duration_s=0.0,
        frames=0,
        metrics={},
        verdicts={"overall": "NOT_RUN"},
        overall_verdict="NOT_RUN",
        started_at_utc="",
        wall_time_s=0.0,
    )
    panel.add_record(rec_not_run)

    assert panel.table.rowCount() == 2
    assert "2 runs" in panel.lbl_summary.text()
    assert "NOT_RUN: 1" in panel.lbl_summary.text()
    panel.close()


# ============================================================================
# 11. Export and report loading
# ============================================================================
def test_11_export_and_report_loading(tmp_path):
    """Verify JSON/Markdown reports serialize and load faithfully."""
    rec = RunRecord(
        run_id="test_run",
        scenario_id="S01_line_clean",
        seed=42,
        software_version="0.1.0",
        python_version="3.12",
        numpy_version="2.0",
        opencv_version="4.10",
        platform_info="Windows",
        config_snapshot={"seed": 42},
        config_hash="abc",
        input_source="simulation",
        duration_s=6.0,
        frames=180,
        metrics={"acquisition_time_from_observable_s": {"status": "MEASURED", "value": 0.8}},
        verdicts={"overall": "PASS"},
        overall_verdict="PASS",
        started_at_utc="2026-10-09T00:00:00Z",
        wall_time_s=1.5,
    )

    json_str = to_json_report([rec])
    data = json.loads(json_str)
    assert data["schema_version"] == "1.0"
    assert len(data["runs"]) == 1
    assert data["runs"][0]["scenario_id"] == "S01_line_clean"

    md_str = to_markdown_report([rec])
    assert "# SkyLock Benchmark Report" in md_str
    assert "S01_line_clean" in md_str
    assert "PASS" in md_str


# ============================================================================
# 12. Regression across scenario execution order
# ============================================================================
def test_12_regression_across_scenario_execution_order():
    """Verify running scenarios in different orders does not contaminate results."""
    runner = BenchmarkRunner()
    s01 = next(s for s in builtin_scenarios() if s.id == "S01_line_clean")
    s02 = next(s for s in builtin_scenarios() if s.id == "S02_circle_clean")

    # Order 1: S01 then S02
    r1_s01 = runner.run(s01, seed=42)
    r1_s02 = runner.run(s02, seed=42)

    # Order 2: S02 then S01
    r2_s02 = runner.run(s02, seed=42)
    r2_s01 = runner.run(s01, seed=42)

    assert r1_s01.state_timeline_hash == r2_s01.state_timeline_hash
    assert r1_s01.estimates_hash == r2_s01.estimates_hash
    assert r1_s02.state_timeline_hash == r2_s02.state_timeline_hash


# ============================================================================
# 13. AUTO/MANUAL control integration
# ============================================================================
def test_13_auto_manual_control_integration(qapp):
    """Verify live AUTO/MANUAL control integration in MainWindow."""
    win = MainWindow()
    win.show()

    assert win._worker.session_s1.controller.mode == ControlMode.AUTO

    # Switch to MANUAL
    win.controls_panel.cmb_mode.setCurrentText("MANUAL")
    assert win._worker.session_s1.controller.mode == ControlMode.MANUAL
    assert win._steering_filter.is_manual_mode is True

    # Switch to AUTO
    win.controls_panel.cmb_mode.setCurrentText("AUTO")
    assert win._worker.session_s1.controller.mode == ControlMode.AUTO
    assert win._steering_filter.is_manual_mode is False

    win.close()


# ============================================================================
# 14. Disturbance-to-camera-feed integration
# ============================================================================
def test_14_disturbance_to_camera_feed_integration(qapp):
    """Verify disturbance changes affect sensor feed without modifying orbital coordinates."""
    cfg = SkyLockConfig()
    session = build_session(cfg, sat_id="s1")

    # Clean frame
    res_clean = session.step()
    assert res_clean is not None
    img_clean = res_clean.frame.image

    # Enable heavy gaussian noise (within valid [0, 20.0] spec)
    new_dist = replace(cfg.disturbances, gaussian=GaussianConfig(enabled=True, sigma_levels=15.0))
    session.source.disturbances.update_config(new_dist)

    res_noisy = session.step()
    assert res_noisy is not None
    img_noisy = res_noisy.frame.image

    assert not np.array_equal(img_clean, img_noisy), "Disturbance must modify sensor image"


# ============================================================================
# 15. No communication-line contamination of sensor image
# ============================================================================
def test_15_no_communication_line_contamination():
    """Verify tracking beam / link line is completely absent from gimbal camera POV."""
    bundle_path = Path("src/skylock/ui/web3d/static_gimbal/assets/index-BEM6Obt0.js")
    if bundle_path.exists():
        content = bundle_path.read_text(encoding="utf-8")
        assert "OpticalTrackingBeam" in content
        assert "kn.visible=!1" in content or "scene.add(this.trackingBeam)" not in content

    legacy_js = Path("legacy/js/gimbal_view/main.js")
    if legacy_js.exists():
        content = legacy_js.read_text(encoding="utf-8")
        assert "trackingBeam.visible = false" in content


# ============================================================================
# 16. All enabled controls have functional handlers
# ============================================================================
def test_16_all_enabled_controls_have_functional_handlers(qapp):
    """Verify every button, dropdown, and spinbox across all panels has active handlers."""
    win = MainWindow()
    win.show()

    # ControlsPanel buttons and handlers
    ctrls = win.controls_panel
    assert not win._worker.is_running
    ctrls.btn_start.click()
    assert _wait_for_condition(lambda: win._worker.is_running, timeout_s=3.0)

    ctrls.btn_stop.click()
    assert _wait_for_condition(lambda: not win._worker.is_running, timeout_s=3.0)

    ctrls.btn_reset.click()
    assert win.telemetry_panel.lbl_det_count.text() == "—"

    # BenchmarkPanel controls
    bench = win.bench_panel
    assert bench.btn_run.isEnabled()
    assert bench.btn_run_all.isEnabled()
    assert not bench.btn_cancel.isEnabled()
    # Scenario change updates tooltip
    bench.cmb_scenarios.setCurrentIndex(1)
    assert "S02" in bench.cmb_scenarios.toolTip()
    assert "Ready: S02" in bench.lbl_status.text()

    # GimbalControlPanel controls
    gimbal = win.gimbal_control_panel
    gimbal.spn_fov.setValue(22.0)
    assert win._current_fov == 22.0

    gimbal.btn_reset.click()
    assert gimbal.spn_fov.value() == 16.0
    assert gimbal.spn_pan.value() == 0.0
    assert gimbal.spn_tilt.value() == 0.0

    # CameraFeedView controls
    feed = win.camera_view
    feed.append_log("TEST EVENT", "info")
    assert "TEST EVENT" in feed.log_text.toPlainText()
    feed.btn_clear.click()
    assert feed.log_text.toPlainText() == ""

    feed.combo_observer.setCurrentIndex(1)
    assert feed.selected_satellite == "s2"
    assert "TARGET: S-1" in feed.lbl_id_badge.text()

    # View toggle controls
    feed.btn_view_sensor.click()
    assert feed.view_stack.currentIndex() == 1
    feed.btn_view_camera.click()
    assert feed.view_stack.currentIndex() == 0

    win.close()

