"""Benchmark runner executing scenarios with deterministic isolation and full provenance."""

from __future__ import annotations

import hashlib
import multiprocessing
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import skylock
from skylock.app.factory import build_session
from skylock.benchmark.scenario import Scenario
from skylock.config.io import config_hash, to_dict
from skylock.config.models import SkyLockConfig
from skylock.core.enums import Verdict
from skylock.metrics.collector import MetricsCollector
from skylock.metrics.logger import CSVFrameLogger
from skylock.metrics.requirements import evaluate


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Complete provenance record for a single benchmark run.

    Contains everything needed to reproduce, compare, and audit the result.
    """

    run_id: str
    scenario_id: str
    seed: int
    software_version: str
    python_version: str
    numpy_version: str
    opencv_version: str
    platform_info: str
    config_snapshot: dict[str, Any]
    config_hash: str
    input_source: str
    duration_s: float
    frames: int
    metrics: dict[str, Any]
    verdicts: dict[str, str]
    overall_verdict: str
    started_at_utc: str
    wall_time_s: float
    git_commit: str | None = None
    status: str = "COMPLETED"
    error: str | None = None
    state_timeline_hash: str | None = None
    estimates_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for JSON output."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "software_version": self.software_version,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "opencv_version": self.opencv_version,
            "platform_info": self.platform_info,
            "config_snapshot": self.config_snapshot,
            "config_hash": self.config_hash,
            "input_source": self.input_source,
            "duration_s": self.duration_s,
            "frames": self.frames,
            "metrics": self.metrics,
            "verdicts": self.verdicts,
            "overall_verdict": self.overall_verdict,
            "started_at_utc": self.started_at_utc,
            "wall_time_s": self.wall_time_s,
            "git_commit": self.git_commit,
            "status": self.status,
            "error": self.error,
            "state_timeline_hash": self.state_timeline_hash,
            "estimates_hash": self.estimates_hash,
        }


@dataclass(slots=True)
class BenchmarkRunner:
    """Executes benchmark scenarios with deterministic isolation.

    Each run creates a fresh Session with no shared mutable state.
    cv2.setNumThreads(1) is set before every run for cross-platform reproducibility.
    """

    base_config: SkyLockConfig = field(default_factory=SkyLockConfig)

    def run(
        self,
        scenario: Scenario,
        seed: int,
        isolate: bool = False,
        output_dir: Path | None = None,
    ) -> RunRecord:
        """Execute a single benchmark scenario with given seed.

        Args:
            scenario: Scenario descriptor to execute.
            seed: Explicit deterministic seed.
            isolate: If True, run in a spawned subprocess for global state isolation.
            output_dir: Optional directory for per-run CSV logs (e.g. frames.csv).

        Returns:
            Complete RunRecord with metrics, verdicts, and provenance.
        """
        if scenario.input_kind == "mp4":
            if not scenario.mp4_path:
                return _not_run_record(scenario, seed, "No MP4 path provided")
            mp4_file = Path(scenario.mp4_path)
            if not mp4_file.exists() or not mp4_file.is_file():
                return _not_run_record(scenario, seed, f"MP4 file not found: {scenario.mp4_path}")

        if isolate:
            return _run_isolated(self.base_config, scenario, seed)

        try:
            return _run_in_process(self.base_config, scenario, seed, output_dir)
        except Exception as e:
            prov = _provenance()
            is_src_err = "SourceError" in type(e).__name__ or "not found" in str(e).lower()
            return RunRecord(
                run_id=str(uuid.uuid4()),
                scenario_id=scenario.id,
                seed=seed,
                software_version=prov["software_version"],
                python_version=prov["python_version"],
                numpy_version=prov["numpy_version"],
                opencv_version=prov["opencv_version"],
                platform_info=prov["platform_info"],
                config_snapshot={},
                config_hash="",
                input_source=scenario.input_kind,
                duration_s=scenario.duration_s,
                frames=0,
                metrics={},
                verdicts={"overall": "NOT_RUN" if is_src_err else "FAIL"},
                overall_verdict="NOT_RUN" if is_src_err else "FAIL",
                started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                wall_time_s=0.0,
                git_commit=_get_git_commit(),
                status="NOT_RUN" if is_src_err else "FAILED",
                error=str(e),
            )


def _get_git_commit() -> str | None:
    """Attempt to retrieve the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _provenance() -> dict[str, str]:
    """Gather environment provenance metadata."""
    return {
        "software_version": skylock.__version__,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "platform_info": platform.platform(),
    }


def _compute_state_timeline_hash(results: list) -> str:
    """Deterministic hash of the state timeline for reproducibility verification."""
    hasher = hashlib.sha256()
    for r in results:
        hasher.update(r.output.state.encode("utf-8"))
    return hasher.hexdigest()


def _compute_estimates_hash(results: list) -> str:
    """Deterministic hash of the estimate sequence for reproducibility verification."""
    hasher = hashlib.sha256()
    for r in results:
        est = r.output.estimate
        if est is not None:
            # Use repr for deterministic float serialization
            hasher.update(f"{est.px!r},{est.py!r},{est.pan_deg!r},{est.tilt_deg!r}".encode())
        else:
            hasher.update(b"None")
    return hasher.hexdigest()


def _run_in_process(
    base_config: SkyLockConfig,
    scenario: Scenario,
    seed: int,
    output_dir: Path | None = None,
) -> RunRecord:
    """Execute a benchmark run in the current process."""
    # Deterministic threading
    cv2.setNumThreads(1)

    run_id = str(uuid.uuid4())
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prov = _provenance()
    git_commit = _get_git_commit()

    # Build fresh config and session - no shared state
    cfg = scenario.apply(base_config, seed)
    cfg_snapshot = to_dict(cfg)
    cfg_hash = config_hash(cfg)

    collector = MetricsCollector(cfg)
    session = build_session(cfg)
    session.collector = collector

    # Optional CSV frame logging
    csv_logger = None
    if output_dir is not None:
        csv_path = output_dir / f"{scenario.id}_seed{seed}_frames.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Include ground truth for simulation runs
        include_gt = (scenario.input_kind == "simulation")
        csv_logger = CSVFrameLogger(csv_path, include_ground_truth=include_gt)

    # Run with wall-clock timing only around the loop
    t0 = time.perf_counter()
    results = (
        session.run(seconds=scenario.duration_s)
        if scenario.duration_s > 0
        else session.run()
    )
    wall_time_s = time.perf_counter() - t0

    # Write frame logs if enabled
    if csv_logger is not None:
        for result in results:
            csv_logger.log(result)
        csv_logger.close()

    # Finalize metrics
    run_metrics = collector.finalize()
    verdicts = evaluate(run_metrics, cfg.requirements)

    # Compute deterministic hashes for comparison
    state_hash = _compute_state_timeline_hash(results)
    estimates_hash = _compute_estimates_hash(results)

    overall = verdicts.get("overall", Verdict.INDETERMINATE)

    return RunRecord(
        run_id=run_id,
        scenario_id=scenario.id,
        seed=seed,
        software_version=prov["software_version"],
        python_version=prov["python_version"],
        numpy_version=prov["numpy_version"],
        opencv_version=prov["opencv_version"],
        platform_info=prov["platform_info"],
        config_snapshot=cfg_snapshot,
        config_hash=cfg_hash,
        input_source=scenario.input_kind,
        duration_s=scenario.duration_s,
        frames=len(results),
        metrics=run_metrics.as_dict(),
        verdicts={k: str(v) for k, v in verdicts.items()},
        overall_verdict=str(overall),
        started_at_utc=started_at,
        wall_time_s=wall_time_s,
        git_commit=git_commit,
        state_timeline_hash=state_hash,
        estimates_hash=estimates_hash,
    )


def _not_run_record(scenario: Scenario, seed: int, reason: str) -> RunRecord:
    """Create a NOT_RUN record for scenarios that cannot execute."""
    prov = _provenance()
    return RunRecord(
        run_id=str(uuid.uuid4()),
        scenario_id=scenario.id,
        seed=seed,
        software_version=prov["software_version"],
        python_version=prov["python_version"],
        numpy_version=prov["numpy_version"],
        opencv_version=prov["opencv_version"],
        platform_info=prov["platform_info"],
        config_snapshot={},
        config_hash="",
        input_source=scenario.input_kind,
        duration_s=0.0,
        frames=0,
        metrics={},
        verdicts={"overall": "NOT_RUN"},
        overall_verdict="NOT_RUN",
        started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        wall_time_s=0.0,
        git_commit=_get_git_commit(),
        status="NOT_RUN",
        error=reason,
    )


def _subprocess_worker(
    base_config_dict: dict[str, Any],
    scenario_dict: dict[str, Any],
    seed: int,
    result_queue: multiprocessing.Queue,  # type: ignore[type-arg]
) -> None:
    """Worker function for subprocess isolation."""
    from skylock.benchmark.scenario import Scenario
    from skylock.config.io import from_dict

    base_cfg = from_dict(base_config_dict)
    scn = Scenario(**scenario_dict)
    record = _run_in_process(base_cfg, scn, seed)
    result_queue.put(record.as_dict())


def _scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Convert Scenario to a dict suitable for subprocess serialization."""
    return {
        "id": scenario.id,
        "description": scenario.description,
        "overrides": scenario.overrides,
        "duration_s": scenario.duration_s,
        "seeds": scenario.seeds,
        "input_kind": scenario.input_kind,
        "mp4_path": scenario.mp4_path,
        "tags": scenario.tags,
    }


def _run_isolated(
    base_config: SkyLockConfig,
    scenario: Scenario,
    seed: int,
) -> RunRecord:
    """Execute a benchmark run in a spawned subprocess for full isolation."""
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[dict[str, Any]] = ctx.Queue()

    base_dict = to_dict(base_config)
    scenario_dict = _scenario_to_dict(scenario)

    proc = ctx.Process(
        target=_subprocess_worker,
        args=(base_dict, scenario_dict, seed, result_queue),
    )
    proc.start()
    proc.join(timeout=300)

    if proc.exitcode != 0:
        prov = _provenance()
        return RunRecord(
            run_id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            seed=seed,
            software_version=prov["software_version"],
            python_version=prov["python_version"],
            numpy_version=prov["numpy_version"],
            opencv_version=prov["opencv_version"],
            platform_info=prov["platform_info"],
            config_snapshot={},
            config_hash="",
            input_source=scenario.input_kind,
            duration_s=0.0,
            frames=0,
            metrics={},
            verdicts={"overall": "FAIL"},
            overall_verdict="FAIL",
            started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_time_s=0.0,
            git_commit=_get_git_commit(),
            status="SUBPROCESS_ERROR",
            error=f"Subprocess exited with code {proc.exitcode}",
        )

    raw = result_queue.get(timeout=10)
    return RunRecord(**raw)


__all__ = ("BenchmarkRunner", "RunRecord")
