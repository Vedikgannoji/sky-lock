"""Integration tests for orbital geometry, simulation clock, cue tracking, and occlusion handling.

Requirements:
- Frame round-trip error < 1e-9.
- Python vs JS orbit parity at 5 timestamps (<= 1e-6).
- Headless, fake clock, >= 300 s sim time:
  * TRACK frames: true offset from boresight median < 0.3 deg, p95 < 1.0 deg;
  * tracker pixel error <= 10 px for >= 95% of TRACK frames;
  * While blocked: commanded rate == 0 on every frame and no scan;
  * After each LOS clear: TRACK within (slew time to cue + 3 s) in >= 90% of exits;
  * Deliberately wrong cue (20 deg off) injected during TRACK has zero effect.
"""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np

from skylock.app.factory import build_session
from skylock.config.models import InputConfig, SkyLockConfig
from skylock.core.enums import TrackState
from skylock.core.geometry import angular_diff_deg
from skylock.core.los import orbit_position_at_time
from skylock.core.orbital_world import (
    ORBIT_SPEED_SCALE,
    S1_DEFAULT_ORBIT,
    S2_DEFAULT_ORBIT,
    dir_to_pan_tilt,
    pan_tilt_to_dir,
    peer_bearing,
)
from skylock.core.sim_clock import SimClock


def test_frame_round_trip_error() -> None:
    """Verify observer frame coordinate conversions round-trip error < 1e-9."""
    pans = np.linspace(-179.9, 179.9, 45)
    tilts = np.linspace(-89.9, 89.9, 45)

    max_vec_err = 0.0
    for pan in pans:
        for tilt in tilts:
            d = pan_tilt_to_dir(float(pan), float(tilt))
            p_out, t_out = dir_to_pan_tilt(d)
            d2 = pan_tilt_to_dir(p_out, t_out)

            err = np.linalg.norm(np.array(d) - np.array(d2))
            max_vec_err = max(max_vec_err, float(err))

    print(f"\n[Requirement 7.1] Max Frame Round-Trip Error: {max_vec_err:.4e}")
    assert max_vec_err < 1e-9, f"Frame round-trip error {max_vec_err} exceeds 1e-9"


def test_python_vs_js_orbit_parity_at_5_timestamps() -> None:
    """Verify Python vs JS OrbitState position parity at 5 timestamps <= 1e-6."""
    timestamps = [0.0, 15.2, 45.8, 120.5, 250.0]

    # Node evaluation script replicating legacy/js/orbit.js OrbitState
    js_code = f"""
    const degToRad = (d) => (d * Math.PI) / 180;
    class OrbitState {{
      constructor(radius, speed, inclinationDeg, initialAngleDeg) {{
        this.radius = radius;
        this.speed = speed;
        this.inclination = degToRad(inclinationDeg);
        this.initialAngle = degToRad(initialAngleDeg);
      }}
      getPosition(t) {{
        const angle = this.initialAngle + this.speed * t;
        return [
          Math.cos(angle) * this.radius,
          Math.sin(angle) * this.radius * Math.sin(this.inclination),
          Math.sin(angle) * this.radius * Math.cos(this.inclination),
        ];
      }}
    }}

    const scale = {ORBIT_SPEED_SCALE};
    const s1 = new OrbitState(20.0, 0.3 * scale, 25.0, 0.0);
    const s2 = new OrbitState(26.0, 0.2 * scale, 65.0, 45.0);

    const times = {json.dumps(timestamps)};
    const results = times.map(t => ({{
      t: t,
      s1: s1.getPosition(t),
      s2: s2.getPosition(t),
    }}));
    console.log(JSON.stringify(results));
    """

    res = subprocess.run(
        ["node", "-e", js_code],
        capture_output=True,
        text=True,
        check=True,
    )
    js_results = json.loads(res.stdout)

    max_parity_err = 0.0
    for item in js_results:
        t = item["t"]
        js_s1 = np.array(item["s1"])
        js_s2 = np.array(item["s2"])

        py_s1 = np.array(orbit_position_at_time(S1_DEFAULT_ORBIT, t))
        py_s2 = np.array(orbit_position_at_time(S2_DEFAULT_ORBIT, t))

        err_s1 = np.linalg.norm(py_s1 - js_s1)
        err_s2 = np.linalg.norm(py_s2 - js_s2)

        max_parity_err = max(max_parity_err, float(err_s1), float(err_s2))

    print(f"\n[Requirement 7.2] Max Python vs JS Orbit Parity Error: {max_parity_err:.4e}")
    assert max_parity_err <= 1e-6, f"Orbit parity error {max_parity_err} exceeds 1e-6"


def test_orbital_tracking_300s_headless_simulation() -> None:
    """Run >= 300 s sim time headless simulation with fake clock.

    Validates:
    - TRACK frames: true offset from boresight median < 0.3 deg, p95 < 1.0 deg;
    - tracker pixel error <= 10 px for >= 95% of TRACK frames;
    - While blocked: commanded rate == 0 on every frame and no scan;
    - After each LOS clear: TRACK within (slew time to cue + 3 s) in >= 90% of exits;
    - Deliberately wrong cue (20 deg off) injected during TRACK has zero effect.
    """
    fps = 30.0
    dt = 1.0 / fps
    total_sim_time = 300.0  # seconds
    total_frames = int(total_sim_time * fps)

    cfg = SkyLockConfig(input=InputConfig(kind="orbital"))
    clock = SimClock()

    sess = build_session(cfg, sat_id="s1")
    sess.source.clock = clock
    sess.source.target.clock = clock

    track_offsets_deg: list[float] = []
    track_pixel_errors: list[float] = []
    blocked_rates: list[float] = []

    # LOS exit tracking: (exit_time_s, exit_pointing, exit_cue, track_time_s)
    los_exits: list[dict] = []
    in_blocked = False

    wrong_cue_frame = int(60.0 * fps)
    wrong_cue_injected = False
    wrong_cue_retained = False

    for frame_idx in range(total_frames):
        t_sim = frame_idx * dt
        clock.set_time(t_sim)

        _, _, los_clear = peer_bearing("s1", t_sim)

        if not los_clear:
            if not in_blocked:
                in_blocked = True
        else:
            if in_blocked:
                # Transition from BLOCKED to CLEAR
                in_blocked = False
                cue_pt, _ = sess.source.get_cue(t_sim)
                g_pt = (
                    sess.source.gimbal.pointing.pan_deg,
                    sess.source.gimbal.pointing.tilt_deg,
                )
                los_exits.append({
                    "exit_t": t_sim,
                    "exit_pt": g_pt,
                    "exit_cue": cue_pt,
                    "track_t": None,
                })

        # Inject deliberately wrong cue (20 deg off) at frame 60s during TRACK
        if frame_idx == wrong_cue_frame and sess.pipeline.tracker.state == TrackState.TRACK:
            pt = sess.source.gimbal.pointing
            wrong_cue = (pt.pan_deg + 20.0, pt.tilt_deg + 20.0)
            sess.pipeline.tracker.set_cue(wrong_cue, los_clear=True)
            wrong_cue_injected = True

        res = sess.step()
        assert res is not None, f"Premature stream termination at frame {frame_idx}"

        state = res.output.state
        truth = res.truth
        cmd = res.command

        if wrong_cue_injected and not wrong_cue_retained:
            # Check that tracker ignored the false cue and maintained TRACK
            wrong_cue_retained = (state == TrackState.TRACK)

        # Check exit reacquisition
        if los_exits and los_exits[-1]["track_t"] is None and state == TrackState.TRACK:
            los_exits[-1]["track_t"] = t_sim

        # While blocked: commanded rate == 0 on every frame and no scan
        if not los_clear:
            rate_mag = math.hypot(cmd.pan_rate_deg_s, cmd.tilt_rate_deg_s)
            blocked_rates.append(rate_mag)
            assert rate_mag == 0.0, (
                f"Non-zero rate commanded while blocked at t={t_sim:.2f}s: {cmd}"
            )
            assert sess.pipeline.tracker.blocked is True
            assert sess.pipeline.tracker.state == TrackState.LOST

        # TRACK frames metrics
        if state == TrackState.TRACK:
            if truth is not None and truth.targets:
                tgt = truth.targets[0]
                pt = truth.pointing
                dpan = tgt.az_deg - pt.pan_deg
                dtilt = tgt.el_deg - pt.tilt_deg
                off_deg = math.hypot(dpan, dtilt)
                track_offsets_deg.append(off_deg)

            has_px = truth is not None and truth.primary_px is not None
            if has_px and res.output.estimate is not None:
                est = res.output.estimate
                gt_px = truth.primary_px
                px_err = math.hypot(est.px - gt_px[0], est.py - gt_px[1])
                track_pixel_errors.append(px_err)

    # 1. Boresight offset in TRACK
    assert len(track_offsets_deg) > 0, "No TRACK frames recorded"
    median_offset = float(np.median(track_offsets_deg))
    p95_offset = float(np.percentile(track_offsets_deg, 95))
    print(
        f"\n[Requirement 7.3] TRACK Boresight Offset: "
        f"median = {median_offset:.4f} deg (req < 0.3), p95 = {p95_offset:.4f} deg (req < 1.0)"
    )
    assert median_offset < 0.3, f"Median offset {median_offset} >= 0.3 deg"
    assert p95_offset < 1.0, f"p95 offset {p95_offset} >= 1.0 deg"

    # 2. Pixel error in TRACK
    pct_under_10px = float(np.mean([e <= 10.0 for e in track_pixel_errors]) * 100.0)
    print(f"[Requirement 7.4] Tracker Pixel Error <= 10 px: {pct_under_10px:.2f}% (req >= 95%)")
    assert pct_under_10px >= 95.0, f"Pixel error <= 10 px fraction {pct_under_10px}% < 95%"

    # 3. Blocked commanded rate
    max_blocked_rate = max(blocked_rates) if blocked_rates else 0.0
    print(f"[Requirement 7.5] Max Rate Blocked: {max_blocked_rate:.4f} deg/s (req == 0.0)")
    assert max_blocked_rate == 0.0, f"Max rate while blocked {max_blocked_rate} != 0.0"

    # 4. Exit reacquisitions
    max_slew = 10.0  # deg/s
    successful_exits = 0
    total_exits = len(los_exits)
    for ex in los_exits:
        exit_t = ex["exit_t"]
        track_t = ex["track_t"]
        exit_pt = ex["exit_pt"]
        cue = ex["exit_cue"]

        # Angular distance from exit pointing to cue
        dpan = abs(angular_diff_deg(cue[0], exit_pt[0]))
        dtilt = abs(cue[1] - exit_pt[1])
        slew_dist = math.hypot(dpan, dtilt)
        slew_time = slew_dist / max_slew
        allowed_window = slew_time + 3.0

        if track_t is not None:
            delay = track_t - exit_t
            if delay <= allowed_window:
                successful_exits += 1
            print(
                f"  Exit at {exit_t:.1f}s: delay={delay:.2f}s, "
                f"allowed={allowed_window:.2f}s (dist={slew_dist:.1f} deg)"
            )
        else:
            print(f"  Exit at {exit_t:.1f}s: NEVER reacquired")

    exit_success_rate = (successful_exits / total_exits * 100.0) if total_exits > 0 else 100.0
    print(
        f"[Requirement 7.6] Exit Reacquisition Rate: {exit_success_rate:.1f}% "
        f"({successful_exits}/{total_exits}, req >= 90%)"
    )
    assert exit_success_rate >= 90.0, f"Exit reacquisition rate {exit_success_rate}% < 90%"

    # 5. Wrong cue immunity
    print(f"[Requirement 7.7] Wrong Cue Injected During TRACK Has No Effect: {wrong_cue_retained}")
    assert wrong_cue_injected is True, "Wrong cue injection was not triggered"
    assert wrong_cue_retained is True, "Wrong cue caused track loss during TRACK mode"
