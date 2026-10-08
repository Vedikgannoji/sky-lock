"""Geometric Line-of-Sight, Earth occlusion, and 3D look-angle calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from skylock.core.geometry import angular_diff_deg, wrap_deg

EARTH_RADIUS = 10.0


@dataclass
class OrbitParams:
    """Parameters defining a circular inclined satellite orbit."""

    radius: float
    speed: float  # rad/s
    inclination_deg: float
    phase_deg: float = 0.0


def orbit_position_at_time(params: OrbitParams, t_s: float) -> tuple[float, float, float]:
    """Calculate 3D Cartesian position (x, y, z) on orbit at time t_s."""
    angle = math.radians(params.phase_deg) + params.speed * t_s
    inc = math.radians(params.inclination_deg)

    x = math.cos(angle) * params.radius
    y = math.sin(angle) * params.radius * math.sin(inc)
    z = math.sin(angle) * params.radius * math.cos(inc)
    return (x, y, z)


def get_satellite_position(
    sat_id: str,
    t_s: float = 0.0,
    orbit: OrbitParams | None = None,
) -> tuple[float, float, float]:
    """Calculate 3D position (x, y, z) of a satellite relative to Earth at time t_s."""
    if orbit is not None:
        return orbit_position_at_time(orbit, t_s)
    from skylock.core.orbital_world import S1_DEFAULT_ORBIT, S2_DEFAULT_ORBIT

    sat_key = sat_id.lower().replace("-", "")
    params = S2_DEFAULT_ORBIT if sat_key in ("s2", "sat2") else S1_DEFAULT_ORBIT
    return orbit_position_at_time(params, t_s)


def check_line_of_sight(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    body_radius: float = EARTH_RADIUS,
    body_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bool:
    """Determine whether the direct optical line-of-sight between p1 and p2 is clear.

    Returns True if the line segment between p1 and p2 does NOT intersect the occluding
    body sphere. Returns False if occluded/blocked by the sphere.
    """
    # Relative positions with respect to body center
    x1 = p1[0] - body_center[0]
    y1 = p1[1] - body_center[1]
    z1 = p1[2] - body_center[2]

    x2 = p2[0] - body_center[0]
    y2 = p2[1] - body_center[1]
    z2 = p2[2] - body_center[2]

    # Segment vector D = p2 - p1
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    seg_len_sq = dx * dx + dy * dy + dz * dz

    if seg_len_sq < 1e-12:
        # Endpoints coincide
        return math.sqrt(x1 * x1 + y1 * y1 + z1 * z1) >= body_radius

    # Parametric line: P(t) = P1 + t * D, t in [0, 1]
    # Projection of origin onto infinite line:
    # dot(P1 + t*D, D) = 0 => t = -dot(P1, D) / dot(D, D)
    dot_p1_d = x1 * dx + y1 * dy + z1 * dz
    t = -dot_p1_d / seg_len_sq

    # Clamp t to segment [0, 1] to find closest point on segment
    t_clamped = max(0.0, min(1.0, t))
    cx = x1 + t_clamped * dx
    cy = y1 + t_clamped * dy
    cz = z1 + t_clamped * dz

    dist_sq = cx * cx + cy * cy + cz * cz
    # Clear if distance to closest point is strictly greater than body radius
    return dist_sq >= (body_radius * body_radius)


def calculate_look_angles(
    obs_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Calculate azimuth (pan) and elevation (tilt) from observer to target in degrees.

    Returns (pan_deg, tilt_deg, distance).
    """
    dx = target_pos[0] - obs_pos[0]
    dy = target_pos[1] - obs_pos[1]
    dz = target_pos[2] - obs_pos[2]

    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance < 1e-6:
        return (0.0, 0.0, 0.0)

    # In standard coordinate frame:
    # Pan (azimuth in X-Z plane): 0° along +Z (or -Z), atan2(dx, dz)
    pan_deg = math.degrees(math.atan2(dx, -dz))
    # Tilt (elevation above X-Z plane):
    horiz_dist = math.sqrt(dx * dx + dz * dz)
    tilt_deg = math.degrees(math.atan2(dy, horiz_dist))

    return (wrap_deg(pan_deg), wrap_deg(tilt_deg), distance)


def is_target_in_fov(
    current_pan: float,
    current_tilt: float,
    target_pan: float,
    target_tilt: float,
    fov_deg: float,
    aspect: float = 4.0 / 3.0,
) -> bool:
    """Check if target look angles fall inside camera rectangular FOV frustum."""
    d_pan = abs(angular_diff_deg(target_pan, current_pan))
    d_tilt = abs(angular_diff_deg(target_tilt, current_tilt))

    half_fov_h = (fov_deg * aspect) / 2.0
    half_fov_v = fov_deg / 2.0

    return (d_pan <= half_fov_h) and (d_tilt <= half_fov_v)


def slew_toward_target(
    current_pan: float,
    current_tilt: float,
    target_pan: float,
    target_tilt: float,
    max_slew_deg_s: float,
    dt_s: float,
) -> tuple[float, float, bool]:
    """Step gimbal pan and tilt toward target angles respecting max slew rate limit.

    Returns (new_pan, new_tilt, is_aligned).
    """
    max_step = max_slew_deg_s * dt_s

    diff_pan = angular_diff_deg(target_pan, current_pan)
    if abs(diff_pan) <= max_step:
        new_pan = target_pan
        pan_aligned = True
    else:
        new_pan = current_pan + math.copysign(max_step, diff_pan)
        pan_aligned = False

    diff_tilt = angular_diff_deg(target_tilt, current_tilt)
    if abs(diff_tilt) <= max_step:
        new_tilt = target_tilt
        tilt_aligned = True
    else:
        new_tilt = current_tilt + math.copysign(max_step, diff_tilt)
        tilt_aligned = False

    return (wrap_deg(new_pan), wrap_deg(new_tilt), pan_aligned and tilt_aligned)
