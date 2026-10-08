"""Single geometry authority for orbital tracking, observer frames, and bearing angles.

Observer Reference Frame Definition:
------------------------------------
For a satellite on an orbit around Earth (centered at (0, 0, 0)):
- x-axis: Tangent along velocity direction (unit vector in direction of motion).
- z-axis: Nadir pointing directly towards Earth center (-position / ||position||).
- y-axis: Cross product of nadir and velocity (z cross x).
Together (x, y, z) form an orthonormal, right-handed coordinate basis attached to the satellite.

Spherical Look-Angles in Observer Frame:
----------------------------------------
For a target unit direction d = (d_x, d_y, d_z) in the satellite's observer frame:
- up vector: -z (zenith direction).
- elevation (tilt): asin(d_up) = asin(-d_z).
  * tilt = -90° points directly to nadir (Earth center, along +z).
  * tilt = +90° points to zenith (along -z).
  * tilt = 0° lies on the local horizontal plane.
- azimuth (pan): atan2(d_y, d_x).
  * pan = 0° points along velocity (+x).
  * pan = +90° points along +y.
  * pan = ±180° points opposite to velocity (-x).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from skylock.core.geometry import wrap_deg
from skylock.core.los import (
    EARTH_RADIUS,
    OrbitParams,
    check_line_of_sight,
    orbit_position_at_time,
)

if TYPE_CHECKING:
    pass

# Authoritative orbital speed scaling factor chosen so p99 LOS-clear bearing rate <= 8.0 deg/s
ORBIT_SPEED_SCALE: float = 0.25


S1_DEFAULT_ORBIT = OrbitParams(
    radius=20.0,
    speed=0.3 * ORBIT_SPEED_SCALE,
    inclination_deg=25.0,
    phase_deg=0.0,
)
S2_DEFAULT_ORBIT = OrbitParams(
    radius=26.0,
    speed=0.2 * ORBIT_SPEED_SCALE,
    inclination_deg=65.0,
    phase_deg=45.0,
)


def get_default_orbit(sat_id: str) -> OrbitParams:
    """Return default OrbitParams for sat_id with authoritative ORBIT_SPEED_SCALE applied."""
    sat_key = sat_id.lower().replace("-", "")
    if sat_key in ("s2", "sat2"):
        return S2_DEFAULT_ORBIT
    return S1_DEFAULT_ORBIT


def dir_to_pan_tilt(d: tuple[float, float, float] | np.ndarray) -> tuple[float, float]:
    """Convert unit direction vector in observer frame to (pan_deg, tilt_deg).

    Conventions:
    - pan = atan2(d.y, d.x) in degrees, wrapped to (-180, 180].
    - tilt = asin(d.up) where up = -z (tilt -90° = nadir).
    """
    dx = float(d[0])
    dy = float(d[1])
    dz = float(d[2])

    d_up = -dz
    d_up_clamped = max(-1.0, min(1.0, d_up))
    tilt_deg = math.degrees(math.asin(d_up_clamped))
    pan_deg = math.degrees(math.atan2(dy, dx))

    return (wrap_deg(pan_deg), tilt_deg)


def pan_tilt_to_dir(pan_deg: float, tilt_deg: float) -> tuple[float, float, float]:
    """Convert (pan_deg, tilt_deg) to unit direction vector in observer frame.

    Inverse of dir_to_pan_tilt.
    """
    pan_rad = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_deg)

    cos_tilt = math.cos(tilt_rad)
    dx = cos_tilt * math.cos(pan_rad)
    dy = cos_tilt * math.sin(pan_rad)
    # d_up = sin(tilt_rad) => dz = -d_up = -sin(tilt_rad)
    dz = -math.sin(tilt_rad)

    return (dx, dy, dz)


def get_satellite_basis(
    sat_id_or_orbit: str | OrbitParams,
    t_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate orthonormal observer basis (x, y, z) in world Cartesian coordinates.

    Returns:
        (x_basis, y_basis, z_basis) as 3D unit numpy arrays in world coordinates.
    """
    orbit = (
        sat_id_or_orbit
        if isinstance(sat_id_or_orbit, OrbitParams)
        else get_default_orbit(sat_id_or_orbit)
    )

    angle = math.radians(orbit.phase_deg) + orbit.speed * t_s
    inc = math.radians(orbit.inclination_deg)

    # x = velocity direction unit vector
    # d/dt [r*cos(A), r*sin(A)*sin(inc), r*sin(A)*cos(inc)]
    # = r*spd * [-sin(A), cos(A)*sin(inc), cos(A)*cos(inc)]
    vx = -math.sin(angle)
    vy = math.cos(angle) * math.sin(inc)
    vz = math.cos(angle) * math.cos(inc)
    x_basis = np.array([vx, vy, vz], dtype=np.float64)
    x_basis /= np.linalg.norm(x_basis)

    # z = nadir unit vector (towards Earth center at origin)
    # -pos / radius = [-cos(A), -sin(A)*sin(inc), -sin(A)*cos(inc)]
    zx = -math.cos(angle)
    zy = -math.sin(angle) * math.sin(inc)
    zz = -math.sin(angle) * math.cos(inc)
    z_basis = np.array([zx, zy, zz], dtype=np.float64)
    z_basis /= np.linalg.norm(z_basis)

    # y = z cross x
    y_basis = np.cross(z_basis, x_basis)
    y_basis /= np.linalg.norm(y_basis)

    return (x_basis, y_basis, z_basis)


def peer_bearing(
    observer: str | OrbitParams,
    t_s: float,
    peer: str | OrbitParams | None = None,
) -> tuple[tuple[float, float, float], float, bool]:
    """Calculate bearing from observer satellite to peer satellite at time t_s.

    Returns:
        (unit_vector_obs, distance, los_clear)
        where unit_vector_obs is in the observer satellite's reference frame.
    """
    if isinstance(observer, str):
        obs_orbit = get_default_orbit(observer)
        peer_id = peer if isinstance(peer, str) else ("s2" if observer.lower() in ("s1", "sat1") else "s1")
        peer_orbit = peer if isinstance(peer, OrbitParams) else get_default_orbit(peer_id)
    else:
        obs_orbit = observer
        if peer is None:
            peer_orbit = get_default_orbit("s2")
        elif isinstance(peer, str):
            peer_orbit = get_default_orbit(peer)
        else:
            peer_orbit = peer

    p_obs = np.array(orbit_position_at_time(obs_orbit, t_s), dtype=np.float64)
    p_peer = np.array(orbit_position_at_time(peer_orbit, t_s), dtype=np.float64)

    delta = p_peer - p_obs
    distance = float(np.linalg.norm(delta))

    if distance < 1e-9:
        return ((1.0, 0.0, 0.0), 0.0, True)

    d_world = delta / distance

    # Line of sight check
    los_clear = check_line_of_sight(
        tuple(p_obs), tuple(p_peer), body_radius=EARTH_RADIUS
    )

    # Project d_world into observer frame
    bx, by, bz = get_satellite_basis(obs_orbit, t_s)
    dx = float(np.dot(d_world, bx))
    dy = float(np.dot(d_world, by))
    dz = float(np.dot(d_world, bz))

    d_obs = np.array([dx, dy, dz], dtype=np.float64)
    norm = np.linalg.norm(d_obs)
    if norm > 1e-9:
        d_obs /= norm

    return ((float(d_obs[0]), float(d_obs[1]), float(d_obs[2])), distance, los_clear)


def earth_bearing(
    observer: str | OrbitParams,
    t_s: float,
) -> tuple[tuple[float, float, float], float, bool]:
    """Calculate bearing from observer satellite to Earth center at time t_s.

    In the observer frame, nadir is along +z, so the unit direction vector is always (0, 0, 1).
    Returns:
        (unit_vector_obs, distance, los_clear)
    """
    obs_orbit = (
        observer
        if isinstance(observer, OrbitParams)
        else get_default_orbit(observer)
    )
    p_obs = orbit_position_at_time(obs_orbit, t_s)
    dist = math.sqrt(p_obs[0] ** 2 + p_obs[1] ** 2 + p_obs[2] ** 2)
    return ((0.0, 0.0, 1.0), dist, True)


def camera_tangent_offset(
    bearing_unit_obs: tuple[float, float, float] | np.ndarray,
    pointing_pan_deg: float,
    pointing_tilt_deg: float,
) -> tuple[float, float, bool]:
    """Compute true tangent-plane angular offsets (dpan, dtilt) relative to camera pointing.

    Uses proper spherical geometry in camera coordinates:
    - Boresight b = pan_tilt_to_dir(pan, tilt)
    - Right r = (-sin(pan), cos(pan), 0)
    - Up u = r cross b
    - Target d = bearing_unit_obs
    - X_cam = d dot r, Y_cam = d dot u, Z_cam = d dot b

    Returns:
        (dpan_deg, dtilt_deg, in_front)
        where in_front indicates Z_cam > 0 (in front of camera sensor plane).
    """
    pan_rad = math.radians(pointing_pan_deg)
    tilt_rad = math.radians(pointing_tilt_deg)

    cos_tilt = math.cos(tilt_rad)
    sin_tilt = math.sin(tilt_rad)
    cos_pan = math.cos(pan_rad)
    sin_pan = math.sin(pan_rad)

    # Boresight unit vector
    bx = cos_tilt * cos_pan
    by = cos_tilt * sin_pan
    bz = -sin_tilt

    # Right unit vector
    rx = -sin_pan
    ry = cos_pan
    rz = 0.0

    # Up unit vector = r cross b
    ux = ry * bz - rz * by  # cos(pan) * (-sin(tilt))
    uy = rz * bx - rx * bz  # -(-sin(pan)) * (-sin(tilt)) = -sin(tilt)*sin(pan)
    uz = rx * by - ry * bx  # -sin^2(pan)*cos(tilt) - cos^2(pan)*cos(tilt) = -cos(tilt)

    dx = float(bearing_unit_obs[0])
    dy = float(bearing_unit_obs[1])
    dz = float(bearing_unit_obs[2])

    x_cam = dx * rx + dy * ry + dz * rz
    y_cam = dx * ux + dy * uy + dz * uz
    z_cam = dx * bx + dy * by + dz * bz

    in_front = z_cam > 1e-6
    # Compute tangent-plane angles
    dpan = math.degrees(math.atan2(x_cam, z_cam if in_front else 1.0))
    dtilt = math.degrees(math.atan2(y_cam, z_cam if in_front else 1.0))

    return (dpan, dtilt, in_front)


__all__ = (
    "ORBIT_SPEED_SCALE",
    "camera_tangent_offset",
    "dir_to_pan_tilt",
    "earth_bearing",
    "get_default_orbit",
    "get_satellite_basis",
    "pan_tilt_to_dir",
    "peer_bearing",
)
