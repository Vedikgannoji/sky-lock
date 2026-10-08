"""Search patterns for the SEARCH and REACQUIRE phases.

Pure deterministic functions of elapsed time:
- RasterScan: Boustrophedon sweep across field of regard.
- LocalSpiral: Archimedean spiral expanding outward around a designated center.
"""

from __future__ import annotations

import math

from skylock.core.geometry import wrap_deg


class RasterScan:
    """Boustrophedon horizontal raster scan across a specified field of regard.

    Sweeps alternating left-to-right and right-to-left across rows spaced by
    fov_v * (1 - overlap). Can be centered at any starting position (typically current pointing).
    """

    def __init__(
        self,
        field_of_regard: tuple[float, float],
        fov: float | tuple[float, float],
        overlap: float = 0.2,
        scan_rate: float = 4.0,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize RasterScan.

        Args:
            field_of_regard: (pan_width_deg, tilt_height_deg).
            fov: Vertical FOV deg or (fov_h_deg, fov_v_deg).
            overlap: Fractional row overlap in [0, 1).
            scan_rate: Scanning slew rate in deg/s.
            center: (center_pan_deg, center_tilt_deg) - starting point for search.
        """
        self.center = (float(center[0]), float(center[1]))
        self.pan_width = float(field_of_regard[0])
        self.tilt_height = float(field_of_regard[1])
        self.scan_rate = float(scan_rate)
        self.overlap = float(overlap)

        if isinstance(fov, (tuple, list)):
            self.fov_h = float(fov[0])
            self.fov_v = float(fov[1])
        else:
            self.fov_v = float(fov)
            self.fov_h = self.fov_v  # assume square FOV if only one value given

        # Row spacing accounts for overlap
        step = self.fov_v * (1.0 - self.overlap)
        self._row_spacing = step if step > 0.0 else self.fov_v

        # Number of rows needed to cover the field of regard
        self._num_rows = max(1, math.ceil(self.tilt_height / self._row_spacing) + 1)

        # Time to sweep one row
        self._sweep_time = self.pan_width / self.scan_rate if self.scan_rate > 0.0 else 1.0

        # Total cycle time
        self._cycle_time = self._sweep_time * self._num_rows

    @property
    def field_of_regard(self) -> tuple[float, float]:
        """Field of regard dimensions (pan_width, tilt_height) in degrees."""
        return (self.pan_width, self.tilt_height)

    def recenter(self, new_center: tuple[float, float]) -> None:
        """Update the center point of the raster scan.

        Args:
            new_center: (pan_deg, tilt_deg) new center position.
        """
        self.center = (float(new_center[0]), float(new_center[1]))

    @property
    def row_spacing(self) -> float:
        """Vertical spacing between consecutive raster rows in degrees."""
        return self._row_spacing

    @property
    def num_rows(self) -> int:
        """Total number of scan rows in one cycle."""
        return self._num_rows

    @property
    def sweep_time(self) -> float:
        """Duration of a single horizontal row sweep in seconds."""
        return self._sweep_time

    @property
    def cycle_time(self) -> float:
        """Total duration of a complete raster pattern cycle in seconds."""
        return self._cycle_time

    def setpoint(self, elapsed: float) -> tuple[float, float]:
        """Compute (pan_deg, tilt_deg) setpoint at the given elapsed time.

        The scan starts from the current center and expands outward, scanning
        rows from top to bottom in a boustrophedon pattern.

        Args:
            elapsed: Elapsed time in seconds since pattern start.

        Returns:
            (pan_deg, tilt_deg) in sky frame.
        """
        if elapsed < 0.0:
            elapsed = 0.0

        t_mod = elapsed % self._cycle_time if self._cycle_time > 0.0 else 0.0
        row = int(t_mod / self._sweep_time) if self._sweep_time > 0.0 else 0
        row = min(row, self._num_rows - 1)

        t_in_row = t_mod - row * self._sweep_time
        frac = t_in_row / self._sweep_time if self._sweep_time > 0.0 else 0.0

        # Alternate sweep directions for even/odd rows (boustrophedon)
        pan_min = self.center[0] - self.pan_width / 2.0
        pan_max = self.center[0] + self.pan_width / 2.0
        pan = (
            pan_min + frac * self.pan_width
            if row % 2 == 0
            else pan_max - frac * self.pan_width
        )

        if self._num_rows == 1:
            tilt = self.center[1]
        else:
            tilt_min = self.center[1] - (self._num_rows - 1) * self._row_spacing / 2.0
            tilt = tilt_min + row * self._row_spacing

        return (wrap_deg(pan), tilt)

    def __call__(self, elapsed: float) -> tuple[float, float]:
        """Callable interface matching setpoint(elapsed)."""
        return self.setpoint(elapsed)


class LocalSpiral:
    """Expanding Archimedean spiral search pattern for REACQUIRE.

    Expands outward from center point, bounded strictly within max_radius.
    """

    def __init__(
        self,
        center: tuple[float, float],
        max_radius: float,
        spacing: float,
        rate: float = 4.0,
    ) -> None:
        """Initialize LocalSpiral.

        Args:
            center: (center_pan_deg, center_tilt_deg).
            max_radius: Maximum search radius in degrees.
            spacing: Radial spacing per full 360-degree turn in degrees.
            rate: Scanning velocity in deg/s.
        """
        self.center = (float(center[0]), float(center[1]))
        self.max_radius = max(0.0, float(max_radius))
        self.spacing = max(1e-4, float(spacing))
        self.rate = float(rate)

        # Growth rate: degrees of radius per radian of rotation
        self._growth = self.spacing / (2.0 * math.pi)

        # Angular rate: radians per second
        if self.rate > 0.0 and self.spacing > 0.0:
            self._omega = (self.rate / self.spacing) * (2.0 * math.pi)
        else:
            self._omega = 1.0

    @property
    def growth(self) -> float:
        """Radial growth per radian in degrees."""
        return self._growth

    def is_done(self, elapsed: float) -> bool:
        """Return True if spiral has reached or exceeded max_radius."""
        if elapsed <= 0.0:
            return False
        return (self._growth * self._omega * elapsed) >= self.max_radius

    def setpoint(self, elapsed: float) -> tuple[float, float]:
        """Compute (pan_deg, tilt_deg) setpoint at the given elapsed time.

        Args:
            elapsed: Elapsed time in seconds since pattern start.

        Returns:
            (pan_deg, tilt_deg) strictly within max_radius of center.
        """
        if elapsed <= 0.0 or self.max_radius <= 0.0:
            return (wrap_deg(self.center[0]), self.center[1])

        theta = self._omega * elapsed
        radius = min(self.max_radius, self._growth * theta)

        pan_offset = radius * math.cos(theta)
        tilt_offset = radius * math.sin(theta)

        return (wrap_deg(self.center[0] + pan_offset), self.center[1] + tilt_offset)

    def __call__(self, elapsed: float) -> tuple[float, float]:
        """Callable interface matching setpoint(elapsed)."""
        return self.setpoint(elapsed)


__all__ = ("LocalSpiral", "RasterScan")
