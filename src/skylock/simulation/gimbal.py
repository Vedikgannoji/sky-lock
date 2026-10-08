"""Virtual gimbal actuator plant simulation with acceleration and slew rate limits."""

from __future__ import annotations

import math

from skylock.config.models import GimbalConfig
from skylock.core.interfaces import GimbalPlant
from skylock.core.types import ControlCommand, Pointing


def _sign(x: float) -> float:
    """Return sign of x as float (-1.0, 0.0, 1.0)."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


class VirtualGimbal(GimbalPlant):
    """Two-axis pan/tilt gimbal plant with acceleration-limited slew dynamics.

    Implements the GimbalPlant protocol.
    Enforces:
    - Slew velocity limits: |v| <= min(slew_rate, max_slew_rate, 10.0 deg/s)
    - Acceleration limits: |a| <= accel_deg_s2
    - Joint travel limits with velocity zeroing on contact and at_limit flags
    - Substep numerical integration for continuous trajectory fidelity
    """

    def __init__(self, config: GimbalConfig | None = None) -> None:
        """Initialize VirtualGimbal.

        Args:
            config: Physical actuator and joint configuration.
        """
        self.config = config if config is not None else GimbalConfig()

        self._pan: float = float(self.config.initial[0])
        self._tilt: float = float(self.config.initial[1])
        self._pan_rate: float = 0.0
        self._tilt_rate: float = 0.0

        self._mode: str = "RATE"  # "RATE" or "POSITION"
        self._target_pan: float = self._pan
        self._target_tilt: float = self._tilt
        self._cmd_pan_rate: float = 0.0
        self._cmd_tilt_rate: float = 0.0

        self._at_limit_pan: bool = False
        self._at_limit_tilt: bool = False

        self._clamp_to_limits()

    @property
    def pointing(self) -> Pointing:
        """Current gimbal pointing angles in degrees."""
        return Pointing(pan_deg=self._pan, tilt_deg=self._tilt)

    @property
    def pan_rate_deg_s(self) -> float:
        """Current pan angular velocity in deg/s."""
        return self._pan_rate

    @property
    def tilt_rate_deg_s(self) -> float:
        """Current tilt angular velocity in deg/s."""
        return self._tilt_rate

    @property
    def at_limit_pan(self) -> bool:
        """Whether pan axis is currently resting against a travel limit."""
        return self._at_limit_pan

    @property
    def at_limit_tilt(self) -> bool:
        """Whether tilt axis is currently resting against a travel limit."""
        return self._at_limit_tilt

    @property
    def at_limit(self) -> bool:
        """Whether either gimbal axis is resting against a travel limit."""
        return self._at_limit_pan or self._at_limit_tilt

    @property
    def max_slew_rate(self) -> float:
        """Effective maximum velocity magnitude in deg/s."""
        return min(self.config.slew_rate_deg_s, self.config.max_slew_rate_deg_s, 10.0)

    def reset(self) -> None:
        """Restore initial pointing angles, zero velocities, and clear limit flags."""
        self._pan = float(self.config.initial[0])
        self._tilt = float(self.config.initial[1])
        self._pan_rate = 0.0
        self._tilt_rate = 0.0
        self._mode = "RATE"
        self._target_pan = self._pan
        self._target_tilt = self._tilt
        self._cmd_pan_rate = 0.0
        self._cmd_tilt_rate = 0.0
        self._at_limit_pan = False
        self._at_limit_tilt = False
        self._clamp_to_limits()

    def set_pointing(self, pan_deg: float, tilt_deg: float) -> None:
        """Directly set gimbal pointing angles (e.g. at session start to cue)."""
        self._pan = float(pan_deg)
        self._tilt = float(tilt_deg)
        self._target_pan = self._pan
        self._target_tilt = self._tilt
        self._pan_rate = 0.0
        self._tilt_rate = 0.0
        self._clamp_to_limits()

    def command(self, cmd: ControlCommand, dt: float) -> None:
        """Apply rate command and integrate gimbal dynamics across timestep dt.

        Args:
            cmd: Commanded motor angular rates.
            dt: Timestep duration in seconds.
        """
        self._mode = "RATE"
        self._cmd_pan_rate = cmd.pan_rate_deg_s
        self._cmd_tilt_rate = cmd.tilt_rate_deg_s
        self.step(dt)

    def command_position(self, pan_deg: float, tilt_deg: float, dt: float) -> None:
        """Apply position setpoint command and integrate dynamics across timestep dt.

        Args:
            pan_deg: Target pan angle in degrees.
            tilt_deg: Target tilt angle in degrees.
            dt: Timestep duration in seconds.
        """
        self._mode = "POSITION"
        pan_min, pan_max = self.config.pan_limit_deg
        tilt_min, tilt_max = self.config.tilt_limit_deg
        self._target_pan = max(pan_min, min(pan_max, float(pan_deg)))
        self._target_tilt = max(tilt_min, min(tilt_max, float(tilt_deg)))
        self.step(dt)

    def step(self, dt: float) -> None:
        """Advance actuator dynamics across timestep dt using substep integration.

        Args:
            dt: Timestep duration in seconds.
        """
        if dt <= 0.0:
            return

        vmax = self.max_slew_rate
        accel = self.config.accel_deg_s2
        substeps = max(1, self.config.substeps)
        sub_dt = dt / substeps
        max_dv = accel * sub_dt

        pan_min, pan_max = self.config.pan_limit_deg
        tilt_min, tilt_max = self.config.tilt_limit_deg

        pan_is_wrapped = (pan_min <= -180.0 and pan_max >= 180.0)

        for _ in range(substeps):
            # Compute desired velocity for PAN
            if self._mode == "POSITION":
                if pan_is_wrapped:
                    from skylock.core.geometry import angular_diff_deg
                    err_pan = angular_diff_deg(self._target_pan, self._pan)
                else:
                    err_pan = self._target_pan - self._pan
                abs_err_pan = abs(err_pan)
                if abs_err_pan < 0.005 and abs(self._pan_rate) < 0.2:
                    self._pan = self._target_pan
                    self._pan_rate = 0.0
                    desired_pan_vel = 0.0
                else:
                    stopping_v = math.sqrt(max(0.0, 2.0 * accel * abs_err_pan))
                    desired_pan_vel = _sign(err_pan) * min(vmax, stopping_v)
            else:
                desired_pan_vel = max(-vmax, min(vmax, self._cmd_pan_rate))

            # Accelerate towards desired velocity
            dv_pan = desired_pan_vel - self._pan_rate
            self._pan_rate += max(-max_dv, min(max_dv, dv_pan))
            self._pan_rate = max(-vmax, min(vmax, self._pan_rate))
            self._pan += self._pan_rate * sub_dt

            # Pan joint limits
            if pan_is_wrapped:
                from skylock.core.geometry import wrap_deg
                self._pan = wrap_deg(self._pan)
                self._at_limit_pan = False
            elif self._pan >= pan_max:
                self._pan = pan_max
                self._at_limit_pan = True
                if self._pan_rate > 0.0:
                    self._pan_rate = 0.0
            elif self._pan <= pan_min:
                self._pan = pan_min
                self._at_limit_pan = True
                if self._pan_rate < 0.0:
                    self._pan_rate = 0.0
            else:
                self._at_limit_pan = False

            # Compute desired velocity for TILT
            if self._mode == "POSITION":
                err_tilt = self._target_tilt - self._tilt
                abs_err_tilt = abs(err_tilt)
                if abs_err_tilt < 0.005 and abs(self._tilt_rate) < 0.2:
                    self._tilt = self._target_tilt
                    self._tilt_rate = 0.0
                    desired_tilt_vel = 0.0
                else:
                    stopping_v = math.sqrt(max(0.0, 2.0 * accel * abs_err_tilt))
                    desired_tilt_vel = _sign(err_tilt) * min(vmax, stopping_v)
            else:
                desired_tilt_vel = max(-vmax, min(vmax, self._cmd_tilt_rate))

            # Accelerate towards desired velocity
            dv_tilt = desired_tilt_vel - self._tilt_rate
            self._tilt_rate += max(-max_dv, min(max_dv, dv_tilt))
            self._tilt_rate = max(-vmax, min(vmax, self._tilt_rate))
            self._tilt += self._tilt_rate * sub_dt

            # Tilt joint limits
            if self._tilt >= tilt_max:
                self._tilt = tilt_max
                self._at_limit_tilt = True
                if self._tilt_rate > 0.0:
                    self._tilt_rate = 0.0
            elif self._tilt <= tilt_min:
                self._tilt = tilt_min
                self._at_limit_tilt = True
                if self._tilt_rate < 0.0:
                    self._tilt_rate = 0.0
            else:
                self._at_limit_tilt = False

    def _clamp_to_limits(self) -> None:
        """Enforce physical joint limits on current angles."""
        pan_min, pan_max = self.config.pan_limit_deg
        tilt_min, tilt_max = self.config.tilt_limit_deg

        pan_is_wrapped = (pan_min <= -180.0 and pan_max >= 180.0)

        if pan_is_wrapped:
            from skylock.core.geometry import wrap_deg
            self._pan = wrap_deg(self._pan)
            self._at_limit_pan = False
        elif self._pan >= pan_max:
            self._pan = pan_max
            self._at_limit_pan = True
        elif self._pan <= pan_min:
            self._pan = pan_min
            self._at_limit_pan = True
        else:
            self._at_limit_pan = False

        if self._tilt >= tilt_max:
            self._tilt = tilt_max
            self._at_limit_tilt = True
        elif self._tilt <= tilt_min:
            self._tilt = tilt_min
            self._at_limit_tilt = True
        else:
            self._at_limit_tilt = False


__all__ = ("VirtualGimbal",)
