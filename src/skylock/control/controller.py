"""PointingController calculating commanded gimbal motor angular rates."""

from __future__ import annotations

from skylock.config.models import CameraConfig, ControlConfig
from skylock.control.pid import PID
from skylock.core.enums import ControlIntentMode, ControlMode
from skylock.core.geometry import angular_diff_deg
from skylock.core.los import (
    OrbitParams,
    calculate_look_angles,
    check_line_of_sight,
    get_satellite_position,
)
from skylock.core.types import ControlCommand, ControlIntent, Pointing, TargetEstimate


class PointingController:
    """Line-of-sight tracking controller converting ControlIntent to motor rate commands.

    Strict contract:
    - Never accepts, references, or imports GroundTruthSample.
    - Operates purely on ControlIntent, TargetEstimate, and Pointing telemetry.
    """

    def __init__(
        self,
        control_cfg: ControlConfig | None = None,
        camera_cfg: CameraConfig | None = None,
        max_slew_rate_deg_s: float = 10.0,
        sat_id: str = "s1",
        orbit_params: OrbitParams | None = None,
        mode: ControlMode | str | None = None,
    ) -> None:
        """Initialize PointingController.

        Args:
            control_cfg: PID gains and controller settings.
            camera_cfg: Camera parameters for IFOV and slew calculation.
            max_slew_rate_deg_s: Upper velocity cap in deg/s (hard ceiling <= 10.0).
            sat_id: Identifier of the satellite hosting this gimbal ('s1' or 's2').
            orbit_params: Optional orbital parameters for Earth boresight calculation.
            mode: Initial control mode (AUTO, MANUAL, or EARTH/EARTH_BORESIGHT).
        """
        self.control_cfg = control_cfg if control_cfg is not None else ControlConfig()
        self.camera_cfg = camera_cfg if camera_cfg is not None else CameraConfig()
        self.max_slew_rate_deg_s = min(10.0, float(max_slew_rate_deg_s))
        self._sat_id = sat_id.lower().replace("-", "")
        self._orbit_params = orbit_params

        self.pid_pan = PID(
            kp=self.control_cfg.kp,
            ki=self.control_cfg.ki,
            kd=self.control_cfg.kd,
            d_alpha=self.control_cfg.d_filter_alpha,
            i_clamp=self.control_cfg.integral_clamp,
        )
        self.pid_tilt = PID(
            kp=self.control_cfg.kp,
            ki=self.control_cfg.ki,
            kd=self.control_cfg.kd,
            d_alpha=self.control_cfg.d_filter_alpha,
            i_clamp=self.control_cfg.integral_clamp,
        )

        if mode is not None:
            self.set_mode(mode)
        else:
            self._mode = (
                ControlMode.MANUAL if self.control_cfg.mode == "MANUAL" else ControlMode.AUTO
            )
        self._manual_pan_rate: float = 0.0
        self._manual_tilt_rate: float = 0.0
        self._current_time_s: float = 0.0
        self._custom_sat_pos: tuple[float, float, float] | None = None

    @property
    def sat_id(self) -> str:
        """Satellite hosting this gimbal ('s1' or 's2')."""
        return self._sat_id

    @sat_id.setter
    def sat_id(self, value: str) -> None:
        self._sat_id = str(value).lower().replace("-", "")

    @property
    def orbit_params(self) -> OrbitParams | None:
        """Orbital parameters used for Earth boresight geometric calculations."""
        return self._orbit_params

    @orbit_params.setter
    def orbit_params(self, params: OrbitParams | None) -> None:
        self._orbit_params = params

    @property
    def current_time_s(self) -> float:
        """Current internal simulation time in seconds."""
        return self._current_time_s

    @current_time_s.setter
    def current_time_s(self, value: float) -> None:
        self._current_time_s = float(value)

    def set_sim_time(self, t_s: float) -> None:
        """Set the simulation time for orbital tracking calculations."""
        self._current_time_s = float(t_s)

    def set_satellite_pos(self, pos: tuple[float, float, float] | None) -> None:
        """Explicitly override the satellite's position relative to Earth (optional)."""
        self._custom_sat_pos = pos

    @property
    def mode(self) -> ControlMode:
        """Current operating mode (AUTO, MANUAL, or EARTH)."""
        return self._mode

    def set_mode(self, mode: ControlMode | str) -> None:
        """Set operating mode (AUTO, MANUAL, or EARTH)."""
        if isinstance(mode, str):
            m = mode.upper()
            if m == "MANUAL":
                self._mode = ControlMode.MANUAL
            elif m in ("EARTH", "EARTH_BORESIGHT"):
                self._mode = ControlMode.EARTH
            else:
                self._mode = ControlMode.AUTO
        else:
            if mode in (ControlMode.EARTH, ControlMode.EARTH_BORESIGHT):
                self._mode = ControlMode.EARTH
            else:
                self._mode = mode

    @property
    def manual_pan_rate_deg_s(self) -> float:
        """Manual pan slew rate in deg/s."""
        return self._manual_pan_rate

    @property
    def manual_tilt_rate_deg_s(self) -> float:
        """Manual tilt slew rate in deg/s."""
        return self._manual_tilt_rate

    def set_manual_rate(self, pan_rate_deg_s: float, tilt_rate_deg_s: float) -> None:
        """Set manual angular rates to be executed when mode is MANUAL."""
        self._manual_pan_rate = float(pan_rate_deg_s)
        self._manual_tilt_rate = float(tilt_rate_deg_s)

    def reset(self) -> None:
        """Reset internal PID controllers, time, and manual rates."""
        self.pid_pan.reset()
        self.pid_tilt.reset()
        self._manual_pan_rate = 0.0
        self._manual_tilt_rate = 0.0
        self._current_time_s = 0.0
        self._custom_sat_pos = None
        self._mode = (
            ControlMode.MANUAL if self.control_cfg.mode == "MANUAL" else ControlMode.AUTO
        )

    def step(
        self,
        intent: ControlIntent,
        estimate: TargetEstimate | None,
        pointing: Pointing | None,
        dt: float,
    ) -> ControlCommand:
        """Calculate commanded gimbal angular rates for the current frame step.

        Args:
            intent: Desired control intent from tracking state machine.
            estimate: Filtered target estimate or None.
            pointing: Current gimbal pointing telemetry or None.
            dt: Frame timestep duration in seconds.

        Returns:
            ControlCommand containing pan and tilt rates in deg/s.
        """
        max_slew = self.max_slew_rate_deg_s
        self._current_time_s += max(0.0, dt)

        if self._mode == ControlMode.MANUAL:
            pan_cmd = max(-max_slew, min(max_slew, self._manual_pan_rate))
            tilt_cmd = max(-max_slew, min(max_slew, self._manual_tilt_rate))
            return ControlCommand(pan_rate_deg_s=pan_cmd, tilt_rate_deg_s=tilt_cmd)

        if self._mode in (ControlMode.EARTH, ControlMode.EARTH_BORESIGHT):
            self.pid_pan.reset()
            self.pid_tilt.reset()

            if self._custom_sat_pos is not None:
                target_pan, target_tilt, _ = calculate_look_angles(
                    self._custom_sat_pos, (0.0, 0.0, 0.0)
                )
            else:
                from skylock.core.orbital_world import earth_bearing

                target_pan, target_tilt = earth_bearing(self._sat_id, self._current_time_s)

            pt = pointing if pointing is not None else Pointing(0.0, 0.0)
            err_pan = angular_diff_deg(target_pan, pt.pan_deg)
            err_tilt = target_tilt - pt.tilt_deg

            k_goto = self.control_cfg.kp if self.control_cfg.kp > 0 else 5.0
            pan_cmd = max(-max_slew, min(max_slew, k_goto * err_pan))
            tilt_cmd = max(-max_slew, min(max_slew, k_goto * err_tilt))
            return ControlCommand(pan_rate_deg_s=pan_cmd, tilt_rate_deg_s=tilt_cmd)

        # In AUTO mode, check geometric line-of-sight between observer and target
        if self._mode == ControlMode.AUTO:
            target_sat = "s2" if self._sat_id in ("s1", "1") else "s1"
            obs_pos = (
                self._custom_sat_pos
                if self._custom_sat_pos is not None
                else get_satellite_position(self._sat_id, self._current_time_s, self._orbit_params)
            )
            target_pos = get_satellite_position(target_sat, self._current_time_s)
            if not check_line_of_sight(obs_pos, target_pos):
                # While LOS is blocked by Earth: stop actively searching or slewing; hold pointing direction
                self.pid_pan.reset()
                self.pid_tilt.reset()
                return ControlCommand(pan_rate_deg_s=0.0, tilt_rate_deg_s=0.0)

        if intent.mode == ControlIntentMode.HOLD or dt <= 0.0:
            self.pid_pan.reset()
            self.pid_tilt.reset()
            return ControlCommand(pan_rate_deg_s=0.0, tilt_rate_deg_s=0.0)

        if intent.mode == ControlIntentMode.GOTO:
            pt = pointing if pointing is not None else Pointing(0.0, 0.0)
            err_pan = angular_diff_deg(intent.setpoint_pan_deg, pt.pan_deg)
            err_tilt = intent.setpoint_tilt_deg - pt.tilt_deg

            # Proportional slew rate toward setpoint
            k_goto = self.control_cfg.kp
            pan_cmd = max(-max_slew, min(max_slew, k_goto * err_pan))
            tilt_cmd = max(-max_slew, min(max_slew, k_goto * err_tilt))
            return ControlCommand(pan_rate_deg_s=pan_cmd, tilt_rate_deg_s=tilt_cmd)

        # intent.mode == ControlIntentMode.TRACK
        err_x_px, err_y_px = (
            intent.image_error_px if intent.image_error_px is not None else (0.0, 0.0)
        )

        # Deadband thresholding
        deadband = self.control_cfg.deadband_px
        if abs(err_x_px) < deadband:
            err_x_px = 0.0
        if abs(err_y_px) < deadband:
            err_y_px = 0.0

        # Convert image error to angular angle errors
        ifov_h = self.camera_cfg.fov_h_deg / self.camera_cfg.width
        ifov_v = self.camera_cfg.fov_v_deg / self.camera_cfg.height
        err_pan_deg = err_x_px * ifov_h
        err_tilt_deg = -err_y_px * ifov_v

        # Feed-forward angular velocity from target kinematic estimate
        ff_pan = self.control_cfg.kff * estimate.pan_rate if estimate is not None else 0.0
        ff_tilt = self.control_cfg.kff * estimate.tilt_rate if estimate is not None else 0.0

        # PID integration
        u_pan = self.pid_pan.step(err_pan_deg, dt, limit=max_slew) + ff_pan
        u_tilt = self.pid_tilt.step(err_tilt_deg, dt, limit=max_slew) + ff_tilt

        pan_cmd = max(-max_slew, min(max_slew, u_pan))
        tilt_cmd = max(-max_slew, min(max_slew, u_tilt))

        return ControlCommand(pan_rate_deg_s=pan_cmd, tilt_rate_deg_s=tilt_cmd)


__all__ = ("PointingController",)
