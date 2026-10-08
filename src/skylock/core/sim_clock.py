"""Thread-safe lazy simulation clock for SkyLock orbital tracking."""

from __future__ import annotations

import threading
import time
from typing import Callable


class SimClock:
    """Thread-safe monotonic simulation clock with lazy calculation and pause/resume/reset.

    Guarantees:
    - now() is computed lazily from time_fn() * speed.
    - No background timer threads.
    - Full thread-safety across worker threads and UI threads.
    - Deterministic pause/resume/reset mechanics.
    """

    def __init__(
        self,
        speed: float = 1.0,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._time_fn = time_fn if time_fn is not None else time.monotonic
        self._speed = max(0.0, float(speed))
        self._sim_time_base: float = 0.0
        self._mono_base: float = self._time_fn()
        self._paused: bool = False
        self._paused_time: float = 0.0

    def now(self) -> float:
        """Return current simulation time in seconds (lazy computation)."""
        with self._lock:
            if self._paused:
                return self._paused_time
            now_mono = self._time_fn()
            return self._sim_time_base + (now_mono - self._mono_base) * self._speed

    def pause(self) -> None:
        """Pause the clock."""
        with self._lock:
            if not self._paused:
                now_mono = self._time_fn()
                self._paused_time = self._sim_time_base + (now_mono - self._mono_base) * self._speed
                self._paused = True

    def resume(self) -> None:
        """Resume the clock."""
        with self._lock:
            if self._paused:
                self._sim_time_base = self._paused_time
                self._mono_base = self._time_fn()
                self._paused = False

    def toggle_pause(self) -> bool:
        """Toggle paused state and return new is_paused state."""
        with self._lock:
            now_mono = self._time_fn()
            if self._paused:
                self._sim_time_base = self._paused_time
                self._mono_base = now_mono
                self._paused = False
            else:
                self._paused_time = self._sim_time_base + (now_mono - self._mono_base) * self._speed
                self._paused = True
            return self._paused

    @property
    def is_paused(self) -> bool:
        """Whether the clock is currently paused."""
        with self._lock:
            return self._paused

    def reset(self, t: float = 0.0) -> None:
        """Reset simulation time to t seconds."""
        with self._lock:
            self._sim_time_base = float(t)
            self._paused_time = float(t)
            self._mono_base = self._time_fn()

    def set_speed(self, speed: float) -> None:
        """Set simulation speed multiplier."""
        with self._lock:
            now_mono = self._time_fn()
            if not self._paused:
                self._sim_time_base += (now_mono - self._mono_base) * self._speed
                self._mono_base = now_mono
            self._speed = max(0.0, float(speed))

    def get_speed(self) -> float:
        """Get simulation speed multiplier."""
        with self._lock:
            return self._speed

    def set_time(self, t: float) -> None:
        """Directly set current simulation time."""
        with self._lock:
            self._sim_time_base = float(t)
            self._paused_time = float(t)
            self._mono_base = self._time_fn()


_GLOBAL_SIM_CLOCK: SimClock | None = None
_GLOBAL_CLOCK_LOCK = threading.Lock()


def get_shared_clock() -> SimClock:
    """Return the global shared simulation clock singleton."""
    global _GLOBAL_SIM_CLOCK
    with _GLOBAL_CLOCK_LOCK:
        if _GLOBAL_SIM_CLOCK is None:
            _GLOBAL_SIM_CLOCK = SimClock()
        return _GLOBAL_SIM_CLOCK


def reset_shared_clock(t: float = 0.0, speed: float = 1.0) -> SimClock:
    """Reset and reconfigure the global shared simulation clock."""
    global _GLOBAL_SIM_CLOCK
    with _GLOBAL_CLOCK_LOCK:
        if _GLOBAL_SIM_CLOCK is None:
            _GLOBAL_SIM_CLOCK = SimClock(speed=speed)
        _GLOBAL_SIM_CLOCK.reset(t)
        _GLOBAL_SIM_CLOCK.set_speed(speed)
        return _GLOBAL_SIM_CLOCK


__all__ = ("SimClock", "get_shared_clock", "reset_shared_clock")
