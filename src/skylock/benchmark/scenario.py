"""Benchmark scenario definition and configuration application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skylock.config.io import override
from skylock.config.models import SkyLockConfig


@dataclass(frozen=True, slots=True)
class Scenario:
    """Deterministic benchmark scenario descriptor.

    Attributes:
        id: Unique human-readable identifier (e.g. 'S01_line_clean').
        description: Multi-line explanation of what the scenario tests.
        overrides: Flat dot-path config overrides applied via config.io.override.
        duration_s: Simulation duration in seconds.
        seeds: Default seed set for reproducibility.
        input_kind: 'simulation' or 'mp4'.
        mp4_path: Path to MP4 file (only for mp4 scenarios, None otherwise).
        tags: Classification tags for filtering.
    """

    id: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 6.0
    seeds: tuple[int, ...] = (42,)
    input_kind: str = "simulation"
    mp4_path: str | None = None
    tags: tuple[str, ...] = ()

    def apply(self, base_cfg: SkyLockConfig, seed: int) -> SkyLockConfig:
        """Create a concrete SkyLockConfig for this scenario with the given seed.

        Applies scenario overrides and injects seed via config.io.override.
        All randomness flows through SkyLockConfig.seed -> derive_rng.

        Args:
            base_cfg: Base configuration to overlay scenario overrides on.
            seed: Explicit deterministic seed (no time-based fallback).

        Returns:
            Fully validated SkyLockConfig ready for Session construction.
        """
        combined: dict[str, Any] = {}
        combined.update(self.overrides)
        combined["seed"] = seed
        if self.input_kind == "mp4" and self.mp4_path is not None:
            combined["input.kind"] = "mp4"
            combined["input.mp4_path"] = self.mp4_path
        elif self.input_kind == "orbital":
            combined["input.kind"] = "orbital"
        elif self.input_kind == "simulation":
            combined["input.kind"] = "simulation"
        return override(base_cfg, combined)


__all__ = ("Scenario",)
